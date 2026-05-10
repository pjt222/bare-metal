/*
 * hgemm_tiled.cu — 8-warp 128×128 FP16 GEMM with cp.async double-buffer
 *
 * Adapts the igemm_8warp.cu pattern for FP16 Tensor Cores (HMMA.16816.F32).
 * Establishes a fair tiled HGEMM baseline for comparison with online-quant IGEMM.
 *
 * Warp layout: 4×2 grid on 128×128 output tile
 *   Each warp covers 32×64 = 2×4 WMMA tiles (16×16 each)
 *   Per K-step (WMMA_K=16): 8 mma_sync per warp → 16 HMMA.16816.F32 SASS
 *   Per K-tile (BK=32, 2 K-steps): 16 mma_sync per warp → 32 HMMA
 *
 * Shared memory: 40 KB total (under 50 KB cliff)
 *   smem_a: 2 × 128 × 32 × 2B = 16 KB (double-buffer, FP16)
 *   smem_b: 2 × 32 × 128 × 2B = 16 KB (double-buffer, FP16)
 *   epilogue: 8 × 16 × 16 × 4B = 8 KB (FP32 coalesced write)
 *
 * cp.async: 8 bytes per call (4 __half values)
 *   4 calls per thread for A, 4 for B = 8 total (matches IGEMM instruction count)
 *
 * Arithmetic intensity: 128×128×32×2 / (128×32 + 32×128) × 2B = 128 ops/byte
 *
 * C = A * B, A: M×K (fp16), B: K×N (fp16), C: M×N (fp32)
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o hgemm_tiled.sm_86.cubin hgemm_tiled.cu
 *   cuobjdump -sass hgemm_tiled.sm_86.cubin | grep HMMA
 */

#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BM 128
#define BN 128
#define BK 32

#define NUM_WARPS   8
#define WARP_SIZE   32
#define BLOCK_SIZE  (NUM_WARPS * WARP_SIZE)  // 256 threads

// 4×2 warp grid: 4 warp-rows × 2 warp-cols
#define WARPS_Y 4
#define WARPS_X 2

// Each warp covers 32×64 = 2×4 WMMA tiles
#define WARP_TILES_M 2   // 32 / 16
#define WARP_TILES_N 4   // 64 / 16

// cp.async: 8 bytes per call = 4 __half values
// A buffer: BM × BK × 2 bytes = 8192 bytes per buffer
// B buffer: BK × BN × 2 bytes = 8192 bytes per buffer
// Calls per thread: 8192 / 256 / 8 = 4 for each of A and B
#define CP_ASYNC_BYTES  8
#define ELEMS_PER_COPY  (CP_ASYNC_BYTES / 2)   // 4 __half values per cp.async
#define CP_ELEMS_A      (BM * BK / BLOCK_SIZE * 2 / CP_ASYNC_BYTES)   // 4
#define CP_ELEMS_B      (BK * BN / BLOCK_SIZE * 2 / CP_ASYNC_BYTES)   // 4

// 1 block/SM — accumulator alone needs 64 regs (8 fragments × 8 regs)
extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void hgemm_tiled(
    const __half * __restrict__ matrix_a,   // M×K row-major (fp16)
    const __half * __restrict__ matrix_b,   // K×N row-major (fp16)
    float        * __restrict__ matrix_c,   // M×N row-major (fp32)
    int M, int N, int K
) {
    // Double-buffer for cp.async K-tiles (8-byte aligned for LDGSTS.64)
    __shared__ __align__(8) __half smem_a[2][BM * BK];   // 2 × 4096 halfs = 16 KB
    __shared__ __align__(8) __half smem_b[2][BK * BN];   // 2 × 4096 halfs = 16 KB

    // Epilogue: one 16×16 float tile per warp for coalesced global writes
    __shared__ float epilogue_tile[NUM_WARPS][WMMA_M * WMMA_N];  // 8 KB

    int tid     = threadIdx.x;    // 1D block: 256 threads
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int wy      = warp_id / WARPS_X;   // 0..3
    int wx      = warp_id % WARPS_X;   // 0..1

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    // Each warp maintains 2×4 = 8 FP32 accumulator fragments
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        acc[WARP_TILES_M][WARP_TILES_N];
    #pragma unroll
    for (int i = 0; i < WARP_TILES_M; i++)
        #pragma unroll
        for (int j = 0; j < WARP_TILES_N; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);

    int num_tiles = (K + BK - 1) / BK;

    // ====================================================================
    // Helper: cp.async load A tile into smem buffer
    //   Maps 256 threads × 4 calls × 4 elems = 4096 elements = BM × BK
    // ====================================================================
    #define LOAD_A_TILE(buf, k_base)                                          \
        _Pragma("unroll")                                                     \
        for (int _i = 0; _i < CP_ELEMS_A; _i++) {                           \
            int _elem = (tid + _i * BLOCK_SIZE) * ELEMS_PER_COPY;           \
            int _row  = _elem / BK;                                          \
            int _col  = _elem % BK;                                          \
            int _grow = block_row + _row;                                    \
            int _gcol = (k_base) + _col;                                    \
            if (_grow < M && _gcol + ELEMS_PER_COPY - 1 < K) {             \
                __pipeline_memcpy_async(                                      \
                    &smem_a[buf][_elem],                                      \
                    &matrix_a[_grow * K + _gcol],                            \
                    CP_ASYNC_BYTES);                                          \
            } else {                                                          \
                for (int _b = 0; _b < ELEMS_PER_COPY; _b++) {              \
                    int _gc = _gcol + _b;                                    \
                    smem_a[buf][_elem + _b] = (_grow < M && _gc < K)        \
                        ? matrix_a[_grow * K + _gc] : __float2half(0.0f);   \
                }                                                             \
            }                                                                 \
        }

    #define LOAD_B_TILE(buf, k_base)                                          \
        _Pragma("unroll")                                                     \
        for (int _i = 0; _i < CP_ELEMS_B; _i++) {                           \
            int _elem = (tid + _i * BLOCK_SIZE) * ELEMS_PER_COPY;           \
            int _row  = _elem / BN;                                          \
            int _col  = _elem % BN;                                          \
            int _grow = (k_base) + _row;                                    \
            int _gcol = block_col + _col;                                    \
            if (_grow < K && _gcol + ELEMS_PER_COPY - 1 < N) {             \
                __pipeline_memcpy_async(                                      \
                    &smem_b[buf][_elem],                                      \
                    &matrix_b[_grow * N + _gcol],                            \
                    CP_ASYNC_BYTES);                                          \
            } else {                                                          \
                for (int _b = 0; _b < ELEMS_PER_COPY; _b++) {              \
                    int _gc = _gcol + _b;                                    \
                    smem_b[buf][_elem + _b] = (_grow < K && _gc < N)        \
                        ? matrix_b[_grow * N + _gc] : __float2half(0.0f);   \
                }                                                             \
            }                                                                 \
        }

    // Compute HMMA on a smem buffer: 2 K-steps × 8 mma_sync = 32 HMMA
    #define COMPUTE_TILE(buf)                                                  \
        _Pragma("unroll")                                                     \
        for (int k_local = 0; k_local < BK; k_local += WMMA_K) {            \
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,          \
                           __half, wmma::row_major> a_frag[WARP_TILES_M];   \
            _Pragma("unroll")                                                 \
            for (int wi = 0; wi < WARP_TILES_M; wi++) {                      \
                int a_row = wy * 32 + wi * WMMA_M;                           \
                wmma::load_matrix_sync(a_frag[wi],                            \
                    &smem_a[buf][a_row * BK + k_local], BK);                 \
            }                                                                  \
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,          \
                           __half, wmma::row_major> b_frag[WARP_TILES_N];   \
            _Pragma("unroll")                                                 \
            for (int wj = 0; wj < WARP_TILES_N; wj++) {                      \
                int b_col = wx * 64 + wj * WMMA_N;                           \
                wmma::load_matrix_sync(b_frag[wj],                            \
                    &smem_b[buf][k_local * BN + b_col], BN);                 \
            }                                                                  \
            _Pragma("unroll")                                                 \
            for (int wi = 0; wi < WARP_TILES_M; wi++)                         \
                _Pragma("unroll")                                             \
                for (int wj = 0; wj < WARP_TILES_N; wj++)                    \
                    wmma::mma_sync(acc[wi][wj], a_frag[wi], b_frag[wj],      \
                                   acc[wi][wj]);                              \
        }

    // ====================================================================
    // Prologue: cp.async tile 0 → buffer 0
    // ====================================================================
    LOAD_A_TILE(0, 0);
    LOAD_B_TILE(0, 0);
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    // ====================================================================
    // Main loop: cp.async(tile+1) overlapped with HMMA(tile)
    // ====================================================================
    for (int tile = 0; tile < num_tiles - 1; tile++) {
        int next_k_base = (tile + 1) * BK;
        int cur_buf     = tile & 1;
        int next_buf    = 1 - cur_buf;

        // Phase 1: Issue cp.async for NEXT tile → next_buf
        LOAD_A_TILE(next_buf, next_k_base);
        LOAD_B_TILE(next_buf, next_k_base);
        __pipeline_commit();

        // Phase 2: Compute HMMA on CURRENT tile (overlaps with LDGSTS)
        COMPUTE_TILE(cur_buf);

        // Phase 3: Wait for cp.async, sync
        __pipeline_wait_prior(0);
        __syncthreads();
    }

    // ====================================================================
    // Epilogue: Compute HMMA on last tile
    // ====================================================================
    {
        int last_buf = (num_tiles - 1) & 1;
        COMPUTE_TILE(last_buf);
    }

    #undef LOAD_A_TILE
    #undef LOAD_B_TILE
    #undef COMPUTE_TILE

    // ====================================================================
    // Store — coalesced epilogue via smem (Insight 17: +3.1%)
    //
    // Store FP32 accumulator to shared memory (row-major), then write
    // to global memory with explicit (row, col) indexing for perfect
    // coalescing.
    // ====================================================================
    #pragma unroll
    for (int wi = 0; wi < WARP_TILES_M; wi++) {
        #pragma unroll
        for (int wj = 0; wj < WARP_TILES_N; wj++) {
            int c_row = block_row + wy * 32 + wi * WMMA_M;
            int c_col = block_col + wx * 64 + wj * WMMA_N;

            if (c_row + WMMA_M > M || c_col + WMMA_N > N) continue;

            // Store FP32 accumulator to warp's epilogue tile
            wmma::store_matrix_sync(
                epilogue_tile[warp_id],
                acc[wi][wj], WMMA_N, wmma::mem_row_major);
            __syncwarp();

            // Coalesced write: each thread handles 8 elements
            for (int elem = lane_id; elem < WMMA_M * WMMA_N; elem += WARP_SIZE) {
                int local_row = elem / WMMA_N;
                int local_col = elem % WMMA_N;

                matrix_c[(c_row + local_row) * N + (c_col + local_col)] =
                    epilogue_tile[warp_id][elem];
            }
            __syncwarp();
        }
    }
}
