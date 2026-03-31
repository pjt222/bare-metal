/*
 * hgemm_256x128.cu — 16-warp 256×128 FP16 GEMM, asymmetric tile
 *
 * Raises arithmetic intensity from 64 to 85 ops/byte by widening BM.
 * Uses 8×2 warp grid (each warp covers 32×64 = 2×4 WMMA tiles).
 *
 * Trade-off: 48 KB smem (no padding room under 50 KB cliff).
 * Bank conflicts return but higher AI may compensate.
 *
 * Register budget:
 *   Accumulator: 2×4 = 8 fragments × 8 regs = 64
 *   A fragments: 2 × 8 = 16
 *   B fragments: 4 × 8 = 32
 *   Control: ~36
 *   Total: ~148 → EXCEEDS 128 limit for 2 blocks/SM
 *   → Must use __launch_bounds__(512, 1) = 1 block/SM = 16 warps/SM
 *
 * Smem per block: 48 KB (no padding, under 50 KB cliff)
 *   But at 1 block/SM, could use up to 100 KB — padding IS possible
 *   With padding: 57 KB → still 1 block/SM, fits easily
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o hgemm_256x128.sm_86.cubin hgemm_256x128.cu
 */

#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BM 256
#define BN 128
#define BK 32

#define NUM_WARPS   16
#define WARP_SIZE   32
#define BLOCK_SIZE  (NUM_WARPS * WARP_SIZE)  // 512

// 8×2 warp grid
#define WARPS_Y 8
#define WARPS_X 2

// Each warp covers 32×64 = 2×4 WMMA tiles
#define WARP_TILES_M 2
#define WARP_TILES_N 4

// No padding — 48 KB is the static smem limit on sm_86
// Bank conflicts return but higher AI (85 vs 64 ops/byte) may compensate
#define STRIDE_A BK    // 32
#define STRIDE_B BN    // 128

// cp.async: 16 bytes per call = 8 __half values
// A buffer: 256 × 32 × 2 = 16384 bytes → 16384/512/16 = 2 calls per thread
// B buffer: 32 × 128 × 2 = 8192 bytes → 8192/512/16 = 1 call per thread
#define CP_ASYNC_BYTES  16
#define ELEMS_PER_COPY  (CP_ASYNC_BYTES / 2)
#define CP_ELEMS_A      (BM * BK / BLOCK_SIZE * 2 / CP_ASYNC_BYTES)   // 2
#define CP_ELEMS_B      (BK * BN / BLOCK_SIZE * 2 / CP_ASYNC_BYTES)   // 1

// 1 block/SM — 2×4 warp tiles need ~148 regs (exceeds 128 for 2-block)
extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void hgemm_256x128(
    const __half * __restrict__ matrix_a,
    const __half * __restrict__ matrix_b,
    float        * __restrict__ matrix_c,
    int M, int N, int K
) {
    __shared__ __align__(16) __half smem_a[2][BM * STRIDE_A];  // 32 KB
    __shared__ __align__(16) __half smem_b[2][BK * STRIDE_B];  // 16 KB
    // Total: 48 KB — at the static smem limit, 1 block/SM

    int tid     = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int wy      = warp_id / WARPS_X;   // 0..7
    int wx      = warp_id % WARPS_X;   // 0..1

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    // 2×4 = 8 FP32 accumulator fragments (64 regs)
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        acc[WARP_TILES_M][WARP_TILES_N];
    #pragma unroll
    for (int i = 0; i < WARP_TILES_M; i++)
        #pragma unroll
        for (int j = 0; j < WARP_TILES_N; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);

    int num_tiles = (K + BK - 1) / BK;

    #define LOAD_A_TILE(buf, k_base)                                          \
        _Pragma("unroll")                                                     \
        for (int _i = 0; _i < CP_ELEMS_A; _i++) {                           \
            int _flat = (tid + _i * BLOCK_SIZE) * ELEMS_PER_COPY;           \
            int _row  = _flat / BK;                                          \
            int _col  = _flat % BK;                                          \
            int _soff = _row * STRIDE_A + _col;                             \
            int _grow = block_row + _row;                                    \
            int _gcol = (k_base) + _col;                                    \
            if (_grow < M && _gcol + ELEMS_PER_COPY - 1 < K) {             \
                __pipeline_memcpy_async(                                      \
                    &smem_a[buf][_soff],                                      \
                    &matrix_a[_grow * K + _gcol],                            \
                    CP_ASYNC_BYTES);                                          \
            } else {                                                          \
                for (int _b = 0; _b < ELEMS_PER_COPY; _b++) {              \
                    int _gc = _gcol + _b;                                    \
                    smem_a[buf][_soff + _b] = (_grow < M && _gc < K)        \
                        ? matrix_a[_grow * K + _gc] : __float2half(0.0f);   \
                }                                                             \
            }                                                                 \
        }

    #define LOAD_B_TILE(buf, k_base)                                          \
        _Pragma("unroll")                                                     \
        for (int _i = 0; _i < CP_ELEMS_B; _i++) {                           \
            int _flat = (tid + _i * BLOCK_SIZE) * ELEMS_PER_COPY;           \
            int _row  = _flat / BN;                                          \
            int _col  = _flat % BN;                                          \
            int _soff = _row * STRIDE_B + _col;                             \
            int _grow = (k_base) + _row;                                    \
            int _gcol = block_col + _col;                                    \
            if (_grow < K && _gcol + ELEMS_PER_COPY - 1 < N) {             \
                __pipeline_memcpy_async(                                      \
                    &smem_b[buf][_soff],                                      \
                    &matrix_b[_grow * N + _gcol],                            \
                    CP_ASYNC_BYTES);                                          \
            } else {                                                          \
                for (int _b = 0; _b < ELEMS_PER_COPY; _b++) {              \
                    int _gc = _gcol + _b;                                    \
                    smem_b[buf][_soff + _b] = (_grow < K && _gc < N)        \
                        ? matrix_b[_grow * N + _gc] : __float2half(0.0f);   \
                }                                                             \
            }                                                                 \
        }

    #define COMPUTE_TILE(buf)                                                  \
        _Pragma("unroll")                                                     \
        for (int k_local = 0; k_local < BK; k_local += WMMA_K) {            \
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,          \
                           __half, wmma::row_major> a_frag[WARP_TILES_M];   \
            _Pragma("unroll")                                                 \
            for (int wi = 0; wi < WARP_TILES_M; wi++) {                      \
                int a_row = wy * 32 + wi * WMMA_M;                           \
                wmma::load_matrix_sync(a_frag[wi],                            \
                    &smem_a[buf][a_row * STRIDE_A + k_local], STRIDE_A);     \
            }                                                                  \
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,          \
                           __half, wmma::row_major> b_frag[WARP_TILES_N];   \
            _Pragma("unroll")                                                 \
            for (int wj = 0; wj < WARP_TILES_N; wj++) {                      \
                int b_col = wx * 64 + wj * WMMA_N;                           \
                wmma::load_matrix_sync(b_frag[wj],                            \
                    &smem_b[buf][k_local * STRIDE_B + b_col], STRIDE_B);     \
            }                                                                  \
            _Pragma("unroll")                                                 \
            for (int wi = 0; wi < WARP_TILES_M; wi++)                         \
                _Pragma("unroll")                                             \
                for (int wj = 0; wj < WARP_TILES_N; wj++)                    \
                    wmma::mma_sync(acc[wi][wj], a_frag[wi], b_frag[wj],      \
                                   acc[wi][wj]);                              \
        }

    // Prologue
    LOAD_A_TILE(0, 0);
    LOAD_B_TILE(0, 0);
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    // Main loop
    for (int tile = 0; tile < num_tiles - 1; tile++) {
        int next_k_base = (tile + 1) * BK;
        int cur_buf     = tile & 1;
        int next_buf    = 1 - cur_buf;

        LOAD_A_TILE(next_buf, next_k_base);
        LOAD_B_TILE(next_buf, next_k_base);
        __pipeline_commit();
        COMPUTE_TILE(cur_buf);
        __pipeline_wait_prior(0);
        __syncthreads();
    }

    // Last tile
    {
        int last_buf = (num_tiles - 1) & 1;
        COMPUTE_TILE(last_buf);
    }

    #undef LOAD_A_TILE
    #undef LOAD_B_TILE
    #undef COMPUTE_TILE

    // Direct store to global memory
    #pragma unroll
    for (int wi = 0; wi < WARP_TILES_M; wi++) {
        #pragma unroll
        for (int wj = 0; wj < WARP_TILES_N; wj++) {
            int c_row = block_row + wy * 32 + wi * WMMA_M;
            int c_col = block_col + wx * 64 + wj * WMMA_N;

            if (c_row + WMMA_M > M || c_col + WMMA_N > N) continue;

            wmma::store_matrix_sync(
                &matrix_c[c_row * N + c_col],
                acc[wi][wj], N, wmma::mem_row_major);
        }
    }
}
