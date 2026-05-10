/*
 * hgemm_16warp.cu — 16-warp 128×128 FP16 GEMM targeting 2 blocks/SM
 *
 * Key idea: reduce warp tile from 2×4 to 2×2, use 16 warps (4×4 grid).
 * This cuts accumulator regs from 64 to 32, enabling 2 blocks/SM = 32 warps/SM.
 *
 * Warp layout: 4×4 grid on 128×128 output tile
 *   Each warp covers 32×32 = 2×2 WMMA tiles (16×16 each)
 *   Per K-step: 4 mma_sync per warp → 8 HMMA
 *   Per K-tile (BK=32, 2 K-steps): 8 mma_sync per warp → 16 HMMA
 *   Total HMMA per tile: 16 warps × 16 = 256 (same as 8-warp version)
 *
 * Register budget (target ≤128 for 2 blocks/SM):
 *   Accumulator: 2×2 = 4 fragments × 8 regs = 32
 *   A fragments: 2 × 8 = 16
 *   B fragments: 2 × 8 = 16
 *   Control/pointers: ~36
 *   Total: ~100 (well under 128)
 *
 * Shared memory per block: 32 KB data + direct store = 32 KB
 *   2 blocks × 32 KB = 64 KB < 100 KB max ✓
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o hgemm_16warp.sm_86.cubin hgemm_16warp.cu
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

// Bank-conflict-free padding: offset row stride so LDSM accesses
// hit different banks. Without padding, stride=64B (A) and 256B (B)
// cause 4-way bank conflicts under the 128-byte bank period.
#define PAD_A    8    // smem_a row stride: BK+8 = 40 halfs = 80 bytes
#define PAD_B    8    // smem_b row stride: BN+8 = 136 halfs = 272 bytes
#define STRIDE_A (BK + PAD_A)
#define STRIDE_B (BN + PAD_B)

#define NUM_WARPS   16
#define WARP_SIZE   32
#define BLOCK_SIZE  (NUM_WARPS * WARP_SIZE)  // 512 threads

// 4×4 warp grid
#define WARPS_Y 4
#define WARPS_X 4

// Each warp covers 32×32 = 2×2 WMMA tiles
#define WARP_TILES_M 2
#define WARP_TILES_N 2

// cp.async: 16 bytes per call = 8 __half values (LDGSTS.E.128)
// With 512 threads: 8192 / 512 / 16 = 1 call per thread for each of A and B
#define CP_ASYNC_BYTES  16
#define ELEMS_PER_COPY  (CP_ASYNC_BYTES / 2)
#define CP_ELEMS_A      (BM * BK / BLOCK_SIZE * 2 / CP_ASYNC_BYTES)   // 1
#define CP_ELEMS_B      (BK * BN / BLOCK_SIZE * 2 / CP_ASYNC_BYTES)   // 1

// 2 blocks/SM — requires ≤128 regs/thread, ≤50 KB smem/block
extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 2)
void hgemm_16warp(
    const __half * __restrict__ matrix_a,
    const __half * __restrict__ matrix_b,
    float        * __restrict__ matrix_c,
    int M, int N, int K
) {
    __shared__ __align__(16) __half smem_a[2][BM * STRIDE_A];   // 20 KB (padded)
    __shared__ __align__(16) __half smem_b[2][BK * STRIDE_B];  // 17 KB (padded)
    // Total: 37 KB — under 50 KB cliff for 2 blocks/SM

    int tid     = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int wy      = warp_id / WARPS_X;   // 0..3
    int wx      = warp_id % WARPS_X;   // 0..3

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    // 2×2 = 4 FP32 accumulator fragments (32 regs vs 64 for 2×4)
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
                int b_col = wx * 32 + wj * WMMA_N;                           \
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
            int c_col = block_col + wx * 32 + wj * WMMA_N;

            if (c_row + WMMA_M > M || c_col + WMMA_N > N) continue;

            wmma::store_matrix_sync(
                &matrix_c[c_row * N + c_col],
                acc[wi][wj], N, wmma::mem_row_major);
        }
    }
}
