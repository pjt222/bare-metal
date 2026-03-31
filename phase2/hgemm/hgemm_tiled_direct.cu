/*
 * hgemm_tiled_direct.cu — Same as hgemm_tiled.cu but with direct global store
 *
 * Removes the smem epilogue (saves 8 KB, eliminates syncwarp overhead).
 * Uses wmma::store_matrix_sync directly to global memory.
 *
 * Hypothesis: the smem epilogue hurts rather than helps for HGEMM because
 * FP32 accumulators can be stored directly without dequantization (unlike IGEMM).
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o hgemm_tiled_direct.sm_86.cubin hgemm_tiled_direct.cu
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
#define BLOCK_SIZE  (NUM_WARPS * WARP_SIZE)

#define WARPS_Y 4
#define WARPS_X 2
#define WARP_TILES_M 2
#define WARP_TILES_N 4

#define CP_ASYNC_BYTES  8
#define ELEMS_PER_COPY  (CP_ASYNC_BYTES / 2)
#define CP_ELEMS_A      (BM * BK / BLOCK_SIZE * 2 / CP_ASYNC_BYTES)
#define CP_ELEMS_B      (BK * BN / BLOCK_SIZE * 2 / CP_ASYNC_BYTES)

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void hgemm_tiled_direct(
    const __half * __restrict__ matrix_a,
    const __half * __restrict__ matrix_b,
    float        * __restrict__ matrix_c,
    int M, int N, int K
) {
    // No epilogue smem — saves 8 KB (32 KB total vs 40 KB)
    __shared__ __align__(8) __half smem_a[2][BM * BK];
    __shared__ __align__(8) __half smem_b[2][BK * BN];

    int tid     = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int wy      = warp_id / WARPS_X;
    int wx      = warp_id % WARPS_X;

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

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

    // Direct store — no smem epilogue
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
