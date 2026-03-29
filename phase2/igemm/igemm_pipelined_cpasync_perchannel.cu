/*
 * igemm_pipelined_cpasync_perchannel.cu — Per-channel asymmetric INT8 GEMM
 *
 * Same cp.async double-buffered K-loop as igemm_pipelined_cpasync.cu.
 * Only the epilogue changes: instead of a single dequant_scale, each
 * output channel has its own scale and zero-point.
 *
 * Dequantization per element:
 *   float_out = ((float)acc_int32 - (float)zero_point[channel]) * scale[channel]
 *
 * The per-channel arrays are indexed by the output column (N dimension).
 * To avoid depending on undocumented WMMA fragment layout, we store
 * the INT32 accumulator to shared memory via wmma::store_matrix_sync,
 * then each thread reads back with explicit (row, col) indices.
 *
 * Shared memory: 8 KB (double-buffer) + 4 KB (epilogue tiles) = 12 KB total
 *                Well under the 50 KB cliff.
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o igemm_pipelined_cpasync_perchannel.sm_86.cubin igemm_pipelined_cpasync_perchannel.cu
 */

#include <mma.h>
#include <cuda_pipeline.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BM 64
#define BN 64
#define BK 32

#define NUM_WARPS   4
#define WARP_SIZE   32
#define BLOCK_SIZE  (NUM_WARPS * WARP_SIZE)

#define WARPS_Y 2
#define WARPS_X 2
#define WARP_TILES_Y 2
#define WARP_TILES_N 2

#define CP_ASYNC_SIZE 4
#define CP_ELEMS_A  (BM * BK / BLOCK_SIZE / CP_ASYNC_SIZE)
#define CP_ELEMS_B  (BK * BN / BLOCK_SIZE / CP_ASYNC_SIZE)

extern "C" __global__ __launch_bounds__(BLOCK_SIZE)
void igemm_pipelined_cpasync_perchannel(
    const signed char * __restrict__ matrix_a,
    const signed char * __restrict__ matrix_b,
    float             * __restrict__ matrix_c,
    int M, int N, int K,
    const float * __restrict__ per_channel_scale,
    const int   * __restrict__ per_channel_zero_point
) {
    // Double-buffer for cp.async K-tiles
    __shared__ signed char smem_a[2][BM * BK];   // 2 × 2048 = 4096 bytes
    __shared__ signed char smem_b[2][BK * BN];   // 2 × 2048 = 4096 bytes

    // Epilogue: one 16×16 int tile per warp (reuse after K-loop done)
    __shared__ int epilogue_tile[NUM_WARPS][WMMA_M * WMMA_N];  // 4 × 1024 = 4096 bytes

    int tid     = threadIdx.y * blockDim.x + threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int wy      = warp_id / WARPS_X;
    int wx      = warp_id % WARPS_X;

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> acc[WARP_TILES_Y][WARP_TILES_N];
    #pragma unroll
    for (int i = 0; i < WARP_TILES_Y; i++)
        #pragma unroll
        for (int j = 0; j < WARP_TILES_N; j++)
            wmma::fill_fragment(acc[i][j], 0);

    int num_tiles = (K + BK - 1) / BK;

    // ====================================================================
    // Prologue: cp.async tile 0 → buffer 0
    // ====================================================================
    #pragma unroll
    for (int i = 0; i < CP_ELEMS_A; i++) {
        int byte_idx = (tid + i * BLOCK_SIZE) * CP_ASYNC_SIZE;
        int row      = byte_idx / BK;
        int col      = byte_idx % BK;
        int g_row    = block_row + row;
        int g_col    = col;

        if (g_row < M && g_col + CP_ASYNC_SIZE - 1 < K) {
            __pipeline_memcpy_async(
                smem_a[0] + byte_idx,
                matrix_a + g_row * K + g_col,
                CP_ASYNC_SIZE);
        } else {
            for (int b = 0; b < CP_ASYNC_SIZE; b++) {
                int gc = g_col + b;
                smem_a[0][byte_idx + b] = (g_row < M && gc < K)
                    ? matrix_a[g_row * K + gc] : (signed char)0;
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < CP_ELEMS_B; i++) {
        int byte_idx = (tid + i * BLOCK_SIZE) * CP_ASYNC_SIZE;
        int row      = byte_idx / BN;
        int col      = byte_idx % BN;
        int g_row    = row;
        int g_col    = block_col + col;

        if (g_row < K && g_col + CP_ASYNC_SIZE - 1 < N) {
            __pipeline_memcpy_async(
                smem_b[0] + byte_idx,
                matrix_b + g_row * N + g_col,
                CP_ASYNC_SIZE);
        } else {
            for (int b = 0; b < CP_ASYNC_SIZE; b++) {
                int gc = g_col + b;
                smem_b[0][byte_idx + b] = (g_row < K && gc < N)
                    ? matrix_b[g_row * N + gc] : (signed char)0;
            }
        }
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    // ====================================================================
    // Main loop: cp.async(tile+1) overlapped with IMMA(tile)
    // ====================================================================
    for (int tile = 0; tile < num_tiles - 1; tile++) {
        int next_k_base = (tile + 1) * BK;
        int cur_buf     = tile & 1;
        int next_buf    = 1 - cur_buf;

        #pragma unroll
        for (int i = 0; i < CP_ELEMS_A; i++) {
            int byte_idx = (tid + i * BLOCK_SIZE) * CP_ASYNC_SIZE;
            int row      = byte_idx / BK;
            int col      = byte_idx % BK;
            int g_row    = block_row + row;
            int g_col    = next_k_base + col;

            if (g_row < M && g_col + CP_ASYNC_SIZE - 1 < K) {
                __pipeline_memcpy_async(
                    smem_a[next_buf] + byte_idx,
                    matrix_a + g_row * K + g_col,
                    CP_ASYNC_SIZE);
            } else {
                for (int b = 0; b < CP_ASYNC_SIZE; b++) {
                    int gc = g_col + b;
                    smem_a[next_buf][byte_idx + b] = (g_row < M && gc < K)
                        ? matrix_a[g_row * K + gc] : (signed char)0;
                }
            }
        }
        #pragma unroll
        for (int i = 0; i < CP_ELEMS_B; i++) {
            int byte_idx = (tid + i * BLOCK_SIZE) * CP_ASYNC_SIZE;
            int row      = byte_idx / BN;
            int col      = byte_idx % BN;
            int g_row    = next_k_base + row;
            int g_col    = block_col + col;

            if (g_row < K && g_col + CP_ASYNC_SIZE - 1 < N) {
                __pipeline_memcpy_async(
                    smem_b[next_buf] + byte_idx,
                    matrix_b + g_row * N + g_col,
                    CP_ASYNC_SIZE);
            } else {
                for (int b = 0; b < CP_ASYNC_SIZE; b++) {
                    int gc = g_col + b;
                    smem_b[next_buf][byte_idx + b] = (g_row < K && gc < N)
                        ? matrix_b[g_row * N + gc] : (signed char)0;
                }
            }
        }
        __pipeline_commit();

        #pragma unroll
        for (int k_local = 0; k_local < BK; k_local += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           signed char, wmma::row_major> a_frag[WARP_TILES_Y];
            #pragma unroll
            for (int wi = 0; wi < WARP_TILES_Y; wi++) {
                int a_row = wy * 32 + wi * WMMA_M;
                wmma::load_matrix_sync(a_frag[wi],
                    smem_a[cur_buf] + a_row * BK + k_local, BK);
            }

            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           signed char, wmma::row_major> b_frag[WARP_TILES_N];
            #pragma unroll
            for (int wj = 0; wj < WARP_TILES_N; wj++) {
                int b_col = wx * 32 + wj * WMMA_N;
                wmma::load_matrix_sync(b_frag[wj],
                    smem_b[cur_buf] + k_local * BN + b_col, BN);
            }

            #pragma unroll
            for (int wi = 0; wi < WARP_TILES_Y; wi++)
                #pragma unroll
                for (int wj = 0; wj < WARP_TILES_N; wj++)
                    wmma::mma_sync(acc[wi][wj], a_frag[wi], b_frag[wj], acc[wi][wj]);
        }

        __pipeline_wait_prior(0);
        __syncthreads();
    }

    // ====================================================================
    // Epilogue: Compute IMMA on last tile
    // ====================================================================
    {
        int last_buf = (num_tiles - 1) & 1;
        #pragma unroll
        for (int k_local = 0; k_local < BK; k_local += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           signed char, wmma::row_major> a_frag[WARP_TILES_Y];
            #pragma unroll
            for (int wi = 0; wi < WARP_TILES_Y; wi++) {
                int a_row = wy * 32 + wi * WMMA_M;
                wmma::load_matrix_sync(a_frag[wi],
                    smem_a[last_buf] + a_row * BK + k_local, BK);
            }

            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           signed char, wmma::row_major> b_frag[WARP_TILES_N];
            #pragma unroll
            for (int wj = 0; wj < WARP_TILES_N; wj++) {
                int b_col = wx * 32 + wj * WMMA_N;
                wmma::load_matrix_sync(b_frag[wj],
                    smem_b[last_buf] + k_local * BN + b_col, BN);
            }

            #pragma unroll
            for (int wi = 0; wi < WARP_TILES_Y; wi++)
                #pragma unroll
                for (int wj = 0; wj < WARP_TILES_N; wj++)
                    wmma::mma_sync(acc[wi][wj], a_frag[wi], b_frag[wj], acc[wi][wj]);
        }
    }

    // ====================================================================
    // Per-channel dequantize and store
    //
    // Store INT32 accumulator to shared memory (wmma::store_matrix_sync)
    // so we can index by (row, col) with known layout, then apply
    // per-channel scale and zero_point.
    // ====================================================================
    #pragma unroll
    for (int wi = 0; wi < WARP_TILES_Y; wi++) {
        #pragma unroll
        for (int wj = 0; wj < WARP_TILES_N; wj++) {
            int c_row = block_row + wy * 32 + wi * WMMA_M;
            int c_col = block_col + wx * 32 + wj * WMMA_N;

            if (c_row + WMMA_M > M || c_col + WMMA_N > N) continue;

            // Store INT32 accumulator to warp's epilogue tile (row-major)
            wmma::store_matrix_sync(
                epilogue_tile[warp_id],
                acc[wi][wj], WMMA_N, wmma::mem_row_major);

            // All lanes in warp have written; synchronize warp
            __syncwarp();

            // Each thread dequantizes 8 elements (256 total / 32 threads)
            for (int elem = lane_id; elem < WMMA_M * WMMA_N; elem += WARP_SIZE) {
                int local_row = elem / WMMA_N;
                int local_col = elem % WMMA_N;
                int channel   = c_col + local_col;

                int   acc_val = epilogue_tile[warp_id][elem];
                float val     = ((float)acc_val - (float)per_channel_zero_point[channel])
                              * per_channel_scale[channel];

                matrix_c[(c_row + local_row) * N + channel] = val;
            }

            // Ensure all lanes done before next tile overwrites epilogue_tile
            __syncwarp();
        }
    }
}
