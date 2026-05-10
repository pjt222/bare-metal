/*
 * igemm_persistent.cu — Persistent-grid 128×256 INT8 GEMM with L2 tile reuse
 *
 * Same compute body as igemm_8warp_256.cu but with a persistent kernel grid:
 * instead of launching ceil(N/256) × ceil(M/128) blocks, we launch exactly
 * 48 blocks (1 per SM) and each block loops, grabbing output tiles via
 * atomicAdd on a global counter.
 *
 * L2 reuse strategy — column-first tile ordering:
 *   Standard grid: blockIdx.x varies fastest → row-major tile order
 *   Persistent:    m varies fastest → column-first tile order
 *
 *   Column-first means CTAs processing tiles (0,n), (1,n), (2,n), ...
 *   run close together in time. These tiles share the SAME B K-tiles
 *   (K×BN slice of matrix B), so when CTA_i loads B[k, n*BN : (n+1)*BN],
 *   CTA_{i+1} finds it still in L2.
 *
 *   At 4096³ with BM=128, BN=256, BK=32:
 *     B K-tile = 32 × 256 × 1 byte = 8 KB
 *     48 CTAs × 12 KB working set = 576 KB — fits in 4 MB L2
 *     32 tiles per column → 32/48 CTAs share B K-tiles simultaneously
 *
 * Grid:  (48, 1, 1) — 1 block per SM on GA104
 * Block: (256, 1, 1) = 8 warps
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o igemm_persistent.sm_86.cubin igemm_persistent.cu
 */

#include <mma.h>
#include <cuda_pipeline.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BM 128
#define BN 256
#define BK 32

#define NUM_WARPS   8
#define WARP_SIZE   32
#define BLOCK_SIZE  (NUM_WARPS * WARP_SIZE)  // 256 threads

// 4×2 warp grid: 4 warp-rows × 2 warp-cols
#define WARPS_Y 4
#define WARPS_X 2

// Each warp covers 32×128 = 2×8 WMMA tiles
#define WARP_TILES_M 2   // 32 / 16
#define WARP_TILES_N 8   // 128 / 16

// cp.async: 4 bytes per call
#define CP_ASYNC_SIZE 4
#define CP_ELEMS_A  (BM * BK / BLOCK_SIZE / CP_ASYNC_SIZE)   // 128*32/256/4 = 4
#define CP_ELEMS_B  (BK * BN / BLOCK_SIZE / CP_ASYNC_SIZE)   // 32*256/256/4 = 8

// 1 block/SM — allows up to 255 regs/thread and 100 KB smem
extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void igemm_persistent(
    const signed char * __restrict__ matrix_a,
    const signed char * __restrict__ matrix_b,
    float             * __restrict__ matrix_c,
    int M, int N, int K,
    float scale_a,
    float scale_b,
    int * __restrict__ tile_counter   // global atomic counter, must be zeroed before launch
) {
    // Double-buffer for cp.async K-tiles
    __shared__ signed char smem_a[2][BM * BK];   // 2 × 4096 = 8192 bytes
    __shared__ signed char smem_b[2][BK * BN];   // 2 × 8192 = 16384 bytes

    // Epilogue: one 16×16 int tile per warp for coalesced global writes
    __shared__ int epilogue_tile[NUM_WARPS][WMMA_M * WMMA_N];  // 8 × 1024 = 8192 bytes

    // Work-stealing broadcast
    __shared__ int shared_tile_idx;

    int tid     = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int wy      = warp_id / WARPS_X;   // 0..3
    int wx      = warp_id % WARPS_X;   // 0..1

    // Compute tile grid from matrix dims (avoids extra kernel params → fewer registers)
    int tiles_m     = (M + BM - 1) / BM;
    int total_tiles = tiles_m * ((N + BN - 1) / BN);
    int num_k_tiles = (K + BK - 1) / BK;

    // Each warp maintains 2×8 = 16 INT32 accumulator fragments
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int>
        acc[WARP_TILES_M][WARP_TILES_N];

    // ====================================================================
    // Persistent work-stealing loop
    // ====================================================================
    while (true) {
        // --- Grab next output tile (thread 0 does atomic, broadcast via smem) ---
        if (tid == 0) {
            shared_tile_idx = atomicAdd(tile_counter, 1);
        }
        __syncthreads();
        int tile_idx = shared_tile_idx;

        if (tile_idx >= total_tiles) break;

        // --- Column-first decode: m varies fastest → B-tile L2 reuse ---
        int m_tile    = tile_idx % tiles_m;
        int n_tile    = tile_idx / tiles_m;
        int block_row = m_tile * BM;
        int block_col = n_tile * BN;

        // --- Zero accumulators for this output tile ---
        #pragma unroll
        for (int i = 0; i < WARP_TILES_M; i++)
            #pragma unroll
            for (int j = 0; j < WARP_TILES_N; j++)
                wmma::fill_fragment(acc[i][j], 0);

        // ================================================================
        // Prologue: cp.async tile 0 → buffer 0
        // ================================================================
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

        // ================================================================
        // Main K-loop: cp.async(tile+1) overlapped with IMMA(tile)
        // ================================================================
        for (int tile = 0; tile < num_k_tiles - 1; tile++) {
            int next_k_base = (tile + 1) * BK;
            int cur_buf     = tile & 1;
            int next_buf    = 1 - cur_buf;

            // --- Phase 1: Issue cp.async for NEXT K-tile → next_buf ---
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

            // --- Phase 2: Compute IMMA on CURRENT K-tile ---
            #pragma unroll
            for (int k_local = 0; k_local < BK; k_local += WMMA_K) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                               signed char, wmma::row_major> a_frag[WARP_TILES_M];
                #pragma unroll
                for (int wi = 0; wi < WARP_TILES_M; wi++) {
                    int a_row = wy * 32 + wi * WMMA_M;
                    wmma::load_matrix_sync(a_frag[wi],
                        smem_a[cur_buf] + a_row * BK + k_local, BK);
                }

                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                               signed char, wmma::row_major> b_frag[WARP_TILES_N];
                #pragma unroll
                for (int wj = 0; wj < WARP_TILES_N; wj++) {
                    int b_col = wx * 128 + wj * WMMA_N;
                    wmma::load_matrix_sync(b_frag[wj],
                        smem_b[cur_buf] + k_local * BN + b_col, BN);
                }

                #pragma unroll
                for (int wi = 0; wi < WARP_TILES_M; wi++)
                    #pragma unroll
                    for (int wj = 0; wj < WARP_TILES_N; wj++)
                        wmma::mma_sync(acc[wi][wj], a_frag[wi], b_frag[wj], acc[wi][wj]);
            }

            // --- Phase 3: Wait for cp.async, sync ---
            __pipeline_wait_prior(0);
            __syncthreads();
        }

        // ================================================================
        // Last K-tile: compute IMMA (no more loads to issue)
        // ================================================================
        {
            int last_buf = (num_k_tiles - 1) & 1;
            #pragma unroll
            for (int k_local = 0; k_local < BK; k_local += WMMA_K) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                               signed char, wmma::row_major> a_frag[WARP_TILES_M];
                #pragma unroll
                for (int wi = 0; wi < WARP_TILES_M; wi++) {
                    int a_row = wy * 32 + wi * WMMA_M;
                    wmma::load_matrix_sync(a_frag[wi],
                        smem_a[last_buf] + a_row * BK + k_local, BK);
                }

                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                               signed char, wmma::row_major> b_frag[WARP_TILES_N];
                #pragma unroll
                for (int wj = 0; wj < WARP_TILES_N; wj++) {
                    int b_col = wx * 128 + wj * WMMA_N;
                    wmma::load_matrix_sync(b_frag[wj],
                        smem_b[last_buf] + k_local * BN + b_col, BN);
                }

                #pragma unroll
                for (int wi = 0; wi < WARP_TILES_M; wi++)
                    #pragma unroll
                    for (int wj = 0; wj < WARP_TILES_N; wj++)
                        wmma::mma_sync(acc[wi][wj], a_frag[wi], b_frag[wj], acc[wi][wj]);
            }
        }

        // ================================================================
        // Dequantize and store — coalesced epilogue via smem
        // ================================================================
        {
            float dequant_scale = scale_a * scale_b;
            #pragma unroll
            for (int wi = 0; wi < WARP_TILES_M; wi++) {
                #pragma unroll
                for (int wj = 0; wj < WARP_TILES_N; wj++) {
                    int c_row = block_row + wy * 32 + wi * WMMA_M;
                    int c_col = block_col + wx * 128 + wj * WMMA_N;

                    if (c_row + WMMA_M > M || c_col + WMMA_N > N) continue;

                    wmma::store_matrix_sync(
                        epilogue_tile[warp_id],
                        acc[wi][wj], WMMA_N, wmma::mem_row_major);
                    __syncwarp();

                    for (int elem = lane_id; elem < WMMA_M * WMMA_N; elem += WARP_SIZE) {
                        int local_row = elem / WMMA_N;
                        int local_col = elem % WMMA_N;

                        int   acc_val = epilogue_tile[warp_id][elem];
                        float val     = (float)acc_val * dequant_scale;

                        matrix_c[(c_row + local_row) * N + (c_col + local_col)] = val;
                    }
                    __syncwarp();
                }
            }
        }

        // Ensure all global writes complete before smem reuse in next tile
        __syncthreads();
    }
}
