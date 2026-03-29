/*
 * igemm_8warp_tribuf.cu — 8-warp 128×128 INT8 GEMM with cp.async TRIPLE-buffer
 *
 * Key insight: 1 block/SM × 8 warps = same 8-warp occupancy as 2 blocks × 4 warps,
 * but with 100 KB smem budget instead of 50 KB. This unlocks 128×128 tiles.
 *
 * The prior 128×128 attempt (igemm_register_blocked.cu) failed at 0.84× because:
 *   1. Only 4 warps → 16 mma_sync per warp per K-step (loop too long, Insight 13)
 *   2. No pipelining → all warps stall at __syncthreads during loads
 *
 * This kernel fixes both:
 *   1. 8 warps (4×2 grid) → 8 mma_sync per warp per K-step (short loop preserved)
 *   2. cp.async double-buffer → compute(N) overlaps with load(N+1)
 *
 * Warp layout: 4×2 grid on 128×128 output tile
 *   Each warp covers 32×64 = 2×4 WMMA tiles (16×16 each)
 *   Per K-step: 8 mma_sync per warp = 16 IMMA per warp
 *   Per K-tile (BK=32, 2 K-steps): 16 mma_sync per warp = 32 IMMA per warp
 *
 * Arithmetic intensity: 128 ops/byte (vs 64 for 64×64 tiles) — 2× improvement
 *
 * Shared memory: 16 KB double-buffer + 8 KB epilogue = 24 KB total
 *   Trivially under 100 KB limit for 1 block/SM
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o igemm_8warp.sm_86.cubin igemm_8warp.cu
 */

#include <mma.h>
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

// cp.async: 4 bytes per call, same load distribution as 64×64 with 128 threads
#define CP_ASYNC_SIZE 4
#define CP_ELEMS_A  (BM * BK / BLOCK_SIZE / CP_ASYNC_SIZE)   // 128*32/256/4 = 4
#define CP_ELEMS_B  (BK * BN / BLOCK_SIZE / CP_ASYNC_SIZE)   // 32*128/256/4 = 4

// Request exactly 1 block/SM — allows up to 255 regs/thread and 100 KB smem
extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void igemm_8warp_tribuf(
    const signed char * __restrict__ matrix_a,
    const signed char * __restrict__ matrix_b,
    float             * __restrict__ matrix_c,
    int M, int N, int K,
    float scale_a,
    float scale_b
) {
    // Triple-buffer for cp.async K-tiles
    __shared__ signed char smem_a[3][BM * BK];   // 3 × 4096 = 12288 bytes
    __shared__ signed char smem_b[3][BK * BN];   // 3 × 4096 = 12288 bytes

    // Epilogue: one 16×16 int tile per warp for coalesced global writes
    __shared__ int epilogue_tile[NUM_WARPS][WMMA_M * WMMA_N];  // 8 × 1024 = 8192 bytes

    int tid     = threadIdx.y * blockDim.x + threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int wy      = warp_id / WARPS_X;   // 0..3
    int wx      = warp_id % WARPS_X;   // 0..1

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    // Each warp maintains 2×4 = 8 INT32 accumulator fragments
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int>
        acc[WARP_TILES_M][WARP_TILES_N];
    #pragma unroll
    for (int i = 0; i < WARP_TILES_M; i++)
        #pragma unroll
        for (int j = 0; j < WARP_TILES_N; j++)
            wmma::fill_fragment(acc[i][j], 0);

    int num_tiles = (K + BK - 1) / BK;

    // ====================================================================
    // Prologue: cp.async tiles 0 and 1 → buffers 0 and 1
    // ====================================================================
    // Helper lambda for issuing cp.async for a given K-tile into a buffer
    #define ISSUE_CPASYNC_TILE(buf_idx, k_base_val) do { \
        _Pragma("unroll") \
        for (int i = 0; i < CP_ELEMS_A; i++) { \
            int byte_idx = (tid + i * BLOCK_SIZE) * CP_ASYNC_SIZE; \
            int row      = byte_idx / BK; \
            int col      = byte_idx % BK; \
            int g_row    = block_row + row; \
            int g_col    = (k_base_val) + col; \
            if (g_row < M && g_col + CP_ASYNC_SIZE - 1 < K) { \
                __pipeline_memcpy_async( \
                    smem_a[buf_idx] + byte_idx, \
                    matrix_a + g_row * K + g_col, \
                    CP_ASYNC_SIZE); \
            } else { \
                for (int b = 0; b < CP_ASYNC_SIZE; b++) { \
                    int gc = g_col + b; \
                    smem_a[buf_idx][byte_idx + b] = (g_row < M && gc < K) \
                        ? matrix_a[g_row * K + gc] : (signed char)0; \
                } \
            } \
        } \
        _Pragma("unroll") \
        for (int i = 0; i < CP_ELEMS_B; i++) { \
            int byte_idx = (tid + i * BLOCK_SIZE) * CP_ASYNC_SIZE; \
            int row      = byte_idx / BN; \
            int col      = byte_idx % BN; \
            int g_row    = (k_base_val) + row; \
            int g_col    = block_col + col; \
            if (g_row < K && g_col + CP_ASYNC_SIZE - 1 < N) { \
                __pipeline_memcpy_async( \
                    smem_b[buf_idx] + byte_idx, \
                    matrix_b + g_row * N + g_col, \
                    CP_ASYNC_SIZE); \
            } else { \
                for (int b = 0; b < CP_ASYNC_SIZE; b++) { \
                    int gc = g_col + b; \
                    smem_b[buf_idx][byte_idx + b] = (g_row < K && gc < N) \
                        ? matrix_b[g_row * N + gc] : (signed char)0; \
                } \
            } \
        } \
        __pipeline_commit(); \
    } while(0)

    // Issue tiles 0 and 1
    ISSUE_CPASYNC_TILE(0, 0);
    if (num_tiles > 1) {
        ISSUE_CPASYNC_TILE(1, BK);
    }
    // Wait for tile 0 to be ready (allow tile 1 to remain in-flight)
    __pipeline_wait_prior(num_tiles > 1 ? 1 : 0);
    __syncthreads();

    // ====================================================================
    // Main loop: compute(tile), issue(tile+2), wait for (tile+1)
    // Triple-buffer: buffers rotate through 0, 1, 2
    // ====================================================================
    for (int tile = 0; tile < num_tiles - 2; tile++) {
        int cur_buf  = tile % 3;
        int next2_k  = (tile + 2) * BK;
        int next2_buf = (tile + 2) % 3;

        // --- Phase 1: Issue cp.async for tile+2 → next2_buf ---
        ISSUE_CPASYNC_TILE(next2_buf, next2_k);

        // --- Phase 2: Compute IMMA on CURRENT tile ---
        #pragma unroll
        for (int k_local = 0; k_local < BK; k_local += WMMA_K) {
            // Load A fragments: 2 per warp (WARP_TILES_M = 2)
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           signed char, wmma::row_major> a_frag[WARP_TILES_M];
            #pragma unroll
            for (int wi = 0; wi < WARP_TILES_M; wi++) {
                int a_row = wy * 32 + wi * WMMA_M;
                wmma::load_matrix_sync(a_frag[wi],
                    smem_a[cur_buf] + a_row * BK + k_local, BK);
            }

            // Load B fragments: 4 per warp (WARP_TILES_N = 4)
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           signed char, wmma::row_major> b_frag[WARP_TILES_N];
            #pragma unroll
            for (int wj = 0; wj < WARP_TILES_N; wj++) {
                int b_col = wx * 64 + wj * WMMA_N;
                wmma::load_matrix_sync(b_frag[wj],
                    smem_b[cur_buf] + k_local * BN + b_col, BN);
            }

            // 2×4 outer product: 8 mma_sync per K-step
            #pragma unroll
            for (int wi = 0; wi < WARP_TILES_M; wi++)
                #pragma unroll
                for (int wj = 0; wj < WARP_TILES_N; wj++)
                    wmma::mma_sync(acc[wi][wj], a_frag[wi], b_frag[wj], acc[wi][wj]);
        }

        // --- Phase 3: Wait for tile+1 to be ready (allow tile+2 in-flight) ---
        __pipeline_wait_prior(1);
        __syncthreads();
    }

    // ====================================================================
    // Drain: Compute IMMA on last 2 tiles (no more tiles to issue)
    // ====================================================================
    for (int tile = (num_tiles >= 2 ? num_tiles - 2 : 0); tile < num_tiles; tile++) {
        if (tile > 0 && tile == num_tiles - 1) {
            // Wait for the very last in-flight tile
            __pipeline_wait_prior(0);
            __syncthreads();
        }
        int drain_buf = tile % 3;
        #pragma unroll
        for (int k_local = 0; k_local < BK; k_local += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           signed char, wmma::row_major> a_frag[WARP_TILES_M];
            #pragma unroll
            for (int wi = 0; wi < WARP_TILES_M; wi++) {
                int a_row = wy * 32 + wi * WMMA_M;
                wmma::load_matrix_sync(a_frag[wi],
                    smem_a[drain_buf] + a_row * BK + k_local, BK);
            }

            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           signed char, wmma::row_major> b_frag[WARP_TILES_N];
            #pragma unroll
            for (int wj = 0; wj < WARP_TILES_N; wj++) {
                int b_col = wx * 64 + wj * WMMA_N;
                wmma::load_matrix_sync(b_frag[wj],
                    smem_b[drain_buf] + k_local * BN + b_col, BN);
            }

            #pragma unroll
            for (int wi = 0; wi < WARP_TILES_M; wi++)
                #pragma unroll
                for (int wj = 0; wj < WARP_TILES_N; wj++)
                    wmma::mma_sync(acc[wi][wj], a_frag[wi], b_frag[wj], acc[wi][wj]);
        }
    }

    // ====================================================================
    // Dequantize and store — coalesced epilogue via smem
    //
    // Store INT32 accumulator to shared memory (row-major), then write
    // to global memory with explicit (row, col) indexing for perfect
    // coalescing (Insight 17: +3.1% over wmma::store_matrix_sync).
    // ====================================================================
    float dequant_scale = scale_a * scale_b;

    #pragma unroll
    for (int wi = 0; wi < WARP_TILES_M; wi++) {
        #pragma unroll
        for (int wj = 0; wj < WARP_TILES_N; wj++) {
            int c_row = block_row + wy * 32 + wi * WMMA_M;
            int c_col = block_col + wx * 64 + wj * WMMA_N;

            if (c_row + WMMA_M > M || c_col + WMMA_N > N) continue;

            // Store INT32 accumulator to warp's epilogue tile
            wmma::store_matrix_sync(
                epilogue_tile[warp_id],
                acc[wi][wj], WMMA_N, wmma::mem_row_major);
            __syncwarp();

            // Coalesced write: each thread handles 8 elements
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
