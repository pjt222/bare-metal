/*
 * igemm_pipelined.cu — Software-pipelined INT8 GEMM with double-buffered smem
 *
 * The tiled igemm_tiled.cu loads A/B into shared memory, syncs, then computes
 * IMMA — fully sequential load→compute per K-tile. At 4096³ this leaves ~5.7 ms
 * of pipeline bubbles where IMMA waits for data and LDG waits for IMMA.
 *
 * This version double-buffers shared memory so that global loads for tile N+1
 * overlap with IMMA compute on tile N:
 *
 *   Prologue: Load tile 0 → smem buf[0], sync
 *   Loop:     LDG tile N+1 → registers  (non-blocking, memory pipeline)
 *             IMMA on buf[N%2]           (overlaps with LDG in flight)
 *             sync                       (ensure IMMA reads done)
 *             STS registers → buf[(N+1)%2]
 *             sync                       (ensure STS done)
 *   Epilogue: IMMA on last buf
 *
 * Shared memory: smem_a[2][BM*BK] + smem_b[2][BK*BN] = 2×(2+2) = 8 KB
 *   Well under the 50 KB smem cliff (GA104: 100 KB/SM ÷ 2 blocks = 50 KB/block)
 *
 * Block tile: BM×BN = 64×64, 4 warps in 2×2, each warp 32×32 (2×2 WMMA tiles)
 * Same tile geometry as igemm_tiled.cu — only the K-loop structure changes.
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o igemm_pipelined.sm_86.cubin igemm_pipelined.cu
 *   cuobjdump -sass igemm_pipelined.sm_86.cubin | grep -E 'IMMA|LDG|STS'
 */

#include <mma.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BM 64       // output rows per block
#define BN 64       // output cols per block
#define BK 32       // K-tile width (2 WMMA K-steps per smem tile)

#define NUM_WARPS   4
#define WARP_SIZE   32
#define BLOCK_SIZE  (NUM_WARPS * WARP_SIZE)  // 128 threads

#define WARPS_Y 2
#define WARPS_X 2
#define WARP_TILES_Y 2
#define WARP_TILES_N 2

// Each thread prefetches this many int8 elements per matrix per K-tile
#define ELEMS_PER_THREAD_A  (BM * BK / BLOCK_SIZE)   // 2048 / 128 = 16
#define ELEMS_PER_THREAD_B  (BK * BN / BLOCK_SIZE)   // 2048 / 128 = 16

extern "C" __global__ __launch_bounds__(BLOCK_SIZE)
void igemm_pipelined(
    const signed char * __restrict__ matrix_a,   // M×K row-major (int8)
    const signed char * __restrict__ matrix_b,   // K×N row-major (int8)
    float             * __restrict__ matrix_c,   // M×N row-major (fp32, dequantized)
    int M, int N, int K,
    float scale_a,
    float scale_b
) {
    // Double-buffered shared memory
    __shared__ signed char smem_a[2][BM * BK];   // 2 × 2048 = 4096 bytes
    __shared__ signed char smem_b[2][BK * BN];   // 2 × 2048 = 4096 bytes
    // Total: 8 KB

    int tid     = threadIdx.y * blockDim.x + threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int wy      = warp_id / WARPS_X;
    int wx      = warp_id % WARPS_X;

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    // Accumulators: 2×2 INT32 fragments per warp
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> acc[WARP_TILES_Y][WARP_TILES_N];
    #pragma unroll
    for (int i = 0; i < WARP_TILES_Y; i++)
        #pragma unroll
        for (int j = 0; j < WARP_TILES_N; j++)
            wmma::fill_fragment(acc[i][j], 0);

    int num_tiles = (K + BK - 1) / BK;

    // ====================================================================
    // Prologue: Load tile 0 into buffer 0
    // ====================================================================
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD_A; i++) {
        int idx   = tid + i * BLOCK_SIZE;
        int row   = idx / BK;
        int col   = idx % BK;
        int g_row = block_row + row;
        int g_col = col;
        smem_a[0][idx] = (g_row < M && g_col < K)
            ? matrix_a[g_row * K + g_col] : (signed char)0;
    }
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD_B; i++) {
        int idx   = tid + i * BLOCK_SIZE;
        int row   = idx / BN;
        int col   = idx % BN;
        int g_row = row;
        int g_col = block_col + col;
        smem_b[0][idx] = (g_row < K && g_col < N)
            ? matrix_b[g_row * N + g_col] : (signed char)0;
    }
    __syncthreads();

    // ====================================================================
    // Main loop: overlap LDG(tile+1) with IMMA(tile)
    // ====================================================================
    for (int tile = 0; tile < num_tiles - 1; tile++) {
        int next_k_base = (tile + 1) * BK;
        int cur_buf     = tile & 1;
        int next_buf    = 1 - cur_buf;

        // --- Phase 1: Prefetch NEXT tile into registers (LDG, non-blocking) ---
        signed char reg_a[ELEMS_PER_THREAD_A];
        signed char reg_b[ELEMS_PER_THREAD_B];

        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD_A; i++) {
            int idx   = tid + i * BLOCK_SIZE;
            int row   = idx / BK;
            int col   = idx % BK;
            int g_row = block_row + row;
            int g_col = next_k_base + col;
            reg_a[i] = (g_row < M && g_col < K)
                ? matrix_a[g_row * K + g_col] : (signed char)0;
        }
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD_B; i++) {
            int idx   = tid + i * BLOCK_SIZE;
            int row   = idx / BN;
            int col   = idx % BN;
            int g_row = next_k_base + row;
            int g_col = block_col + col;
            reg_b[i] = (g_row < K && g_col < N)
                ? matrix_b[g_row * N + g_col] : (signed char)0;
        }

        // --- Phase 2: Compute IMMA on CURRENT tile (overlaps with LDG in flight) ---
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

        // --- Phase 3: Drain registers into next smem buffer ---
        __syncthreads();  // Ensure all warps done reading cur_buf

        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD_A; i++)
            smem_a[next_buf][tid + i * BLOCK_SIZE] = reg_a[i];
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD_B; i++)
            smem_b[next_buf][tid + i * BLOCK_SIZE] = reg_b[i];

        __syncthreads();  // Ensure STS complete before next iteration reads
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
    // Dequantize and store: INT32 → FP32
    // ====================================================================
    float dequant_scale = scale_a * scale_b;

    #pragma unroll
    for (int wi = 0; wi < WARP_TILES_Y; wi++) {
        #pragma unroll
        for (int wj = 0; wj < WARP_TILES_N; wj++) {
            int c_row = block_row + wy * 32 + wi * WMMA_M;
            int c_col = block_col + wx * 32 + wj * WMMA_N;

            if (c_row + WMMA_M > M || c_col + WMMA_N > N) continue;

            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> float_acc;
            #pragma unroll
            for (int i = 0; i < acc[wi][wj].num_elements; i++) {
                float_acc.x[i] = (float)acc[wi][wj].x[i] * dequant_scale;
            }

            wmma::store_matrix_sync(
                matrix_c + c_row * N + c_col,
                float_acc, N, wmma::mem_row_major);
        }
    }
}
