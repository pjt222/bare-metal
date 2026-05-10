/*
 * igemm_tiled.cu — Tiled INT8 GEMM with shared memory + Tensor Cores
 *
 * The naive igemm.cu loads A and B tiles directly from global memory.
 * At 4096³, each INT8 element is redundantly loaded ~128× across blocks.
 * Total global reads: ~8 GB → 13 ms at 608 GB/s.
 *
 * This tiled version loads A/B into shared memory first. Each element is
 * loaded from global once per block, then reused by multiple warps.
 *
 * Block tile: BM×BN = 64×64 output elements
 *   4 warps in 2×2 arrangement, each warp computes 32×32 (2×2 WMMA tiles)
 *   smem_a: [BM × BK] = [64 × 32] INT8 = 2 KB
 *   smem_b: [BK × BN] = [32 × 64] INT8 = 2 KB
 *   Total shared memory: 4 KB
 *
 * Data reuse per K-tile:
 *   - Each A row used by 2 warps (same warp-row, both warp-cols)
 *   - Each B col used by 2 warps (same warp-col, both warp-rows)
 *   - Within each warp: 2 A frags × 2 B frags = 4 WMMA calls per K-step
 *   Global reads: 2 GB at 4096³ (vs 8 GB naive → 4× reduction)
 *
 * SASS instructions:
 *   IMMA.16816.S8.S8 — INT8 Tensor Core (same as naive)
 *   LDG.E / LDG.E.128 — global loads (into shared memory)
 *   LDS — shared memory loads (into WMMA fragments)
 *   STS — shared memory stores
 *   BAR.SYNC — tile synchronization
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o igemm_tiled.sm_86.cubin igemm_tiled.cu
 *   cuobjdump -sass igemm_tiled.sm_86.cubin | grep IMMA
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

// Warp arrangement: 2×2 grid, each warp covers 32×32 of the 64×64 output
#define WARPS_Y 2
#define WARPS_X 2
// Within each warp: 2×2 WMMA tiles (each 16×16)
#define WARP_TILES_Y 2
#define WARP_TILES_N 2

extern "C" __global__ __launch_bounds__(BLOCK_SIZE)
void igemm_tiled(
    const signed char * __restrict__ matrix_a,   // M×K row-major (int8)
    const signed char * __restrict__ matrix_b,   // K×N row-major (int8)
    float             * __restrict__ matrix_c,   // M×N row-major (fp32, dequantized)
    int M, int N, int K,
    float scale_a,
    float scale_b
) {
    // Shared memory for A and B tiles
    __shared__ signed char smem_a[BM * BK];   // 64×32 = 2048 bytes
    __shared__ signed char smem_b[BK * BN];   // 32×64 = 2048 bytes

    int tid     = threadIdx.y * blockDim.x + threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int wy      = warp_id / WARPS_X;   // 0 or 1
    int wx      = warp_id % WARPS_X;   // 0 or 1

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    // Each warp maintains 2×2 = 4 INT32 accumulator fragments
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> acc[WARP_TILES_Y][WARP_TILES_N];
    #pragma unroll
    for (int i = 0; i < WARP_TILES_Y; i++)
        #pragma unroll
        for (int j = 0; j < WARP_TILES_N; j++)
            wmma::fill_fragment(acc[i][j], 0);

    // ====================================================================
    // K-tile loop: load A/B into smem, then WMMA from smem
    // ====================================================================
    for (int k_base = 0; k_base < K; k_base += BK) {

        // --- Cooperative load: A tile [BM × BK] from global to smem ---
        // 64×32 = 2048 bytes, 128 threads → 16 bytes each
        for (int idx = tid; idx < BM * BK; idx += BLOCK_SIZE) {
            int row = idx / BK;
            int col = idx % BK;
            int g_row = block_row + row;
            int g_col = k_base + col;
            smem_a[idx] = (g_row < M && g_col < K)
                ? matrix_a[g_row * K + g_col] : (signed char)0;
        }

        // --- Cooperative load: B tile [BK × BN] from global to smem ---
        for (int idx = tid; idx < BK * BN; idx += BLOCK_SIZE) {
            int row = idx / BN;
            int col = idx % BN;
            int g_row = k_base + row;
            int g_col = block_col + col;
            smem_b[idx] = (g_row < K && g_col < N)
                ? matrix_b[g_row * N + g_col] : (signed char)0;
        }

        __syncthreads();

        // --- WMMA from shared memory ---
        // Inner loop over K within the tile: BK/WMMA_K = 2 steps
        #pragma unroll
        for (int k_local = 0; k_local < BK; k_local += WMMA_K) {

            // Each warp loads 2 A fragments (for its 2 output row-tiles)
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           signed char, wmma::row_major> a_frag[WARP_TILES_Y];
            #pragma unroll
            for (int wi = 0; wi < WARP_TILES_Y; wi++) {
                int a_row = wy * 32 + wi * WMMA_M;
                wmma::load_matrix_sync(a_frag[wi],
                    smem_a + a_row * BK + k_local, BK);
            }

            // Each warp loads 2 B fragments (for its 2 output col-tiles)
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           signed char, wmma::row_major> b_frag[WARP_TILES_N];
            #pragma unroll
            for (int wj = 0; wj < WARP_TILES_N; wj++) {
                int b_col = wx * 32 + wj * WMMA_N;
                wmma::load_matrix_sync(b_frag[wj],
                    smem_b + k_local * BN + b_col, BN);
            }

            // 2×2 outer product: 4 mma_sync calls per K-step
            #pragma unroll
            for (int wi = 0; wi < WARP_TILES_Y; wi++)
                #pragma unroll
                for (int wj = 0; wj < WARP_TILES_N; wj++)
                    wmma::mma_sync(acc[wi][wj], a_frag[wi], b_frag[wj], acc[wi][wj]);
        }

        __syncthreads();
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
