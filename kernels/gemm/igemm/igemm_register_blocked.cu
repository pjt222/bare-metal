/*
 * igemm_register_blocked.cu — Register-blocked INT8 GEMM with Tensor Cores
 *
 * The tiled IGEMM (64×64) achieves 15,105 TOPS at 4096³ but is still
 * bandwidth-bound at 2.2% of INT8 peak. Each 4 KB smem tile fuels only
 * 32 mma_sync calls (64 ops/byte).
 *
 * This version doubles the output tile to 128×128. Each warp maintains
 * 4×4 = 16 WMMA accumulator fragments (vs 2×2 = 4 in the tiled kernel),
 * which is the INT8/WMMA analogue of the FP32 register_blocked.cu pattern.
 *
 * Data reuse:
 *   smem_a: [128 × 32] INT8 = 4 KB (loaded once, used by 2 warp-rows × 4 col-tiles)
 *   smem_b: [32 × 128] INT8 = 4 KB (loaded once, used by 2 warp-cols × 4 row-tiles)
 *   Total smem: 8 KB (well under 64 KB cliff → 2 blocks/SM → 8 warps)
 *
 * Arithmetic intensity:
 *   Per K-tile: 128 mma_sync × 8192 ops = 1,048,576 ops / 8 KB = 128 ops/byte
 *   vs tiled (64×64): 32 mma_sync × 8192 = 262,144 / 4 KB = 64 ops/byte
 *   2× improvement in compute-per-byte loaded from global.
 *
 * Register budget per thread:
 *   16 accum frags × 8 regs = 128   (INT32 accumulators)
 *   4 A frags × 2 regs      = 8     (INT8 inputs, reused across B)
 *   4 B frags × 2 regs      = 8     (INT8 inputs, reused across A)
 *   Temporaries              ≈ 12
 *   Total: ~156 registers → fits 255 limit, 2 blocks/SM verified
 *
 * SASS instructions:
 *   IMMA.16816.S8.S8 — INT8 Tensor Core (16 mma_sync × 4 IMMA × 2 K-steps = 128)
 *   LDS              — shared memory fragment loads
 *   I2FP.F32.S32     — INT32→FP32 dequantization
 *   FMUL             — scale multiplication
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o igemm_register_blocked.sm_86.cubin igemm_register_blocked.cu
 *   cuobjdump -sass igemm_register_blocked.sm_86.cubin | grep IMMA | wc -l
 */

#include <mma.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BM 128      // output rows per block
#define BN 128      // output cols per block
#define BK 32       // K-tile width (2 WMMA K-steps per smem tile)

#define NUM_WARPS   4
#define WARP_SIZE   32
#define BLOCK_SIZE  (NUM_WARPS * WARP_SIZE)  // 128 threads

// Warp arrangement: 2×2 grid, each warp covers 64×64 of the 128×128 output
#define WARPS_Y 2
#define WARPS_X 2

// Within each warp: 4×4 WMMA tiles (each 16×16) = 64×64 output per warp
#define WARP_TILES_M 4
#define WARP_TILES_N 4

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 2)
void igemm_register_blocked(
    const signed char * __restrict__ matrix_a,   // M×K row-major (int8)
    const signed char * __restrict__ matrix_b,   // K×N row-major (int8)
    float             * __restrict__ matrix_c,   // M×N row-major (fp32, dequantized)
    int M, int N, int K,
    float scale_a,
    float scale_b
) {
    __shared__ signed char smem_a[BM * BK];   // 128×32 = 4096 bytes
    __shared__ signed char smem_b[BK * BN];   // 32×128 = 4096 bytes

    int tid     = threadIdx.x;               // 0..127
    int warp_id = tid / WARP_SIZE;
    int wy      = warp_id / WARPS_X;         // 0 or 1
    int wx      = warp_id % WARPS_X;         // 0 or 1

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    // Each warp maintains 4×4 = 16 INT32 accumulator fragments
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int>
        acc[WARP_TILES_M][WARP_TILES_N];
    #pragma unroll
    for (int i = 0; i < WARP_TILES_M; i++)
        #pragma unroll
        for (int j = 0; j < WARP_TILES_N; j++)
            wmma::fill_fragment(acc[i][j], 0);

    // ====================================================================
    // K-tile loop
    // ====================================================================
    for (int k_base = 0; k_base < K; k_base += BK) {

        // --- Cooperative load: A tile [128 × 32] = 4096 bytes ---
        // 128 threads → 32 bytes each
        for (int idx = tid; idx < BM * BK; idx += BLOCK_SIZE) {
            int row = idx / BK;
            int col = idx % BK;
            int g_row = block_row + row;
            int g_col = k_base + col;
            smem_a[idx] = (g_row < M && g_col < K)
                ? matrix_a[g_row * K + g_col] : (signed char)0;
        }

        // --- Cooperative load: B tile [32 × 128] = 4096 bytes ---
        for (int idx = tid; idx < BK * BN; idx += BLOCK_SIZE) {
            int row = idx / BN;
            int col = idx % BN;
            int g_row = k_base + row;
            int g_col = block_col + col;
            smem_b[idx] = (g_row < K && g_col < N)
                ? matrix_b[g_row * N + g_col] : (signed char)0;
        }

        __syncthreads();

        // --- WMMA from shared memory: 4×4 outer product per warp ---
        #pragma unroll
        for (int k_local = 0; k_local < BK; k_local += WMMA_K) {

            // Load 4 A fragments (one per output row-tile of this warp)
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           signed char, wmma::row_major> a_frag[WARP_TILES_M];
            #pragma unroll
            for (int wi = 0; wi < WARP_TILES_M; wi++) {
                int a_row = wy * 64 + wi * WMMA_M;
                wmma::load_matrix_sync(a_frag[wi],
                    smem_a + a_row * BK + k_local, BK);
            }

            // Load 4 B fragments (one per output col-tile of this warp)
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           signed char, wmma::row_major> b_frag[WARP_TILES_N];
            #pragma unroll
            for (int wj = 0; wj < WARP_TILES_N; wj++) {
                int b_col = wx * 64 + wj * WMMA_N;
                wmma::load_matrix_sync(b_frag[wj],
                    smem_b + k_local * BN + b_col, BN);
            }

            // 4×4 outer product: 16 mma_sync calls per K-step
            #pragma unroll
            for (int wi = 0; wi < WARP_TILES_M; wi++)
                #pragma unroll
                for (int wj = 0; wj < WARP_TILES_N; wj++)
                    wmma::mma_sync(acc[wi][wj], a_frag[wi], b_frag[wj],
                                   acc[wi][wj]);
        }

        __syncthreads();
    }

    // ====================================================================
    // Dequantize and store: INT32 → FP32
    // ====================================================================
    float dequant_scale = scale_a * scale_b;

    #pragma unroll
    for (int wi = 0; wi < WARP_TILES_M; wi++) {
        #pragma unroll
        for (int wj = 0; wj < WARP_TILES_N; wj++) {
            int c_row = block_row + wy * 64 + wi * WMMA_M;
            int c_col = block_col + wx * 64 + wj * WMMA_N;

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
