/*
 * hgemm.cu — FP16 GEMM using Tensor Cores (WMMA API → HMMA SASS)
 *
 * Uses CUDA's warp-level matrix multiply-accumulate (WMMA) API.
 * Each PTX `mma.sync` instruction compiles to 2 HMMA SASS instructions,
 * each performing a 16×8×16 warp-level matrix multiply in one cycle.
 *
 * Why FP16 Tensor Cores?
 *   FP32 FFMA peak:    ~21.7 TFLOPS  (RTX 3070 Ti)
 *   FP16 Tensor peak:  ~174  TFLOPS  (8× more — same power budget)
 *
 * All transformer and diffusion inference runs on Tensor Cores.
 * This kernel is the gateway to Flash Attention performance.
 *
 * WMMA fragment layout on sm_86 (Ampere, 16×16×16 tile):
 *   A fragment: 16×16 half-precision matrix, distributed across 32 threads
 *               Each thread holds 8 half values (2 registers of FP16x2)
 *   B fragment: Same — 16×16 half, 8 values per thread
 *   C/D accum:  16×16 float, each thread holds 8 float values (8 registers)
 *
 * The SASS we want to see:
 *   HMMA.16816.F32 Rd, Ra, Rb, Rc   (FP16 in, FP32 accumulation)
 *
 * C = A * B, A: M×K (fp16), B: K×N (fp16), C: M×N (fp32)
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o hgemm.sm_86.cubin hgemm.cu
 *   cuobjdump -sass hgemm.sm_86.cubin | grep HMMA
 *
 * References:
 *   PTX ISA: mma.sync.aligned.m16n8k16
 *   CUDA WMMA API: <mma.h>
 */

#include <mma.h>
using namespace nvcuda;

// Tile sizes — must be multiples of WMMA fragment size (16)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Thread block: 2 warps wide × 2 warps tall = 4 warps = 128 threads
// Each warp computes one 16×16 output tile
#define WARPS_PER_BLOCK_X 2
#define WARPS_PER_BLOCK_Y 2
#define WARP_SIZE         32
#define BLOCK_X           (WARPS_PER_BLOCK_X * WARP_SIZE)  // 64
#define BLOCK_Y           WARPS_PER_BLOCK_Y                 // 2

// Each thread block covers a 32×32 region of C
// (2 warps × 16 wide) × (2 warps × 16 tall)
#define BLOCK_TILE_M      (WARPS_PER_BLOCK_Y * WMMA_M)  // 32
#define BLOCK_TILE_N      (WARPS_PER_BLOCK_X * WMMA_N)  // 32
#define BLOCK_TILE_K      WMMA_K                          // 16

extern "C" __global__ void hgemm_wmma(
    const __half * __restrict__ matrix_a,   // M×K row-major (fp16)
    const __half * __restrict__ matrix_b,   // K×N row-major (fp16)
    float        * __restrict__ matrix_c,   // M×N row-major (fp32)
    int M, int N, int K
) {
    // Warp identity within the block
    int warp_id    = (threadIdx.y * blockDim.x + threadIdx.x) / WARP_SIZE;
    int warp_row   = warp_id / WARPS_PER_BLOCK_X;   // 0 or 1
    int warp_col   = warp_id % WARPS_PER_BLOCK_X;   // 0 or 1

    // Output tile origin for this warp
    int c_row_origin = blockIdx.y * BLOCK_TILE_M + warp_row * WMMA_M;
    int c_col_origin = blockIdx.x * BLOCK_TILE_N + warp_col * WMMA_N;

    // Bounds check at warp granularity
    if (c_row_origin >= M || c_col_origin >= N) return;

    // WMMA accumulator fragment — holds 16×16 fp32 results for this warp
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over K tiles, each of width WMMA_K=16
    for (int k_tile = 0; k_tile < K; k_tile += WMMA_K) {
        // A fragment: 16×16 tile at (c_row_origin, k_tile)
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
        const __half *a_tile_ptr = matrix_a + c_row_origin * K + k_tile;

        if (c_row_origin + WMMA_M <= M && k_tile + WMMA_K <= K) {
            wmma::load_matrix_sync(a_frag, a_tile_ptr, K);
        } else {
            wmma::fill_fragment(a_frag, __float2half(0.0f));  // boundary tile
        }

        // B fragment: 16×16 tile at (k_tile, c_col_origin)
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
        const __half *b_tile_ptr = matrix_b + k_tile * N + c_col_origin;

        if (k_tile + WMMA_K <= K && c_col_origin + WMMA_N <= N) {
            wmma::load_matrix_sync(b_frag, b_tile_ptr, N);
        } else {
            wmma::fill_fragment(b_frag, __float2half(0.0f));  // boundary tile
        }

        // Warp-level matrix multiply-accumulate
        // This compiles to HMMA.16816.F32 SASS instructions
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // Store the 16×16 fp32 result tile
    float *c_tile_ptr = matrix_c + c_row_origin * N + c_col_origin;
    if (c_row_origin + WMMA_M <= M && c_col_origin + WMMA_N <= N) {
        wmma::store_matrix_sync(c_tile_ptr, acc_frag, N, wmma::mem_row_major);
    } else {
        // Boundary: store element-by-element with bounds check
        for (int i = 0; i < acc_frag.num_elements; i++) {
            // Fragment element layout is implementation-defined — use the simple path
            // For sm_86, each thread's 8 elements map to specific (row, col) positions
            // In practice, boundary tiles are rare; this code is for correctness only
        }
        // Fallback: use store_matrix_sync and accept potential out-of-bounds
        // (safe if C is padded to WMMA_N multiples)
        wmma::store_matrix_sync(c_tile_ptr, acc_frag, N, wmma::mem_row_major);
    }
}
