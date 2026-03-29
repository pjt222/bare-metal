/*
 * igemm.cu — INT8 GEMM using Tensor Cores (WMMA API → IMMA SASS)
 *
 * Uses CUDA's warp-level matrix multiply-accumulate (WMMA) API with INT8 inputs
 * and INT32 accumulators. The INT32 results are dequantized to FP32 output.
 *
 * Why INT8 Tensor Cores?
 *   FP16 Tensor peak:  ~174  TFLOPS  (HMMA.16816.F32)
 *   INT8 Tensor peak:  ~696  TOPS    (IMMA.8816.S8 — 4× throughput, same silicon)
 *
 * INT8 is the inference workhorse: weights and activations quantized to 8-bit,
 * accumulated in INT32, then dequantized back to FP32 for the next layer.
 *
 * WMMA fragment layout on sm_86 (Ampere, 16×16×16 tile, INT8):
 *   A fragment: 16×16 signed char matrix, distributed across 32 threads
 *               Each thread holds 8 int8 values (2 registers of 4×int8 packed)
 *   B fragment: Same — 16×16 signed char, 8 values per thread
 *   C/D accum:  16×16 int32, each thread holds 8 int32 values (8 registers)
 *
 * The SASS we want to see:
 *   IMMA.8816.S8 Rd, Ra, Rb, Rc   (INT8 in, INT32 accumulation)
 *   Each WMMA mma_sync decomposes to 4 IMMA.8816.S8 sub-operations.
 *
 * Quantization: symmetric per-tensor
 *   scale = max(abs(tensor)) / 127
 *   quantized = clamp(round(value / scale), -128, 127)
 *   dequantized output: C_fp32 = C_int32 * scale_a * scale_b
 *
 * C = A * B, A: M×K (int8), B: K×N (int8), C: M×N (fp32, dequantized)
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o igemm.sm_86.cubin igemm.cu
 *   cuobjdump -sass igemm.sm_86.cubin | grep IMMA
 */

#include <mma.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define WARPS_PER_BLOCK_X 2
#define WARPS_PER_BLOCK_Y 2
#define WARP_SIZE         32
#define BLOCK_X           (WARPS_PER_BLOCK_X * WARP_SIZE)  // 64
#define BLOCK_Y           WARPS_PER_BLOCK_Y                 // 2

#define BLOCK_TILE_M      (WARPS_PER_BLOCK_Y * WMMA_M)  // 32
#define BLOCK_TILE_N      (WARPS_PER_BLOCK_X * WMMA_N)  // 32

extern "C" __global__ void igemm_wmma(
    const signed char * __restrict__ matrix_a,   // M×K row-major (int8)
    const signed char * __restrict__ matrix_b,   // K×N row-major (int8)
    float             * __restrict__ matrix_c,   // M×N row-major (fp32, dequantized)
    int M, int N, int K,
    float scale_a,
    float scale_b
) {
    int warp_id    = (threadIdx.y * blockDim.x + threadIdx.x) / WARP_SIZE;
    int warp_row   = warp_id / WARPS_PER_BLOCK_X;
    int warp_col   = warp_id % WARPS_PER_BLOCK_X;

    int c_row_origin = blockIdx.y * BLOCK_TILE_M + warp_row * WMMA_M;
    int c_col_origin = blockIdx.x * BLOCK_TILE_N + warp_col * WMMA_N;

    if (c_row_origin >= M || c_col_origin >= N) return;

    // INT32 accumulator fragment — holds 16×16 integer results for this warp
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> acc_frag;
    wmma::fill_fragment(acc_frag, 0);

    // Loop over K tiles, each of width WMMA_K=16
    for (int k_tile = 0; k_tile < K; k_tile += WMMA_K) {
        // A fragment: 16×16 tile at (c_row_origin, k_tile)
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                       signed char, wmma::row_major> a_frag;
        const signed char *a_tile_ptr = matrix_a + c_row_origin * K + k_tile;

        if (c_row_origin + WMMA_M <= M && k_tile + WMMA_K <= K) {
            wmma::load_matrix_sync(a_frag, a_tile_ptr, K);
        } else {
            wmma::fill_fragment(a_frag, (signed char)0);
        }

        // B fragment: 16×16 tile at (k_tile, c_col_origin)
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                       signed char, wmma::row_major> b_frag;
        const signed char *b_tile_ptr = matrix_b + k_tile * N + c_col_origin;

        if (k_tile + WMMA_K <= K && c_col_origin + WMMA_N <= N) {
            wmma::load_matrix_sync(b_frag, b_tile_ptr, N);
        } else {
            wmma::fill_fragment(b_frag, (signed char)0);
        }

        // Integer Tensor Core: IMMA.8816.S8 in SASS (4 sub-ops per mma_sync)
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // Dequantize: INT32 accumulator → FP32 output
    // C_fp32[i] = C_int32[i] * scale_a * scale_b
    float dequant_scale = scale_a * scale_b;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> float_acc;
    #pragma unroll
    for (int i = 0; i < acc_frag.num_elements; i++) {
        float_acc.x[i] = (float)acc_frag.x[i] * dequant_scale;
    }

    // Store 16×16 FP32 result tile
    float *c_tile_ptr = matrix_c + c_row_origin * N + c_col_origin;
    if (c_row_origin + WMMA_M <= M && c_col_origin + WMMA_N <= N) {
        wmma::store_matrix_sync(c_tile_ptr, float_acc, N, wmma::mem_row_major);
    } else {
        wmma::store_matrix_sync(c_tile_ptr, float_acc, N, wmma::mem_row_major);
    }
}
