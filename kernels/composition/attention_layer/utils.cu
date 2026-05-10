/*
 * utils.cu — Data format conversion and layout transpose kernels
 *
 * These utility kernels bridge the format gaps between bare-metal primitives:
 *   - LayerNorm outputs FP32, but HGEMM/FlashAttention need FP16 input
 *   - HGEMM outputs FP32, but FlashAttention needs FP16 input
 *   - HGEMM outputs [B*S × H*D] flat, but FlashAttention needs [B × H × S × D]
 *
 * Kernels:
 *   fp32_to_fp16      — elementwise FP32 → FP16
 *   fp16_to_fp32      — elementwise FP16 → FP32
 *   transpose_bshd    — reshape [B, S, H, D] → [B, H, S, D] (seq↔head transpose)
 *   residual_add      — elementwise C = A + B (FP32)
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o utils.sm_86.cubin utils.cu
 */

#include <cuda_fp16.h>

#define BLOCK_SIZE 256
#define ELEMS_PER_THREAD 4

// -----------------------------------------------------------------------
// fp32_to_fp16: elementwise conversion
// Grid: (ceil(n / (BLOCK_SIZE * ELEMS_PER_THREAD)), 1, 1)
// Block: (BLOCK_SIZE, 1, 1)
// -----------------------------------------------------------------------
extern "C" __global__ void fp32_to_fp16(
    const float  * __restrict__ input,
    __half       * __restrict__ output,
    int n
) {
    int base = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * ELEMS_PER_THREAD;
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        int idx = base + i;
        if (idx < n) output[idx] = __float2half(input[idx]);
    }
}

// -----------------------------------------------------------------------
// fp16_to_fp32: elementwise conversion
// Grid: (ceil(n / (BLOCK_SIZE * ELEMS_PER_THREAD)), 1, 1)
// Block: (BLOCK_SIZE, 1, 1)
// -----------------------------------------------------------------------
extern "C" __global__ void fp16_to_fp32(
    const __half * __restrict__ input,
    float        * __restrict__ output,
    int n
) {
    int base = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * ELEMS_PER_THREAD;
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        int idx = base + i;
        if (idx < n) output[idx] = __half2float(input[idx]);
    }
}

// -----------------------------------------------------------------------
// transpose_bshd: [batch, seq, heads, D_HEAD] → [batch, heads, seq, D_HEAD]
//
// Input layout:  element at (b, s, h, d) is at offset
//                b * seq * heads * D + s * heads * D + h * D + d
// Output layout: element at (b, h, s, d) is at offset
//                b * heads * seq * D + h * seq * D + s * D + d
//
// Each thread copies one element. D_HEAD is typically 64.
// Grid: (ceil(total_elements / BLOCK_SIZE), 1, 1)
// Block: (BLOCK_SIZE, 1, 1)
// -----------------------------------------------------------------------
extern "C" __global__ void transpose_bshd(
    const __half * __restrict__ input,   // [batch, seq, heads, D_HEAD]
    __half       * __restrict__ output,  // [batch, heads, seq, D_HEAD]
    int batch, int seq, int heads, int d_head
) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int total = batch * seq * heads * d_head;
    if (idx >= total) return;

    // Decompose flat index into (b, s, h, d) from input layout
    int d = idx % d_head;
    int tmp = idx / d_head;
    int h = tmp % heads;
    tmp = tmp / heads;
    int s = tmp % seq;
    int b = tmp / seq;

    // Compute output flat index for (b, h, s, d) layout
    int out_idx = ((b * heads + h) * seq + s) * d_head + d;
    output[out_idx] = input[idx];
}

// -----------------------------------------------------------------------
// transpose_bhsd: [batch, heads, seq, D_HEAD] → [batch, seq, heads, D_HEAD]
// (inverse of transpose_bshd — for reshaping FlashAttention output back)
//
// Grid: (ceil(total_elements / BLOCK_SIZE), 1, 1)
// Block: (BLOCK_SIZE, 1, 1)
// -----------------------------------------------------------------------
extern "C" __global__ void transpose_bhsd(
    const float * __restrict__ input,    // [batch, heads, seq, D_HEAD] FP32
    float       * __restrict__ output,   // [batch, seq, heads, D_HEAD] FP32
    int batch, int seq, int heads, int d_head
) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int total = batch * seq * heads * d_head;
    if (idx >= total) return;

    // Decompose flat index into (b, h, s, d) from input layout
    int d = idx % d_head;
    int tmp = idx / d_head;
    int s = tmp % seq;
    tmp = tmp / seq;
    int h = tmp % heads;
    int b = tmp / heads;

    // Compute output flat index for (b, s, h, d) layout
    int out_idx = ((b * seq + s) * heads + h) * d_head + d;
    output[out_idx] = input[idx];
}

// -----------------------------------------------------------------------
// residual_add: C[i] = A[i] + B[i] (FP32 elementwise)
// Grid: (ceil(n / (BLOCK_SIZE * ELEMS_PER_THREAD)), 1, 1)
// Block: (BLOCK_SIZE, 1, 1)
// -----------------------------------------------------------------------
extern "C" __global__ void residual_add(
    const float * __restrict__ A,
    const float * __restrict__ B,
    float       * __restrict__ C,
    int n
) {
    int base = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * ELEMS_PER_THREAD;
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        int idx = base + i;
        if (idx < n) C[idx] = A[idx] + B[idx];
    }
}
