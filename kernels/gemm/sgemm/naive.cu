/*
 * naive.cu — Stage A: Naive FP32 SGEMM
 *
 * One thread per output element. No tiling, no shared memory.
 * This is intentionally the worst possible implementation —
 * completely memory-bound, re-reading A and B from global memory
 * for every output element.
 *
 * Purpose: study the simplest possible GEMM SASS, understand
 * how the compiler generates the multiply-accumulate loop.
 *
 * C = A * B   where A is M×K, B is K×N, C is M×N (row-major)
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o naive.sm_86.cubin naive.cu
 *   cuobjdump -sass naive.sm_86.cubin
 */

extern "C" __global__ void sgemm_naive(
    const float * __restrict__ matrix_a,   // M×K row-major
    const float * __restrict__ matrix_b,   // K×N row-major
    float       * __restrict__ matrix_c,   // M×N row-major (output)
    int M, int N, int K
) {
    int output_col = blockIdx.x * blockDim.x + threadIdx.x;
    int output_row = blockIdx.y * blockDim.y + threadIdx.y;

    if (output_row >= M || output_col >= N) return;

    float accumulator = 0.0f;

    for (int k = 0; k < K; k++) {
        float a_element = matrix_a[output_row * K + k];
        float b_element = matrix_b[k * N + output_col];
        accumulator += a_element * b_element;
    }

    matrix_c[output_row * N + output_col] = accumulator;
}
