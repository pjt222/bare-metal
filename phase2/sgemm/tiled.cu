/*
 * tiled.cu — Stage B: Tiled SGEMM with shared memory
 *
 * Uses 32×32 shared memory tiles to reuse data across threads.
 * Each tile of A and B is loaded once into shared memory,
 * then all threads in the block compute from it.
 *
 * This transforms the kernel from memory-bound to compute-bound
 * for large matrices. It's the foundation for all further optimization.
 *
 * Key SASS to study in the disassembly:
 *   LDG.E.128   — 128-bit vectorized global load (4 floats at once)
 *   STS.128     — store 4 floats to shared memory
 *   BAR.SYNC 0  — thread block barrier (wait for all STS to finish)
 *   LDS.128     — load 4 floats from shared memory
 *   FFMA        — fused multiply-add (the compute kernel)
 *   DEPBAR      — dependency barrier for async ops
 *
 * C = A * B   (M×K) * (K×N) = (M×N), all row-major
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o tiled.sm_86.cubin tiled.cu
 *   cuobjdump -sass tiled.sm_86.cubin
 */

#define TILE_SIZE 32

extern "C" __global__ void sgemm_tiled(
    const float * __restrict__ matrix_a,   // M×K row-major
    const float * __restrict__ matrix_b,   // K×N row-major
    float       * __restrict__ matrix_c,   // M×N row-major (output)
    int M, int N, int K
) {
    // Shared memory tiles — TILE_SIZE×TILE_SIZE each
    // Padding by 1 column avoids shared memory bank conflicts
    __shared__ float tile_a[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float tile_b[TILE_SIZE][TILE_SIZE + 1];

    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;
    int output_row = blockIdx.y * TILE_SIZE + thread_row;
    int output_col = blockIdx.x * TILE_SIZE + thread_col;

    float accumulator = 0.0f;

    // Loop over tiles along the K dimension
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        // Each thread loads one element of A and one element of B into shared memory
        int a_col = tile_idx * TILE_SIZE + thread_col;
        int b_row = tile_idx * TILE_SIZE + thread_row;

        tile_a[thread_row][thread_col] = (output_row < M && a_col < K)
            ? matrix_a[output_row * K + a_col]
            : 0.0f;

        tile_b[thread_row][thread_col] = (b_row < K && output_col < N)
            ? matrix_b[b_row * N + output_col]
            : 0.0f;

        // Wait for all threads to finish loading this tile
        __syncthreads();

        // Compute dot product of the tile row/column
        // This inner loop generates the FFMA instructions we want to study
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            accumulator += tile_a[thread_row][k] * tile_b[k][thread_col];
        }

        // Wait before loading the next tile (don't overwrite while others compute)
        __syncthreads();
    }

    if (output_row < M && output_col < N) {
        matrix_c[output_row * N + output_col] = accumulator;
    }
}
