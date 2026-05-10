/*
 * register_blocked.cu — Stage C: Register-blocked SGEMM
 *
 * The key idea: instead of one output element per thread, each thread
 * computes a 4×4 tile of output elements — 16 accumulators in registers.
 *
 * Why this matters:
 *   Tiled (Stage B): each thread does 1 FP32 load from A, 1 from B, 1 FFMA
 *                    per k step → arithmetic intensity ≈ 1 FLOP/byte
 *   Register-blocked: each thread does 4 loads from A, 4 from B, 16 FFMAs
 *                    per k step → arithmetic intensity ≈ 4 FLOP/byte
 *
 * Thread/block/grid decomposition:
 *   Output tile per block:  128 × 128  (BM × BN)
 *   Thread block:            32 ×  32  = 1024 threads
 *   Output tile per thread:   4 ×   4  = 16 accumulators
 *   Shared memory tile:      128 × 8   (BM × BK, then transposed for B)
 *
 * SASS patterns to study:
 *   - 16 FFMA instructions per k step (from the 4×4 unrolled inner product)
 *   - 4 LDS per A column, 4 LDS per B row = 8 LDS per k step
 *   - Compiler interleaves LDS + FFMA for latency hiding
 *   - Much longer FFMA chains → better utilization of the ALU pipeline
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o register_blocked.sm_86.cubin register_blocked.cu
 *   cuobjdump -sass register_blocked.sm_86.cubin | grep -c FFMA
 *   # expect ~256 (8 tiles × 4 × 4 × 2 with unrolling)
 */

// Block tile dimensions
#define BM 64     // rows of C per block
#define BN 64     // cols of C per block
#define BK 32     // k-dimension of shared memory tile (larger → fewer barriers)

// Thread tile dimensions
#define TM 4      // rows of C per thread
#define TN 4      // cols of C per thread

// Derived dimensions
#define BLOCK_ROWS (BM / TM)   // 32 threads in row direction
#define BLOCK_COLS (BN / TN)   // 32 threads in col direction

extern "C" __global__ __launch_bounds__(BLOCK_ROWS * BLOCK_COLS)
void sgemm_register_blocked(
    const float * __restrict__ matrix_a,   // M×K row-major
    const float * __restrict__ matrix_b,   // K×N row-major
    float       * __restrict__ matrix_c,   // M×N row-major (output)
    int M, int N, int K
) {
    // Shared memory — BM×BK tile of A, BK×BN tile of B
    // Padded to avoid bank conflicts
    __shared__ float smem_a[BM][BK + 1];
    __shared__ float smem_b[BK][BN + 1];

    int thread_row = threadIdx.y;  // 0..31
    int thread_col = threadIdx.x;  // 0..31

    // Base output coordinates for this thread's 4×4 tile
    int out_row_base = blockIdx.y * BM + thread_row * TM;
    int out_col_base = blockIdx.x * BN + thread_col * TN;

    // 16 accumulator registers — this is the key difference from tiled
    float acc[TM][TN] = {};   // zero-initialized, lives in registers

    // Loop over K tiles
    int num_k_tiles = (K + BK - 1) / BK;

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int k_base = k_tile * BK;

        // --- Load A tile into shared memory ---
        // BM×BK = 64×32 = 2048 elements, 256 threads → 8 elements each.
        // Stride by number of threads so all elements are covered.
        {
            int num_threads = BLOCK_ROWS * BLOCK_COLS;
            int linear_thread = thread_row * BLOCK_COLS + thread_col;
            for (int i = linear_thread; i < BM * BK; i += num_threads) {
                int smem_row = i / BK;
                int smem_col = i % BK;
                int global_row = blockIdx.y * BM + smem_row;
                int global_col = k_base + smem_col;
                smem_a[smem_row][smem_col] =
                    (global_row < M && global_col < K)
                    ? matrix_a[global_row * K + global_col]
                    : 0.0f;
            }
        }

        // --- Load B tile into shared memory ---
        // BK×BN = 32×64 = 2048 elements, 256 threads → 8 elements each.
        {
            int num_threads = BLOCK_ROWS * BLOCK_COLS;
            int linear_thread = thread_row * BLOCK_COLS + thread_col;
            for (int i = linear_thread; i < BK * BN; i += num_threads) {
                int smem_row = i / BN;
                int smem_col = i % BN;
                int global_row = k_base + smem_row;
                int global_col = blockIdx.x * BN + smem_col;
                smem_b[smem_row][smem_col] =
                    (global_row < K && global_col < N)
                    ? matrix_b[global_row * N + global_col]
                    : 0.0f;
            }
        }

        __syncthreads();

        // --- Inner loop: 4×4 register-level outer product ---
        // This is where the FFMA density happens.
        // Each k step: load 4 values from A row, 4 from B col, do 4×4 FFMAs.
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            // Load 4-element column of A tile (one per thread-row output)
            float a_frag[TM];
            #pragma unroll
            for (int tm = 0; tm < TM; tm++) {
                a_frag[tm] = smem_a[thread_row * TM + tm][k];
            }

            // Load 4-element row of B tile (one per thread-col output)
            float b_frag[TN];
            #pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                b_frag[tn] = smem_b[k][thread_col * TN + tn];
            }

            // 4×4 outer product → 16 FFMA instructions
            #pragma unroll
            for (int tm = 0; tm < TM; tm++) {
                #pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    acc[tm][tn] += a_frag[tm] * b_frag[tn];
                }
            }
        }

        __syncthreads();
    }

    // --- Write 4×4 output tile to global memory ---
    #pragma unroll
    for (int tm = 0; tm < TM; tm++) {
        #pragma unroll
        for (int tn = 0; tn < TN; tn++) {
            int out_row = out_row_base + tm;
            int out_col = out_col_base + tn;
            if (out_row < M && out_col < N) {
                matrix_c[out_row * N + out_col] = acc[tm][tn];
            }
        }
    }
}
