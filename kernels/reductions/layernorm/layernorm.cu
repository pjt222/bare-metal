/*
 * layernorm.cu — Layer Normalization using MUFU.RSQ + SHFL.BFLY reductions
 *
 * Key SASS instructions to observe:
 *   SHFL.BFLY    — butterfly shuffle for warp-level parallel reduction
 *   MUFU.RSQ     — fast hardware reciprocal square root (1/sqrt(x))
 *   MUFU.RCP     — fast reciprocal (1/N for mean computation)
 *   FFMA         — fused multiply-add for normalization
 *
 * LayerNorm formula:
 *   y_i = gamma_i * (x_i - mean) / sqrt(variance + epsilon) + beta_i
 *
 * where:
 *   mean     = (1/N) * sum(x_i)
 *   variance = (1/N) * sum((x_i - mean)^2)
 *
 * Key optimization: Welford's online algorithm computes mean and variance
 * in a single pass over the data (avoids storing intermediate values).
 *
 * Single-pass Welford algorithm per thread:
 *   For each element x, update running count, mean, and M2:
 *     count += 1
 *     delta = x - mean
 *     mean += delta / count
 *     delta2 = x - mean
 *     M2 += delta * delta2
 *   variance = M2 / count
 *
 * Parallelization: Welford can be merged across threads using the
 * parallel Welford combination formulas — no round-trip through shared memory
 * for the mean/variance computation itself.
 *
 * Two kernels:
 *   layernorm_warp  — one warp per row (row_width ≤ 32), pure SHFL
 *   layernorm_block — one block per row (row_width ≤ BLOCK_SIZE * ELEMENTS_PER_THREAD)
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o layernorm.sm_86.cubin layernorm.cu
 *   cuobjdump -sass layernorm.sm_86.cubin | grep -E 'SHFL|MUFU'
 *   → SHFL.BFLY  (warp reduction)
 *   → MUFU.RSQ   (reciprocal sqrt for normalization)
 *   → MUFU.RCP   (reciprocal for mean)
 */

#define WARP_SIZE 32

// -----------------------------------------------------------------------
// Warp-level parallel Welford combination
//
// Combines two (count, mean, M2) statistics into one.
// Used to reduce Welford accumulators across warp lanes.
// Reference: Welford 1962, Chan et al. 1979 parallel algorithm.
// -----------------------------------------------------------------------
__device__ __forceinline__
void welford_combine(
    float  count_a, float  mean_a, float  m2_a,
    float  count_b, float  mean_b, float  m2_b,
    float &count_out, float &mean_out, float &m2_out
) {
    float combined_count = count_a + count_b;
    if (combined_count == 0.0f) {
        count_out = 0.0f; mean_out = 0.0f; m2_out = 0.0f;
        return;
    }
    float delta = mean_b - mean_a;
    mean_out  = mean_a + delta * (count_b / combined_count);
    m2_out    = m2_a + m2_b + delta * delta * (count_a * count_b / combined_count);
    count_out = combined_count;
}

// -----------------------------------------------------------------------
// Kernel 1: layernorm_warp — one warp per row, row_width ≤ 32
//
// Grid:  (num_rows, 1, 1)
// Block: (32, 1, 1) — exactly one warp
// -----------------------------------------------------------------------
extern "C" __global__ void layernorm_warp(
    const float * __restrict__ input,    // [num_rows × row_width]
    const float * __restrict__ gamma,    // [row_width] — scale parameters
    const float * __restrict__ beta,     // [row_width] — shift parameters
    float       * __restrict__ output,   // [num_rows × row_width]
    int num_rows,
    int row_width,
    float epsilon
) {
    int row_index = blockIdx.x;
    int lane      = threadIdx.x;

    if (row_index >= num_rows) return;

    const float *row_in  = input  + (size_t)row_index * row_width;
    float       *row_out = output + (size_t)row_index * row_width;

    // --- Step 1: Load element, initialize Welford accumulators ---
    float element_val = (lane < row_width) ? row_in[lane] : 0.0f;

    // Each lane starts with a single-element Welford accumulator:
    // count=1, mean=element_val, M2=0 (variance of one element is 0)
    float welford_count = (lane < row_width) ? 1.0f : 0.0f;
    float welford_mean  = element_val;
    float welford_m2    = 0.0f;

    // --- Step 2: Warp-level Welford reduction via SHFL.BFLY ---
    // Each round halves the number of independent accumulators.
    // After 5 rounds, lane 0 (and all lanes due to BFLY symmetry)
    // holds the combined statistics for all elements in [0, row_width).
    #pragma unroll
    for (int reduction_offset = WARP_SIZE / 2; reduction_offset > 0; reduction_offset >>= 1) {
        // Gather stats from the symmetric partner lane
        float partner_count = __shfl_xor_sync(0xFFFFFFFF, welford_count, reduction_offset);
        float partner_mean  = __shfl_xor_sync(0xFFFFFFFF, welford_mean,  reduction_offset);
        float partner_m2    = __shfl_xor_sync(0xFFFFFFFF, welford_m2,    reduction_offset);

        // Combine this lane's accumulator with its partner's
        welford_combine(
            welford_count, welford_mean, welford_m2,
            partner_count, partner_mean, partner_m2,
            welford_count, welford_mean, welford_m2
        );
    }
    // welford_mean = row mean, welford_m2 / row_width = row variance

    float row_mean     = welford_mean;
    float row_variance = welford_m2 / (float)row_width;

    // --- Step 3: Normalize using MUFU.RSQ ---
    // rsqrt(variance + epsilon) → MUFU.RSQ in SASS
    // The GPU computes this in one cycle using the dedicated hardware unit.
    float rsqrt_var = rsqrtf(row_variance + epsilon);

    // --- Step 4: Apply scale (gamma) and shift (beta) ---
    // y = gamma * (x - mean) * rsqrt(variance + eps) + beta
    // In SASS: FFMA — fused multiply-add
    if (lane < row_width) {
        float normalized = (element_val - row_mean) * rsqrt_var;
        row_out[lane] = gamma[lane] * normalized + beta[lane];
    }
}

// -----------------------------------------------------------------------
// Kernel 2: layernorm_block — one block per row, wider rows
//
// Grid:  (num_rows, 1, 1)
// Block: (BLOCK_SIZE, 1, 1)
// Each thread processes ELEMENTS_PER_THREAD elements.
// -----------------------------------------------------------------------
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif

#define ELEMENTS_PER_THREAD 4
#define NUM_WARPS_IN_BLOCK (BLOCK_SIZE / WARP_SIZE)

extern "C" __global__ void layernorm_block(
    const float * __restrict__ input,
    const float * __restrict__ gamma,
    const float * __restrict__ beta,
    float       * __restrict__ output,
    int num_rows,
    int row_width,
    float epsilon
) {
    // Shared memory: each warp writes (count, mean, M2) for inter-warp combination
    __shared__ float smem_count[NUM_WARPS_IN_BLOCK];
    __shared__ float smem_mean[NUM_WARPS_IN_BLOCK];
    __shared__ float smem_m2[NUM_WARPS_IN_BLOCK];

    int row_index = blockIdx.x;
    int thread_id = threadIdx.x;
    int warp_id   = thread_id / WARP_SIZE;
    int lane      = thread_id % WARP_SIZE;

    if (row_index >= num_rows) return;

    const float *row_in  = input  + (size_t)row_index * row_width;
    float       *row_out = output + (size_t)row_index * row_width;

    // --- Step 1: Load elements and initialize per-thread Welford accumulators ---
    float thread_vals[ELEMENTS_PER_THREAD];
    float welford_count = 0.0f;
    float welford_mean  = 0.0f;
    float welford_m2    = 0.0f;

    #pragma unroll
    for (int element_offset = 0; element_offset < ELEMENTS_PER_THREAD; element_offset++) {
        int global_col = thread_id + element_offset * BLOCK_SIZE;
        float val = (global_col < row_width) ? row_in[global_col] : 0.0f;
        thread_vals[element_offset] = val;

        if (global_col < row_width) {
            // Online Welford update: incorporate one element at a time
            welford_count += 1.0f;
            float delta  = val - welford_mean;
            welford_mean += delta / welford_count;
            float delta2 = val - welford_mean;
            welford_m2   += delta * delta2;
        }
    }

    // --- Step 2: Warp-level Welford reduction (SHFL.BFLY) ---
    #pragma unroll
    for (int reduction_offset = WARP_SIZE / 2; reduction_offset > 0; reduction_offset >>= 1) {
        float partner_count = __shfl_xor_sync(0xFFFFFFFF, welford_count, reduction_offset);
        float partner_mean  = __shfl_xor_sync(0xFFFFFFFF, welford_mean,  reduction_offset);
        float partner_m2    = __shfl_xor_sync(0xFFFFFFFF, welford_m2,    reduction_offset);

        welford_combine(
            welford_count, welford_mean, welford_m2,
            partner_count, partner_mean, partner_m2,
            welford_count, welford_mean, welford_m2
        );
    }

    // --- Step 3: Write warp result to shared memory ---
    if (lane == 0) {
        smem_count[warp_id] = welford_count;
        smem_mean[warp_id]  = welford_mean;
        smem_m2[warp_id]    = welford_m2;
    }
    __syncthreads();

    // --- Step 4: First warp reduces all warp results ---
    if (warp_id == 0) {
        // Load warp-level results
        welford_count = (lane < NUM_WARPS_IN_BLOCK) ? smem_count[lane] : 0.0f;
        welford_mean  = (lane < NUM_WARPS_IN_BLOCK) ? smem_mean[lane]  : 0.0f;
        welford_m2    = (lane < NUM_WARPS_IN_BLOCK) ? smem_m2[lane]    : 0.0f;

        // Reduce within first warp (only NUM_WARPS_IN_BLOCK active lanes)
        #pragma unroll
        for (int reduction_offset = NUM_WARPS_IN_BLOCK / 2; reduction_offset > 0; reduction_offset >>= 1) {
            float partner_count = __shfl_xor_sync(0xFFFFFFFF, welford_count, reduction_offset);
            float partner_mean  = __shfl_xor_sync(0xFFFFFFFF, welford_mean,  reduction_offset);
            float partner_m2    = __shfl_xor_sync(0xFFFFFFFF, welford_m2,    reduction_offset);

            welford_combine(
                welford_count, welford_mean, welford_m2,
                partner_count, partner_mean, partner_m2,
                welford_count, welford_mean, welford_m2
            );
        }

        // Lane 0 has the final result — write to shared memory for broadcast
        if (lane == 0) {
            smem_mean[0] = welford_mean;
            smem_m2[0]   = welford_m2 / (float)row_width;   // store variance
        }
    }
    __syncthreads();

    float row_mean     = smem_mean[0];
    float row_variance = smem_m2[0];

    // --- Step 5: MUFU.RSQ for reciprocal sqrt ---
    float rsqrt_var = rsqrtf(row_variance + epsilon);

    // --- Step 6: Normalize, scale, shift ---
    #pragma unroll
    for (int element_offset = 0; element_offset < ELEMENTS_PER_THREAD; element_offset++) {
        int global_col = thread_id + element_offset * BLOCK_SIZE;
        if (global_col < row_width) {
            float normalized = (thread_vals[element_offset] - row_mean) * rsqrt_var;
            row_out[global_col] = gamma[global_col] * normalized + beta[global_col];
        }
    }
}
