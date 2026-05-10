/*
 * groupnorm.cu — Group Normalization for diffusion model feature maps
 *
 * Group Normalization (Wu & He, 2018) normalizes over subsets of channels
 * (groups) within each spatial position and sample. Unlike BatchNorm, it
 * doesn't depend on batch size — critical for diffusion models where memory
 * constraints often force batch_size=1 or 2.
 *
 * Formula:
 *   For input X [N, C, H, W], with G groups of C/G channels:
 *   Group g for sample n contains channels [g*(C/G) .. (g+1)*(C/G) - 1]
 *   across all spatial positions H×W.
 *
 *   mean[n, g]     = sum(X[n, c, h, w]) / (C/G * H * W)
 *   var[n, g]      = sum((X - mean)^2)  / (C/G * H * W)
 *   X_norm[n,c,h,w] = (X - mean) / sqrt(var + eps)
 *   Y[n,c,h,w]      = gamma[c] * X_norm + beta[c]
 *
 * Key SASS instructions:
 *   SHFL.BFLY  — warp-level parallel Welford reduction (mean + variance)
 *   MUFU.RSQ   — fast 1/sqrt(var + eps) normalization
 *   FFMA       — fused multiply-add for gamma * normalized + beta
 *
 * Kernel design:
 *   One thread block per (sample, group).
 *   Grid: (N × G, 1, 1)
 *   Block: (WARP_SIZE, 1, 1) — one warp computes the group statistics.
 *
 *   Each thread processes ELEMENTS_PER_THREAD = (C/G * H * W) / WARP_SIZE elements
 *   using the parallel Welford algorithm (same as layernorm from Phase 2).
 *
 * Shared memory:
 *   Used for warp-level broadcast of mean/rsqrt — same pattern as layernorm_block.
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o groupnorm.sm_86.cubin groupnorm.cu
 *   cuobjdump -sass groupnorm.sm_86.cubin | grep -E 'SHFL|MUFU'
 *   → SHFL.BFLY  (Welford warp reduction)
 *   → MUFU.RSQ   (reciprocal sqrt)
 *   → MUFU.RCP   (reciprocal for mean normalization factor)
 */

#define WARP_SIZE  32

// -----------------------------------------------------------------------
// Warp-level parallel Welford combination
// Combines two running statistics (count, mean, M2) from two lane groups.
// Identical to the layernorm_warp implementation in Phase 2.
// -----------------------------------------------------------------------
__device__ __forceinline__
void welford_combine(
    float count_a, float mean_a, float m2_a,
    float count_b, float mean_b, float m2_b,
    float &count_out, float &mean_out, float &m2_out
) {
    float combined_count = count_a + count_b;
    if (combined_count == 0.0f) { count_out = 0.0f; mean_out = 0.0f; m2_out = 0.0f; return; }
    float delta  = mean_b - mean_a;
    mean_out     = mean_a + delta * (count_b / combined_count);
    m2_out       = m2_a + m2_b + delta * delta * (count_a * count_b / combined_count);
    count_out    = combined_count;
}

// -----------------------------------------------------------------------
// Kernel: groupnorm
//
// NHWC layout: X[n][h][w][c] — channel-last, standard for conv outputs.
// Each block processes one (sample n, group g) pair.
//
// Parameters:
//   X:        [N × H × W × C] FP32, NHWC layout
//   gamma:    [C] learned scale
//   beta:     [C] learned bias
//   Y:        [N × H × W × C] FP32 output
//   N, C, H, W:  tensor dimensions
//   num_groups:  G (must divide C evenly)
//   epsilon:     normalization stability constant (typical: 1e-5)
//
// Grid:  (N * num_groups, 1, 1)
// Block: (WARP_SIZE, 1, 1)
// -----------------------------------------------------------------------
extern "C" __global__ void groupnorm(
    const float * __restrict__ X,
    const float * __restrict__ gamma,
    const float * __restrict__ beta,
    float       * __restrict__ Y,
    int N, int C, int H, int W,
    int num_groups,
    float epsilon
) {
    // Shared memory: broadcast mean and rsqrt to all threads after warp reduction
    __shared__ float smem_mean;
    __shared__ float smem_rsqrt;

    int block_id  = blockIdx.x;          // 0 .. N*num_groups - 1
    int sample_n  = block_id / num_groups;
    int group_g   = block_id % num_groups;
    int lane      = threadIdx.x;

    int channels_per_group = C / num_groups;

    // Total elements in this (sample, group): channels_per_group × H × W
    int group_size = channels_per_group * H * W;

    // Channel offset for this group
    int channel_base = group_g * channels_per_group;

    // ---- Phase 1: Compute group mean and variance via parallel Welford ----
    //
    // Each warp thread accumulates ELEMENTS_PER_THREAD = group_size / WARP_SIZE elements.
    // Elements are strided: thread t processes indices t, t+32, t+64, ...
    // For NHWC layout: element at (n, h, w, c) has flat index n*H*W*C + h*W*C + w*C + c.
    // For group g, channels [channel_base .. channel_base + channels_per_group - 1].
    // We iterate over all (h, w, c) in this group.

    float welford_count = 0.0f;
    float welford_mean  = 0.0f;
    float welford_m2    = 0.0f;

    size_t sample_base_nhwc = (size_t)sample_n * H * W * C;

    for (int group_elem = lane; group_elem < group_size; group_elem += WARP_SIZE) {
        // group_elem = local index within this group's (channels_per_group × H × W) block
        int local_c = group_elem % channels_per_group;  // channel within group
        int spatial  = group_elem / channels_per_group; // spatial index (h*W + w)
        int global_c = channel_base + local_c;           // absolute channel

        // NHWC flat index: sample_base + spatial * C + global_c
        float val = X[sample_base_nhwc + (size_t)spatial * C + global_c];

        // Welford online update for this thread's running accumulator
        welford_count += 1.0f;
        float delta  = val - welford_mean;
        welford_mean += delta / welford_count;
        float delta2 = val - welford_mean;
        welford_m2   += delta * delta2;
    }

    // ---- Phase 2: Warp-level Welford reduction via SHFL.BFLY ----
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
    // welford_mean = group mean, welford_m2 / group_size = group variance
    // All lanes now hold identical values (SHFL.BFLY ensures broadcast).

    float group_variance = welford_m2 / (float)group_size;
    float group_rsqrt    = rsqrtf(group_variance + epsilon);  // MUFU.RSQ

    // Broadcast to shared memory (lane 0 writes, all lanes read)
    if (lane == 0) {
        smem_mean   = welford_mean;
        smem_rsqrt  = group_rsqrt;
    }
    __syncwarp();
    float mean_val   = smem_mean;
    float rsqrt_val  = smem_rsqrt;

    // ---- Phase 3: Normalize, scale, shift ----
    // Y[n, h, w, c] = gamma[c] * (X - mean) * rsqrt(var + eps) + beta[c]
    for (int group_elem = lane; group_elem < group_size; group_elem += WARP_SIZE) {
        int local_c  = group_elem % channels_per_group;
        int spatial   = group_elem / channels_per_group;
        int global_c  = channel_base + local_c;
        size_t flat   = sample_base_nhwc + (size_t)spatial * C + global_c;

        float normalized = (X[flat] - mean_val) * rsqrt_val;
        Y[flat] = gamma[global_c] * normalized + beta[global_c];  // FFMA
    }
}

// -----------------------------------------------------------------------
// Kernel: groupnorm_nchw
//
// NCHW layout: X[n][c][h][w] — channel-first, standard for cuDNN kernels.
// Same algorithm, different memory access pattern.
// -----------------------------------------------------------------------
extern "C" __global__ void groupnorm_nchw(
    const float * __restrict__ X,
    const float * __restrict__ gamma,
    const float * __restrict__ beta,
    float       * __restrict__ Y,
    int N, int C, int H, int W,
    int num_groups,
    float epsilon
) {
    __shared__ float smem_mean;
    __shared__ float smem_rsqrt;

    int block_id         = blockIdx.x;
    int sample_n         = block_id / num_groups;
    int group_g          = block_id % num_groups;
    int lane             = threadIdx.x;
    int channels_per_group = C / num_groups;
    int group_size         = channels_per_group * H * W;
    int channel_base       = group_g * channels_per_group;

    // For NCHW: X[n][c][h][w] = X[n*C*H*W + c*H*W + hw]
    size_t sample_base_nchw = (size_t)sample_n * C * H * W;

    float welford_count = 0.0f, welford_mean = 0.0f, welford_m2 = 0.0f;

    for (int group_elem = lane; group_elem < group_size; group_elem += WARP_SIZE) {
        int local_c   = group_elem / (H * W);
        int hw        = group_elem % (H * W);
        int global_c  = channel_base + local_c;
        float val     = X[sample_base_nchw + (size_t)global_c * H * W + hw];

        welford_count += 1.0f;
        float delta  = val - welford_mean;
        welford_mean += delta / welford_count;
        welford_m2   += delta * (val - welford_mean);
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float pc = __shfl_xor_sync(0xFFFFFFFF, welford_count, offset);
        float pm = __shfl_xor_sync(0xFFFFFFFF, welford_mean,  offset);
        float pp = __shfl_xor_sync(0xFFFFFFFF, welford_m2,    offset);
        welford_combine(welford_count, welford_mean, welford_m2,
                        pc, pm, pp,
                        welford_count, welford_mean, welford_m2);
    }

    float rsqrt_var = rsqrtf(welford_m2 / (float)group_size + epsilon);

    if (lane == 0) { smem_mean = welford_mean; smem_rsqrt = rsqrt_var; }
    __syncwarp();
    float mean_val = smem_mean; float rsqrt_val = smem_rsqrt;

    for (int group_elem = lane; group_elem < group_size; group_elem += WARP_SIZE) {
        int local_c   = group_elem / (H * W);
        int hw        = group_elem % (H * W);
        int global_c  = channel_base + local_c;
        size_t flat   = sample_base_nchw + (size_t)global_c * H * W + hw;
        Y[flat] = gamma[global_c] * ((X[flat] - mean_val) * rsqrt_val) + beta[global_c];
    }
}
