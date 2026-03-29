/*
 * resblock_fused.cu — Fused ResNet Block for Diffusion UNet
 *
 * Implements the core ResNet block from Stable Diffusion's UNet:
 *
 *   x_in  ─→ GroupNorm ─→ SiLU ─→ Conv2d(3×3) ─→ GroupNorm ─→ SiLU ─→ Conv2d(3×3) ─→ (+) ─→ x_out
 *                                                                                         ↑
 *                                                                                       x_in (skip)
 *
 * This kernel file contains two fused operations to minimize kernel launch overhead:
 *
 *   groupnorm_silu_fused:
 *       Fuses GroupNorm + SiLU into a single kernel pass.
 *       Computes: Y = silu( gamma * (X - mean) / sqrt(var + eps) + beta )
 *       Single read of X, single write of Y — no intermediate buffer needed.
 *       SASS: SHFL.BFLY + MUFU.RSQ + MUFU.EX2 + MUFU.RCP + FFMA
 *
 *   residual_add:
 *       Element-wise: Y[i] = X_main[i] + X_skip[i]
 *       SASS: LDG.E (2×) + FADD + STG.E
 *
 * The Conv2d 3×3 step uses conv2d.sm_86.cubin from ../conv2d/ (separate kernel).
 * The full ResNet block forward pass is:
 *   1. groupnorm_silu_fused  (GroupNorm + SiLU, pass 1)
 *   2. conv2d_nhwc           (3×3 convolution, pass 1)
 *   3. groupnorm_silu_fused  (GroupNorm + SiLU, pass 2)
 *   4. conv2d_nhwc           (3×3 convolution, pass 2)
 *   5. residual_add          (skip connection)
 *
 * Key SASS summary across the full block:
 *   SHFL.BFLY  — warp Welford reduction (GroupNorm, ×2)
 *   MUFU.RSQ   — rsqrtf(var + eps) (GroupNorm, ×2)
 *   MUFU.EX2   — exp2f for SiLU sigmoid (×2)
 *   MUFU.RCP   — reciprocal for SiLU sigmoid (×2)
 *   FFMA       — Conv2d inner loop (×2, 9×unrolled)
 *   FADD       — residual add
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o resblock.sm_86.cubin resblock_fused.cu
 *   cuobjdump -sass resblock.sm_86.cubin | grep -E 'SHFL|MUFU|FADD'
 */

#define WARP_SIZE 32

// -----------------------------------------------------------------------
// Warp Welford combine (same as groupnorm.cu)
// -----------------------------------------------------------------------
__device__ __forceinline__
void welford_combine(
    float count_a, float mean_a, float m2_a,
    float count_b, float mean_b, float m2_b,
    float &count_out, float &mean_out, float &m2_out
) {
    float combined_count = count_a + count_b;
    if (combined_count == 0.0f) {
        count_out = 0.0f; mean_out = 0.0f; m2_out = 0.0f; return;
    }
    float delta = mean_b - mean_a;
    mean_out  = mean_a + delta * (count_b / combined_count);
    m2_out    = m2_a + m2_b + delta * delta * (count_a * count_b / combined_count);
    count_out = combined_count;
}

// -----------------------------------------------------------------------
// Kernel: groupnorm_silu_fused
//
// Fuses GroupNorm normalization + SiLU activation into one kernel:
//   normalized = (X - mean) / sqrt(var + eps)
//   scaled     = gamma[c] * normalized + beta[c]
//   Y          = scaled * sigmoid(scaled)        ← SiLU
//
// Where sigmoid(x) = 1 / (1 + exp(-x)) = 1 / (1 + exp2(-x * log2e))
//
// Grid:  (N * num_groups, 1, 1)  — one block per (sample, group)
// Block: (WARP_SIZE, 1, 1)
//
// Parameters same as groupnorm.cu but output is SiLU-activated.
// -----------------------------------------------------------------------
#define LOG2E  1.4426950408889634f

extern "C" __global__ void groupnorm_silu_fused(
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

    int block_id  = blockIdx.x;
    int sample_n  = block_id / num_groups;
    int group_g   = block_id % num_groups;
    int lane      = threadIdx.x;

    int channels_per_group = C / num_groups;
    int group_size = channels_per_group * H * W;
    int channel_base = group_g * channels_per_group;

    // ---- Phase 1: Welford reduction for group mean + variance ----
    float welford_count = 0.0f;
    float welford_mean  = 0.0f;
    float welford_m2    = 0.0f;

    size_t sample_base = (size_t)sample_n * H * W * C;

    for (int group_elem = lane; group_elem < group_size; group_elem += WARP_SIZE) {
        int local_c  = group_elem % channels_per_group;
        int spatial   = group_elem / channels_per_group;
        int global_c  = channel_base + local_c;
        float val = X[sample_base + (size_t)spatial * C + global_c];

        welford_count += 1.0f;
        float delta  = val - welford_mean;
        welford_mean += delta / welford_count;
        welford_m2   += delta * (val - welford_mean);
    }

    // ---- Phase 2: Warp SHFL.BFLY reduction ----
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float pc = __shfl_xor_sync(0xFFFFFFFF, welford_count, offset);
        float pm = __shfl_xor_sync(0xFFFFFFFF, welford_mean,  offset);
        float pp = __shfl_xor_sync(0xFFFFFFFF, welford_m2,    offset);
        welford_combine(welford_count, welford_mean, welford_m2,
                        pc, pm, pp,
                        welford_count, welford_mean, welford_m2);
    }

    float group_rsqrt = rsqrtf(welford_m2 / (float)group_size + epsilon);  // MUFU.RSQ

    if (lane == 0) {
        smem_mean  = welford_mean;
        smem_rsqrt = group_rsqrt;
    }
    __syncwarp();
    float mean_val  = smem_mean;
    float rsqrt_val = smem_rsqrt;

    // ---- Phase 3: Normalize + scale/shift + SiLU ----
    for (int group_elem = lane; group_elem < group_size; group_elem += WARP_SIZE) {
        int local_c  = group_elem % channels_per_group;
        int spatial   = group_elem / channels_per_group;
        int global_c  = channel_base + local_c;
        size_t flat   = sample_base + (size_t)spatial * C + global_c;

        // GroupNorm: normalize + affine transform
        float normalized = (X[flat] - mean_val) * rsqrt_val;
        float scaled     = gamma[global_c] * normalized + beta[global_c];  // FFMA

        // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
        // sigmoid(x) = 1 / (1 + exp2(-x * log2e)) → MUFU.EX2 + MUFU.RCP
        float sigmoid_val = 1.0f / (1.0f + exp2f(-scaled * LOG2E));
        Y[flat] = scaled * sigmoid_val;
    }
}

// -----------------------------------------------------------------------
// Kernel: residual_add
//
// Element-wise addition for the skip connection.
//   Y[i] = main[i] + skip[i]
//
// Both tensors must have the same shape [N × H × W × C] (NHWC).
// SASS: LDG.E (×2) + FADD + STG.E
//
// Grid:  (ceil(total / BLOCK_SIZE), 1, 1)
// Block: (BLOCK_SIZE, 1, 1)
// -----------------------------------------------------------------------
#define BLOCK_SIZE 256

extern "C" __global__ void residual_add(
    const float * __restrict__ main_branch,
    const float * __restrict__ skip_connection,
    float       * __restrict__ Y,
    int total_elements
) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= total_elements) return;
    Y[idx] = main_branch[idx] + skip_connection[idx];  // FADD
}
