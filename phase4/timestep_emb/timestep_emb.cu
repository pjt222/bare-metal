/*
 * timestep_emb.cu — Sinusoidal Timestep Embeddings for Diffusion Models
 *
 * In diffusion models (DDPM, DDIM, Stable Diffusion, etc.), the denoising network
 * needs to know what noise level (timestep) it's conditioning on.
 * This is communicated via a fixed sinusoidal embedding — the same trick as
 * positional encoding in Transformers (Vaswani et al., 2017).
 *
 * Formula:
 *   emb[t][2i]   = sin(t / 10000^(2i / d_model))
 *   emb[t][2i+1] = cos(t / 10000^(2i / d_model))
 *
 * Equivalently, define:
 *   freq[i] = exp(-log(10000) * i / (d_model/2))   for i in [0, d_model/2)
 *
 * Then:
 *   emb[t][2i]   = sin(t * freq[i])
 *   emb[t][2i+1] = cos(t * freq[i])
 *
 * Key SASS instructions:
 *   MUFU.SIN  — hardware sine unit (maps from sinf with --use_fast_math)
 *   MUFU.COS  — hardware cosine unit
 *   MUFU.EX2  — exp2f for the frequency computation (exp(-x) = 2^(-x * log2e))
 *   MUFU.RCP  — reciprocal (used internally by some frequency formulations)
 *
 * NOTE: MUFU.SIN and MUFU.COS require --use_fast_math at compile time.
 *       Plain sinf/cosf without --use_fast_math produces multi-instruction software
 *       approximations (using polynomial approximation, not the MUFU unit).
 *
 * Two kernels:
 *   timestep_emb_batch  — for training: multiple timesteps, one row per t
 *   timestep_emb_single — for inference: single timestep t (scalar input)
 *
 * Grid/Block (batch version):
 *   Grid:  (batch_size, 1, 1)
 *   Block: (d_model/2, 1, 1)  — one thread per frequency pair (sin+cos)
 *   Each thread computes emb[t][2i] and emb[t][2i+1].
 *
 * Typical parameters:
 *   d_model = 256 (minimum), 512, 1024 (SD uses 320)
 *   batch_size = 1..512 (training) or 1..2 (inference with CFG)
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 --use_fast_math -o timestep_emb.sm_86.cubin timestep_emb.cu
 *   cuobjdump -sass timestep_emb.sm_86.cubin | grep MUFU
 *   → MUFU.SIN   (from sinf with --use_fast_math)
 *   → MUFU.COS   (from cosf with --use_fast_math)
 *   → MUFU.EX2   (from exp2f for frequency computation)
 */

#include <math.h>

// log(10000) = 9.21034...
// Used to compute frequencies: freq[i] = exp(-log(10000) * i / (d/2))
#define LOG_MAX_PERIOD  9.210340371976183f

// log2(e) for exp2f-based natural exponential
#define LOG2E  1.4426950408889634f

// -----------------------------------------------------------------------
// Kernel 1: timestep_emb_batch
//
// Computes sinusoidal embeddings for a batch of timesteps.
//
// Input:
//   timesteps:  [batch_size] float — diffusion timesteps (e.g., 0..1000)
//   output:     [batch_size × d_model] float — sinusoidal embeddings
//   d_model:    embedding dimension (must be even; thread count = d_model/2)
//   batch_size: number of timesteps in this batch
//
// Launch:
//   Grid:  (batch_size, 1, 1)
//   Block: (d_model/2, 1, 1)  ← one thread per (sin, cos) pair
// -----------------------------------------------------------------------
extern "C" __global__ void timestep_emb_batch(
    const float * __restrict__ timesteps,   // [batch_size]
    float       * __restrict__ output,      // [batch_size × d_model]
    int d_model,
    int batch_size
) {
    int batch_idx = blockIdx.x;   // which timestep in the batch
    int half_dim  = d_model / 2;  // number of (sin,cos) pairs
    int freq_idx  = threadIdx.x;  // which frequency (0 .. half_dim-1)

    if (batch_idx >= batch_size || freq_idx >= half_dim) return;

    float timestep_val = timesteps[batch_idx];

    // Compute frequency: freq = exp(-log(10000) * freq_idx / half_dim)
    // Using exp2f for MUFU.EX2: exp(-x) = exp2(-x * log2e)
    float log_freq      = -LOG_MAX_PERIOD * (float)freq_idx / (float)half_dim;
    float frequency     = exp2f(log_freq * LOG2E);   // MUFU.EX2

    // Argument for sin/cos: t * freq
    float angle = timestep_val * frequency;

    // Compute sin and cos — MUFU.SIN and MUFU.COS with --use_fast_math
    float sin_val = sinf(angle);   // MUFU.SIN
    float cos_val = cosf(angle);   // MUFU.COS

    // Store to output: [batch_idx, freq_idx] = sin, [batch_idx, freq_idx + half_dim] = cos
    float *out_row = output + (size_t)batch_idx * d_model;
    out_row[freq_idx]            = sin_val;
    out_row[freq_idx + half_dim] = cos_val;
}

// -----------------------------------------------------------------------
// Kernel 2: timestep_emb_single
//
// Computes the embedding for a single timestep t.
// Used in inference where we process one image at a time (or 2 with CFG).
//
// Grid:  (1, 1, 1)
// Block: (d_model/2, 1, 1)
// -----------------------------------------------------------------------
extern "C" __global__ void timestep_emb_single(
    float   timestep_val,     // the scalar timestep value
    float * __restrict__ output,  // [d_model] output embedding
    int d_model
) {
    int half_dim = d_model / 2;
    int freq_idx = threadIdx.x;   // 0 .. half_dim-1

    if (freq_idx >= half_dim) return;

    float log_freq  = -LOG_MAX_PERIOD * (float)freq_idx / (float)half_dim;
    float frequency = exp2f(log_freq * LOG2E);   // MUFU.EX2
    float angle     = timestep_val * frequency;

    output[freq_idx]            = sinf(angle);   // MUFU.SIN
    output[freq_idx + half_dim] = cosf(angle);   // MUFU.COS
}

// -----------------------------------------------------------------------
// Kernel 3: timestep_emb_learned_scale
//
// Extended version used in many diffusion models: after the base sinusoidal
// embedding, a learned linear projection scales and shifts the embedding.
// This fused kernel avoids a second kernel launch for the linear layer.
//
// output[i] = silu(W1 @ emb + b1)[i] where W1 and b1 are learned params.
// Here we implement just the first half: computing emb and applying a
// per-channel scale + bias (pointwise, learned at training time).
//
// This demonstrates the MUFU.TANH path (via SiLU = x * sigmoid(x) where
// sigmoid uses MUFU.EX2) in the context of diffusion conditioning.
//
// Grid:  (batch_size, 1, 1)
// Block: (d_model/2, 1, 1)
// -----------------------------------------------------------------------
extern "C" __global__ void timestep_emb_with_silu_scale(
    const float * __restrict__ timesteps,   // [batch_size]
    const float * __restrict__ scale,       // [d_model] learned scale (from linear layer)
    const float * __restrict__ bias,        // [d_model] learned bias
    float       * __restrict__ output,      // [batch_size × d_model]
    int d_model,
    int batch_size
) {
    int batch_idx = blockIdx.x;
    int half_dim  = d_model / 2;
    int freq_idx  = threadIdx.x;

    if (batch_idx >= batch_size || freq_idx >= half_dim) return;

    float timestep_val = timesteps[batch_idx];
    float log_freq     = -LOG_MAX_PERIOD * (float)freq_idx / (float)half_dim;
    float frequency    = exp2f(log_freq * LOG2E);
    float angle        = timestep_val * frequency;

    // Sinusoidal embedding
    float sin_val = sinf(angle);
    float cos_val = cosf(angle);

    // Apply learned scale and bias
    float *out_row = output + (size_t)batch_idx * d_model;
    float raw_sin = sin_val * scale[freq_idx]            + bias[freq_idx];
    float raw_cos = cos_val * scale[freq_idx + half_dim] + bias[freq_idx + half_dim];

    // SiLU activation: x * sigmoid(x) = x * 1/(1 + exp(-x))
    // sigmoid(x) = 1/(1 + exp2(-x * log2e)) → MUFU.EX2 + MUFU.RCP
    // With --use_fast_math, this may also use MUFU.TANH via: silu(x) = x*sigmoid(x)
    // using tanhf based sigmoid approximation
    float sig_sin = 1.0f / (1.0f + exp2f(-raw_sin * LOG2E));  // MUFU.EX2 + MUFU.RCP
    float sig_cos = 1.0f / (1.0f + exp2f(-raw_cos * LOG2E));

    out_row[freq_idx]            = raw_sin * sig_sin;   // SiLU
    out_row[freq_idx + half_dim] = raw_cos * sig_cos;
}
