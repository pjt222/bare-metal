/*
 * activations.cu — GELU and SiLU using MUFU hardware units
 *
 * Key SASS instructions:
 *   MUFU.TANH    — fast tanh (used in GELU approximation)
 *   MUFU.EX2     — fast 2^x (used in SiLU's sigmoid: 1/(1+exp(-x)))
 *   MUFU.RCP     — fast reciprocal (1/(1+exp(-x)) for sigmoid)
 *   FFMA         — fused multiply-add (polynomial coefficients in GELU)
 *
 * Kernels:
 *   gelu_kernel   — approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
 *   silu_kernel   — SiLU: x * sigmoid(x) = x / (1 + exp(-x))
 *   gelu_fast     — fast GELU: x * sigmoid(1.702 * x)  [simpler, GPU-friendly]
 *
 * The MUFU unit runs at 1/4 the throughput of FFMA — one MUFU instruction
 * per 4 cycles. High-throughput activation kernels interleave MUFU calls
 * with FFMA work for adjacent elements to fill the pipeline.
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 --use_fast_math -o activations.sm_86.cubin activations.cu
 *   cuobjdump -sass activations.sm_86.cubin | grep MUFU
 *   → MUFU.TANH (gelu)   ← requires --use_fast_math (tanhf → MUFU.TANH only with fast math)
 *   → MUFU.EX2  (silu, gelu_fast)
 *   → MUFU.RCP  (silu, gelu_fast — reciprocal for sigmoid)
 *
 * Note: Without --use_fast_math, tanhf compiles to a multi-instruction software
 * approximation using MUFU.EX2 (tanh via (e^2x-1)/(e^2x+1)). --use_fast_math
 * enables the single MUFU.TANH instruction, at the cost of ~1 ULP precision.
 */

#define LOG2E  1.4426950408889634f   // log2(e) = 1/ln(2)
#define SQRT_2_OVER_PI 0.7978845608f // sqrt(2/pi)
#define GELU_COEFF     0.044715f     // tanh approximation coefficient

// -----------------------------------------------------------------------
// Kernel 1: GELU — approximate form using MUFU.TANH
//
// GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// The tanh is a single MUFU.TANH instruction on Ampere.
// The polynomial (x + coeff * x^3) is 2 FFMA instructions.
//
// Grid/Block: 1D, elements per thread = ELEMENTS_PER_THREAD
// -----------------------------------------------------------------------
#define ELEMENTS_PER_THREAD 4

extern "C" __global__ void gelu_kernel(
    const float * __restrict__ input,
    float       * __restrict__ output,
    int num_elements
) {
    int thread_base = (blockIdx.x * blockDim.x + threadIdx.x) * ELEMENTS_PER_THREAD;

    #pragma unroll
    for (int element_offset = 0; element_offset < ELEMENTS_PER_THREAD; element_offset++) {
        int element_idx = thread_base + element_offset;
        if (element_idx >= num_elements) return;

        float x = input[element_idx];

        // Polynomial argument: sqrt(2/pi) * (x + 0.044715 * x^3)
        // = SQRT_2_OVER_PI * x + SQRT_2_OVER_PI * GELU_COEFF * x^3
        // In SASS: FFMA x^2 = x*x, FFMA arg = x*(1 + coeff*x^2)*sqrt(2/pi)
        float x_cubed_term = x * x * x * GELU_COEFF;    // x^3 * 0.044715
        float tanh_arg     = SQRT_2_OVER_PI * (x + x_cubed_term);

        // MUFU.TANH — Ampere hardware tanh unit
        // In C: tanhf(x) compiles to MUFU.TANH on sm_86
        float tanh_val = tanhf(tanh_arg);

        output[element_idx] = x * 0.5f * (1.0f + tanh_val);
    }
}

// -----------------------------------------------------------------------
// Kernel 2: SiLU (Swish) — MUFU.EX2 + MUFU.RCP
//
// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x)) = x * (1 / (1 + 2^(-x * log2e)))
//
// We compute sigmoid via:
//   sigmoid(x) = 1 / (1 + exp(-x)) = 1 / (1 + 2^(-x * log2(e)))
//
// In SASS:
//   FNEG  R1, R0         ; -x
//   FMUL  R1, R1, LOG2E  ; -x * log2(e)
//   MUFU.EX2 R1, R1      ; 2^(-x * log2e) = exp(-x)
//   FADD  R1, R1, 1.0    ; 1 + exp(-x)
//   MUFU.RCP R1, R1      ; sigmoid(x) = 1/(1+exp(-x))
//   FMUL  R0, R0, R1     ; x * sigmoid(x)
// -----------------------------------------------------------------------
extern "C" __global__ void silu_kernel(
    const float * __restrict__ input,
    float       * __restrict__ output,
    int num_elements
) {
    int thread_base = (blockIdx.x * blockDim.x + threadIdx.x) * ELEMENTS_PER_THREAD;

    #pragma unroll
    for (int element_offset = 0; element_offset < ELEMENTS_PER_THREAD; element_offset++) {
        int element_idx = thread_base + element_offset;
        if (element_idx >= num_elements) return;

        float x = input[element_idx];

        // sigmoid(x) = 1 / (1 + exp(-x))
        // Use exp2f(-x * log2e) to force MUFU.EX2 in SASS
        float neg_exp = exp2f(-x * LOG2E);    // exp(-x) via MUFU.EX2
        float sigmoid_val = __frcp_rn(1.0f + neg_exp);   // 1/(1+exp(-x)) via MUFU.RCP

        output[element_idx] = x * sigmoid_val;
    }
}

// -----------------------------------------------------------------------
// Kernel 3: Fast GELU — sigmoid approximation (simpler, production-friendly)
//
// fast_gelu(x) = x * sigmoid(1.702 * x)
//
// This is the approximation used by many transformer implementations.
// Avoids the tanh+polynomial of exact GELU. Uses same MUFU.EX2+RCP path as SiLU.
//
// Error vs exact GELU: max ~0.0003 (acceptable for training/inference)
// -----------------------------------------------------------------------
#define FAST_GELU_SCALE 1.702f

extern "C" __global__ void gelu_fast(
    const float * __restrict__ input,
    float       * __restrict__ output,
    int num_elements
) {
    int thread_base = (blockIdx.x * blockDim.x + threadIdx.x) * ELEMENTS_PER_THREAD;

    #pragma unroll
    for (int element_offset = 0; element_offset < ELEMENTS_PER_THREAD; element_offset++) {
        int element_idx = thread_base + element_offset;
        if (element_idx >= num_elements) return;

        float x = input[element_idx];

        // sigmoid(1.702 * x)
        float scaled_x   = FAST_GELU_SCALE * x;
        float neg_exp    = exp2f(-scaled_x * LOG2E);    // MUFU.EX2
        float sigmoid_val = __frcp_rn(1.0f + neg_exp);  // MUFU.RCP

        output[element_idx] = x * sigmoid_val;
    }
}

// -----------------------------------------------------------------------
// Kernel 4: ReLU — no MUFU, just FMAX with zero
//
// Included as baseline. SASS: FMNMX Rd, Rs, RZ, !PT  (float minmax)
// This is the simplest activation — one instruction per element.
// Useful for comparing throughput: ReLU shows the memory bandwidth ceiling.
// -----------------------------------------------------------------------
extern "C" __global__ void relu_kernel(
    const float * __restrict__ input,
    float       * __restrict__ output,
    int num_elements
) {
    int thread_base = (blockIdx.x * blockDim.x + threadIdx.x) * ELEMENTS_PER_THREAD;

    #pragma unroll
    for (int element_offset = 0; element_offset < ELEMENTS_PER_THREAD; element_offset++) {
        int element_idx = thread_base + element_offset;
        if (element_idx >= num_elements) return;

        float x = input[element_idx];
        output[element_idx] = fmaxf(x, 0.0f);   // FMNMX in SASS
    }
}
