#pragma once
/*
 * check.h — Result verification utilities
 *
 * Compare GPU results against a CPU reference.
 * Supports FP32 and FP16 (as uint16_t bit patterns).
 */

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <algorithm>

// -----------------------------------------------------------------------
// FP32 comparison
// -----------------------------------------------------------------------
struct CheckResult {
    int   num_errors;
    int   total_elements;
    float max_abs_error;
    float max_rel_error;
    int   first_error_index;
};

inline CheckResult check_fp32(
    const float *gpu_result,
    const float *cpu_reference,
    int num_elements,
    float abs_tolerance = 1e-3f,
    float rel_tolerance = 1e-3f,
    bool print_first_error = true
) {
    CheckResult result = {0, num_elements, 0.0f, 0.0f, -1};

    for (int element_idx = 0; element_idx < num_elements; element_idx++) {
        float gpu_val = gpu_result[element_idx];
        float ref_val = cpu_reference[element_idx];
        float abs_error = fabsf(gpu_val - ref_val);
        float ref_abs   = fabsf(ref_val);
        float rel_error = (ref_abs > 1e-8f) ? (abs_error / ref_abs) : abs_error;

        result.max_abs_error = fmaxf(result.max_abs_error, abs_error);
        result.max_rel_error = fmaxf(result.max_rel_error, rel_error);

        bool is_wrong = (abs_error > abs_tolerance) && (rel_error > rel_tolerance);
        if (is_wrong) {
            result.num_errors++;
            if (result.first_error_index < 0) {
                result.first_error_index = element_idx;
                if (print_first_error) {
                    printf("  First mismatch at [%d]: GPU=%.6f  REF=%.6f  abs_err=%.2e  rel_err=%.2e\n",
                           element_idx, gpu_val, ref_val, abs_error, rel_error);
                }
            }
        }
    }
    return result;
}

inline void print_check_result(const char *label, const CheckResult &result) {
    if (result.num_errors == 0) {
        printf("  %-30s PASS  (max_abs=%.2e  max_rel=%.2e)\n",
               label, result.max_abs_error, result.max_rel_error);
    } else {
        printf("  %-30s FAIL  %d/%d wrong  (max_abs=%.2e  max_rel=%.2e)\n",
               label, result.num_errors, result.total_elements,
               result.max_abs_error, result.max_rel_error);
    }
}

// -----------------------------------------------------------------------
// CPU SGEMM reference — used to verify GPU kernels
// Computes C = alpha * A * B + beta * C
// A: M×K  B: K×N  C: M×N  (all row-major)
// -----------------------------------------------------------------------
inline void cpu_sgemm(
    int M, int N, int K,
    float alpha,
    const float *A, int lda,   // A[M][K], lda = K
    const float *B, int ldb,   // B[K][N], ldb = N
    float beta,
    float *C, int ldc          // C[M][N], ldc = N
) {
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            float accumulator = 0.0f;
            for (int k = 0; k < K; k++) {
                accumulator += A[row * lda + k] * B[k * ldb + col];
            }
            C[row * ldc + col] = alpha * accumulator + beta * C[row * ldc + col];
        }
    }
}

// -----------------------------------------------------------------------
// Fill matrix with random values in [-1, 1]
// -----------------------------------------------------------------------
inline void fill_random(float *matrix, int num_elements, unsigned int seed = 42) {
    // Simple LCG — reproducible, no stdlib dependency
    unsigned int state = seed;
    for (int element_idx = 0; element_idx < num_elements; element_idx++) {
        state = state * 1664525u + 1013904223u;
        float normalized = (float)(state >> 8) / (float)(1 << 24);  // [0, 1)
        matrix[element_idx] = normalized * 2.0f - 1.0f;             // [-1, 1)
    }
}

inline void fill_zeros(float *matrix, int num_elements) {
    for (int element_idx = 0; element_idx < num_elements; element_idx++) {
        matrix[element_idx] = 0.0f;
    }
}
