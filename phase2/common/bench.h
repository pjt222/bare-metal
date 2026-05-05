#pragma once
/*
 * bench.h — Timing and GFLOPS measurement utilities
 *
 * Usage:
 *   BenchTimer timer;
 *   timer.start();
 *   // ... kernel launch ...
 *   cudaDeviceSynchronize();
 *   double elapsed_ms = timer.stop_ms();
 *   double gflops = compute_gflops_gemm(M, N, K, elapsed_ms);
 */

#include <cuda.h>
#include <cstdio>
#include <ctime>

// -----------------------------------------------------------------------
// CUDA Driver API error check
// -----------------------------------------------------------------------
#define CHECK_CU(call)                                                        \
    do {                                                                      \
        CUresult _result = (call);                                            \
        if (_result != CUDA_SUCCESS) {                                        \
            const char *_err = nullptr;                                       \
            cuGetErrorString(_result, &_err);                                 \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                    \
                    __FILE__, __LINE__, _err ? _err : "unknown");             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// -----------------------------------------------------------------------
// High-resolution wall-clock timer using CUDA events
// -----------------------------------------------------------------------
struct BenchTimer {
    CUevent event_start;
    CUevent event_stop;

    BenchTimer() {
        CHECK_CU(cuEventCreate(&event_start, CU_EVENT_DEFAULT));
        CHECK_CU(cuEventCreate(&event_stop,  CU_EVENT_DEFAULT));
    }

    ~BenchTimer() {
        cuEventDestroy(event_start);
        cuEventDestroy(event_stop);
    }

    void start() {
        CHECK_CU(cuEventRecord(event_start, NULL));
    }

    // Returns elapsed milliseconds
    float stop_ms() {
        float elapsed_ms = 0.0f;
        CHECK_CU(cuEventRecord(event_stop, NULL));
        CHECK_CU(cuEventSynchronize(event_stop));
        CHECK_CU(cuEventElapsedTime(&elapsed_ms, event_start, event_stop));
        return elapsed_ms;
    }
};

// -----------------------------------------------------------------------
// GFLOPS calculations
// -----------------------------------------------------------------------

// GEMM: C = A * B where A is M×K, B is K×N, C is M×N
// Each output element requires K multiplications and K additions = 2K FLOPs
// Total: 2 * M * N * K FLOPs
inline double compute_gflops_gemm(int M, int N, int K, float elapsed_ms) {
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double seconds = elapsed_ms / 1000.0;
    return (flops / seconds) / 1e9;
}

// -----------------------------------------------------------------------
// Warmup — run a kernel a few times before benchmarking
// -----------------------------------------------------------------------
// Usage: WARMUP(3, your_kernel_call());
#define WARMUP(iterations, kernel_call)                 \
    for (int _warmup_i = 0; _warmup_i < (iterations); _warmup_i++) { \
        (kernel_call);                                  \
    }                                                   \
    CHECK_CU(cuCtxSynchronize());

// -----------------------------------------------------------------------
// Benchmark loop — runs kernel N times and returns average ms
// -----------------------------------------------------------------------
// Usage:
//   float avg_ms = BENCH(10, cuLaunchKernel(...));
//   double gflops = compute_gflops_gemm(M, N, K, avg_ms);
#define BENCH(iterations, kernel_call)                                  \
    [&]() -> float {                                                    \
        BenchTimer _timer;                                              \
        _timer.start();                                                 \
        for (int _bench_i = 0; _bench_i < (iterations); _bench_i++) { \
            (kernel_call);                                              \
        }                                                               \
        float _elapsed = _timer.stop_ms();                              \
        return _elapsed / (float)(iterations);                          \
    }()

// -----------------------------------------------------------------------
// Print benchmark results
// -----------------------------------------------------------------------
inline void print_gemm_result(
    const char *label, int M, int N, int K,
    float elapsed_ms, double gflops,
    double reference_gflops = 0.0
) {
    printf("  %-30s %7.3f ms   %8.2f GFLOPS", label, elapsed_ms, gflops);
    if (reference_gflops > 0.0) {
        printf("  (%5.1f%% of reference)", 100.0 * gflops / reference_gflops);
    }
    printf("\n");
}
