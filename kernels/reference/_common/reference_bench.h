#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <random>
#include <vector>

#define CHECK_CUDA_RT(call)                                                     \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA runtime error at %s:%d -- %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(_err));              \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

inline float median_ms(std::vector<float> samples) {
    if (samples.empty()) return 0.0f;
    std::sort(samples.begin(), samples.end());
    const size_t mid = samples.size() / 2;
    if ((samples.size() % 2U) == 1U) return samples[mid];
    return 0.5f * (samples[mid - 1U] + samples[mid]);
}

inline double compute_gemm_tflops(int M, int N, int K, float elapsed_ms) {
    const double flops = 2.0 * static_cast<double>(M) *
                         static_cast<double>(N) *
                         static_cast<double>(K);
    return (flops / (static_cast<double>(elapsed_ms) / 1000.0)) / 1e12;
}

inline double compute_gemm_gflops(int M, int N, int K, float elapsed_ms) {
    return compute_gemm_tflops(M, N, K, elapsed_ms) * 1000.0;
}

inline double compute_gemm_tops(int M, int N, int K, float elapsed_ms) {
    return compute_gemm_tflops(M, N, K, elapsed_ms);
}

inline void print_device_name() {
    int dev = 0;
    cudaDeviceProp prop{};
    CHECK_CUDA_RT(cudaGetDevice(&dev));
    CHECK_CUDA_RT(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s\n\n", prop.name);
}

template <typename T, typename Fn>
inline void fill_host_buffer(T* ptr, size_t count, Fn fn) {
    for (size_t i = 0; i < count; ++i) ptr[i] = fn();
}
