#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <random>
#include <vector>

#include "../_common/reference_bench.h"

#define CHECK_CUBLAS(call)                                                      \
    do {                                                                        \
        cublasStatus_t _st = (call);                                            \
        if (_st != CUBLAS_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "cuBLAS error at %s:%d -- status %d\n",             \
                    __FILE__, __LINE__, static_cast<int>(_st));                 \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

int main(int argc, char** argv) {
    const int M = (argc > 1) ? std::atoi(argv[1]) : 4096;
    const int N = (argc > 2) ? std::atoi(argv[2]) : 4096;
    const int K = (argc > 3) ? std::atoi(argv[3]) : 4096;
    constexpr int warmup_runs = 5;
    constexpr int measured_runs = 11;

    printf("=== cuBLAS HGEMM Reference ===\n");
    printf("Matrix: C[%d x %d] = A[%d x %d] * B[%d x %d]  (FP16 in, FP32 out)\n\n",
           M, N, M, K, K, N);
    print_device_name();

    cublasHandle_t handle = nullptr;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    const size_t a_elems = static_cast<size_t>(M) * K;
    const size_t b_elems = static_cast<size_t>(K) * N;
    const size_t c_elems = static_cast<size_t>(M) * N;

    std::vector<__half> hA(a_elems);
    std::vector<__half> hB(b_elems);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    fill_host_buffer(hA.data(), hA.size(), [&]() { return __float2half(dist(rng)); });
    fill_host_buffer(hB.data(), hB.size(), [&]() { return __float2half(dist(rng)); });

    __half* dA = nullptr;
    __half* dB = nullptr;
    float* dC = nullptr;
    CHECK_CUDA_RT(cudaMalloc(&dA, a_elems * sizeof(__half)));
    CHECK_CUDA_RT(cudaMalloc(&dB, b_elems * sizeof(__half)));
    CHECK_CUDA_RT(cudaMalloc(&dC, c_elems * sizeof(float)));
    CHECK_CUDA_RT(cudaMemcpy(dA, hA.data(), a_elems * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA_RT(cudaMemcpy(dB, hB.data(), b_elems * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA_RT(cudaMemset(dC, 0, c_elems * sizeof(float)));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    auto launch = [&]() {
        CHECK_CUBLAS(cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            &alpha,
            dA, CUDA_R_16F, M,
            dB, CUDA_R_16F, K,
            &beta,
            dC, CUDA_R_32F, M,
            CUBLAS_COMPUTE_32F_FAST_16F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };

    for (int i = 0; i < warmup_runs; ++i) launch();
    CHECK_CUDA_RT(cudaDeviceSynchronize());

    std::vector<float> samples;
    samples.reserve(measured_runs);
    for (int i = 0; i < measured_runs; ++i) {
        cudaEvent_t start, stop;
        CHECK_CUDA_RT(cudaEventCreate(&start));
        CHECK_CUDA_RT(cudaEventCreate(&stop));
        CHECK_CUDA_RT(cudaEventRecord(start));
        launch();
        CHECK_CUDA_RT(cudaEventRecord(stop));
        CHECK_CUDA_RT(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_CUDA_RT(cudaEventElapsedTime(&ms, start, stop));
        samples.push_back(ms);
        CHECK_CUDA_RT(cudaEventDestroy(start));
        CHECK_CUDA_RT(cudaEventDestroy(stop));
    }

    const float ms = median_ms(samples);
    const double gflops = compute_gemm_gflops(M, N, K, ms);
    printf("cublas_hgemm %7.3f ms  %10.0f GFLOPS\n", ms, gflops);

    CHECK_CUDA_RT(cudaFree(dA));
    CHECK_CUDA_RT(cudaFree(dB));
    CHECK_CUDA_RT(cudaFree(dC));
    CHECK_CUBLAS(cublasDestroy(handle));
    return 0;
}
