#include <cudnn.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <limits>
#include <cstdio>
#include <random>
#include <vector>

#include "../_common/reference_bench.h"

#define CHECK_CUDNN(call)                                                        \
    do {                                                                         \
        cudnnStatus_t _st = (call);                                              \
        if (_st != CUDNN_STATUS_SUCCESS) {                                       \
            fprintf(stderr, "cuDNN error at %s:%d -- %s\n",                      \
                    __FILE__, __LINE__, cudnnGetErrorString(_st));               \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

int main(int argc, char** argv) {
    const int N = (argc > 1) ? std::atoi(argv[1]) : 1;
    const int H = (argc > 2) ? std::atoi(argv[2]) : 64;
    const int W = (argc > 3) ? std::atoi(argv[3]) : 64;
    const int Cin = (argc > 4) ? std::atoi(argv[4]) : 320;
    const int Cout = (argc > 5) ? std::atoi(argv[5]) : 320;
    constexpr int warmup_runs = 15;
    constexpr int measured_runs = 11;
    constexpr int kH = 3;
    constexpr int kW = 3;
    constexpr int pad = 1;
    constexpr int stride = 1;
    constexpr int dilation = 1;

    printf("=== cuDNN Conv2d Reference ===\n");
    printf("Conv: N=%d H=%d W=%d Cin=%d Cout=%d  (3x3, pad=1, stride=1, FP16 tensor-op path)\n\n",
           N, H, W, Cin, Cout);
    print_device_name();

    cudnnHandle_t handle = nullptr;
    CHECK_CUDNN(cudnnCreate(&handle));

    cudnnTensorDescriptor_t x_desc = nullptr;
    cudnnTensorDescriptor_t y_desc = nullptr;
    cudnnFilterDescriptor_t w_desc = nullptr;
    cudnnConvolutionDescriptor_t conv_desc = nullptr;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&x_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&y_desc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&w_desc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, N, Cin, H, W));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_HALF, CUDNN_TENSOR_NHWC, Cout, Cin, kH, kW));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
        conv_desc, pad, pad, stride, stride, dilation, dilation, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    CHECK_CUDNN(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH));

    int out_n = 0, out_c = 0, out_h = 0, out_w = 0;
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(
        conv_desc, x_desc, w_desc, &out_n, &out_c, &out_h, &out_w));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, out_n, out_c, out_h, out_w));

    const size_t x_elems = static_cast<size_t>(N) * H * W * Cin;
    const size_t w_elems = static_cast<size_t>(Cout) * kH * kW * Cin;
    const size_t y_elems = static_cast<size_t>(out_n) * out_h * out_w * out_c;
    std::vector<__half> hX(x_elems);
    std::vector<__half> hW(w_elems);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    fill_host_buffer(hX.data(), hX.size(), [&]() { return __float2half(dist(rng)); });
    fill_host_buffer(hW.data(), hW.size(), [&]() { return __float2half(0.1f * dist(rng)); });

    __half* dX = nullptr;
    __half* dW = nullptr;
    __half* dY = nullptr;
    CHECK_CUDA_RT(cudaMalloc(&dX, x_elems * sizeof(__half)));
    CHECK_CUDA_RT(cudaMalloc(&dW, w_elems * sizeof(__half)));
    CHECK_CUDA_RT(cudaMalloc(&dY, y_elems * sizeof(__half)));
    CHECK_CUDA_RT(cudaMemcpy(dX, hX.data(), x_elems * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA_RT(cudaMemcpy(dW, hW.data(), w_elems * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA_RT(cudaMemset(dY, 0, y_elems * sizeof(__half)));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    int returned_algo_count = 0;
    cudnnConvolutionFwdAlgoPerf_t algo_perf[8];
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
        handle, x_desc, w_desc, conv_desc, y_desc, 8, &returned_algo_count, algo_perf));
    if (returned_algo_count <= 0) {
        fprintf(stderr, "cuDNN returned no supported conv2d forward algorithms\n");
        return EXIT_FAILURE;
    }

    struct Candidate {
        cudnnConvolutionFwdAlgo_t algo;
        size_t workspace_size;
        float probe_ms;
    };
    std::vector<Candidate> candidates;
    size_t max_workspace_size = 0;
    for (int i = 0; i < returned_algo_count; ++i) {
        if (algo_perf[i].status != CUDNN_STATUS_SUCCESS) continue;
        size_t ws = 0;
        CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
            handle, x_desc, w_desc, conv_desc, y_desc, algo_perf[i].algo, &ws));
        candidates.push_back(Candidate{algo_perf[i].algo, ws, std::numeric_limits<float>::infinity()});
        if (ws > max_workspace_size) max_workspace_size = ws;
    }
    if (candidates.empty()) {
        fprintf(stderr, "cuDNN returned no successful conv2d forward candidates\n");
        return EXIT_FAILURE;
    }

    void* workspace = nullptr;
    if (max_workspace_size > 0) CHECK_CUDA_RT(cudaMalloc(&workspace, max_workspace_size));

    auto launch = [&](cudnnConvolutionFwdAlgo_t algo, size_t workspace_size) {
        CHECK_CUDNN(cudnnConvolutionForward(
            handle, &alpha, x_desc, dX, w_desc, dW, conv_desc, algo,
            workspace, workspace_size, &beta, y_desc, dY));
    };

    for (auto& candidate : candidates) {
        std::vector<float> probe_samples;
        probe_samples.reserve(3);
        for (int i = 0; i < 2; ++i) launch(candidate.algo, candidate.workspace_size);
        CHECK_CUDA_RT(cudaDeviceSynchronize());
        for (int i = 0; i < 3; ++i) {
            cudaEvent_t start, stop;
            CHECK_CUDA_RT(cudaEventCreate(&start));
            CHECK_CUDA_RT(cudaEventCreate(&stop));
            CHECK_CUDA_RT(cudaEventRecord(start));
            launch(candidate.algo, candidate.workspace_size);
            CHECK_CUDA_RT(cudaEventRecord(stop));
            CHECK_CUDA_RT(cudaEventSynchronize(stop));
            float ms = 0.0f;
            CHECK_CUDA_RT(cudaEventElapsedTime(&ms, start, stop));
            probe_samples.push_back(ms);
            CHECK_CUDA_RT(cudaEventDestroy(start));
            CHECK_CUDA_RT(cudaEventDestroy(stop));
        }
        candidate.probe_ms = median_ms(probe_samples);
    }

    const Candidate* best = &candidates.front();
    for (const auto& candidate : candidates) {
        if (candidate.probe_ms < best->probe_ms) best = &candidate;
    }

    for (int i = 0; i < warmup_runs; ++i) launch(best->algo, best->workspace_size);
    CHECK_CUDA_RT(cudaDeviceSynchronize());

    std::vector<float> samples;
    samples.reserve(measured_runs);
    for (int i = 0; i < measured_runs; ++i) {
        cudaEvent_t start, stop;
        CHECK_CUDA_RT(cudaEventCreate(&start));
        CHECK_CUDA_RT(cudaEventCreate(&stop));
        CHECK_CUDA_RT(cudaEventRecord(start));
        launch(best->algo, best->workspace_size);
        CHECK_CUDA_RT(cudaEventRecord(stop));
        CHECK_CUDA_RT(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_CUDA_RT(cudaEventElapsedTime(&ms, start, stop));
        samples.push_back(ms);
        CHECK_CUDA_RT(cudaEventDestroy(start));
        CHECK_CUDA_RT(cudaEventDestroy(stop));
    }

    const float ms = median_ms(samples);
    const int M_dim = out_n * out_h * out_w;
    const int K_dim = Cin * kH * kW;
    const double gflops = compute_gemm_gflops(M_dim, Cout, K_dim, ms);
    printf("cudnn_conv2d %7.3f ms  %10.0f GFLOPS\n", ms, gflops);

    CHECK_CUDA_RT(cudaFree(workspace));
    CHECK_CUDA_RT(cudaFree(dX));
    CHECK_CUDA_RT(cudaFree(dW));
    CHECK_CUDA_RT(cudaFree(dY));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(w_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(x_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(y_desc));
    CHECK_CUDNN(cudnnDestroy(handle));
    return 0;
}
