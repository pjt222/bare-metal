#include <cuda_runtime.h>
#include <cusparseLt.h>

#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

#include "../_common/reference_bench.h"

#define CHECK_CUSPARSELT(call)                                                   \
    do {                                                                         \
        cusparseStatus_t _st = (call);                                           \
        if (_st != CUSPARSE_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuSPARSELt error at %s:%d -- %s\n",                 \
                    __FILE__, __LINE__, cusparseLtGetErrorString(_st));          \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

int main(int argc, char** argv) {
    const int M = (argc > 1) ? std::atoi(argv[1]) : 4096;
    const int N = (argc > 2) ? std::atoi(argv[2]) : 4096;
    const int K = (argc > 3) ? std::atoi(argv[3]) : 4096;
    constexpr int warmup_runs = 5;
    constexpr int measured_runs = 11;

    if ((M % 16) != 0 || (N % 8) != 0 || (K % 32) != 0) {
        fprintf(stderr, "M must be %%16, N must be %%8, K must be %%32 for cuSPARSELt INT8\n");
        return EXIT_FAILURE;
    }

    printf("=== cuSPARSELt Sparse IGEMM Reference ===\n");
    printf("Matrix: C[%d x %d] = A[%d x %d] * B[%d x %d]  (2:4 sparse INT8, dense-equiv TOPS)\n\n",
           M, N, M, K, K, N);
    print_device_name();

    const size_t a_elems = static_cast<size_t>(M) * K;
    const size_t b_elems = static_cast<size_t>(N) * K;  // B stored row-major as [N, K], opB = T.
    const size_t c_elems = static_cast<size_t>(M) * N;

    std::vector<int8_t> hA(a_elems);
    std::vector<int8_t> hB(b_elems);
    std::vector<int32_t> hC(c_elems, 0);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-2, 2);
    fill_host_buffer(hA.data(), hA.size(), [&]() { return static_cast<int8_t>(dist(rng)); });
    fill_host_buffer(hB.data(), hB.size(), [&]() { return static_cast<int8_t>(dist(rng)); });

    int8_t* dA = nullptr;
    int8_t* dB = nullptr;
    int32_t* dC = nullptr;
    int* d_valid = nullptr;
    CHECK_CUDA_RT(cudaMalloc(&dA, a_elems * sizeof(int8_t)));
    CHECK_CUDA_RT(cudaMalloc(&dB, b_elems * sizeof(int8_t)));
    CHECK_CUDA_RT(cudaMalloc(&dC, c_elems * sizeof(int32_t)));
    CHECK_CUDA_RT(cudaMalloc(&d_valid, sizeof(int)));
    CHECK_CUDA_RT(cudaMemcpy(dA, hA.data(), a_elems * sizeof(int8_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_RT(cudaMemcpy(dB, hB.data(), b_elems * sizeof(int8_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_RT(cudaMemset(dC, 0, c_elems * sizeof(int32_t)));

    cusparseLtHandle_t handle;
    cusparseLtMatDescriptor_t matA, matB, matC;
    cusparseLtMatmulDescriptor_t matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t plan;
    CHECK_CUSPARSELT(cusparseLtInit(&handle));

    constexpr cudaDataType type_ab = CUDA_R_8I;
    constexpr cudaDataType type_c = CUDA_R_32I;
    constexpr cusparseComputeType compute_type = CUSPARSE_COMPUTE_32I;
    constexpr cusparseOrder_t order = CUSPARSE_ORDER_ROW;
    constexpr cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    constexpr cusparseOperation_t opB = CUSPARSE_OPERATION_TRANSPOSE;
    constexpr unsigned alignment = 16;

    CHECK_CUSPARSELT(cusparseLtStructuredDescriptorInit(
        &handle, &matA, M, K, K, alignment, type_ab, order, CUSPARSELT_SPARSITY_50_PERCENT));
    CHECK_CUSPARSELT(cusparseLtDenseDescriptorInit(
        &handle, &matB, N, K, K, alignment, type_ab, order));
    CHECK_CUSPARSELT(cusparseLtDenseDescriptorInit(
        &handle, &matC, M, N, N, alignment, type_c, order));

    CHECK_CUSPARSELT(cusparseLtMatmulDescriptorInit(
        &handle, &matmul, opA, opB, &matA, &matB, &matC, &matC, compute_type));
    CHECK_CUSPARSELT(cusparseLtMatmulAlgSelectionInit(
        &handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT));
    CHECK_CUSPARSELT(cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel));
    CHECK_CUSPARSELT(cusparseLtMatmulDescSetAttribute(
        &handle, &matmul, CUSPARSELT_MATMUL_SPARSE_MAT_POINTER, &dA, sizeof(dA)));

    CHECK_CUSPARSELT(cusparseLtSpMMAPrune(
        &handle, &matmul, dA, dA, CUSPARSELT_PRUNE_SPMMA_TILE, nullptr));
    CHECK_CUSPARSELT(cusparseLtSpMMAPruneCheck(&handle, &matmul, dA, d_valid, nullptr));
    int is_valid = 0;
    CHECK_CUDA_RT(cudaMemcpy(&is_valid, d_valid, sizeof(int), cudaMemcpyDeviceToHost));
    if (is_valid != 0) {
        fprintf(stderr, "cuSPARSELt prune check failed\n");
        return EXIT_FAILURE;
    }

    size_t compressed_size = 0;
    size_t compressed_buffer_size = 0;
    CHECK_CUSPARSELT(cusparseLtSpMMACompressedSize(
        &handle, &plan, &compressed_size, &compressed_buffer_size));
    int8_t* dA_compressed = nullptr;
    void* dA_compressed_buffer = nullptr;
    CHECK_CUDA_RT(cudaMalloc(&dA_compressed, compressed_size));
    CHECK_CUDA_RT(cudaMalloc(&dA_compressed_buffer, compressed_buffer_size));
    CHECK_CUSPARSELT(cusparseLtSpMMACompress(
        &handle, &plan, dA, dA_compressed, dA_compressed_buffer, nullptr));

    const int32_t alpha = 1;
    const int32_t beta = 0;
    CHECK_CUSPARSELT(cusparseLtMatmulSearch(
        &handle, &plan, &alpha, dA_compressed, dB, &beta, dC, dC, nullptr, nullptr, 0));
    CHECK_CUDA_RT(cudaMemset(dC, 0, c_elems * sizeof(int32_t)));

    size_t workspace_size = 0;
    CHECK_CUSPARSELT(cusparseLtMatmulGetWorkspace(&handle, &plan, &workspace_size));
    void* d_workspace = nullptr;
    if (workspace_size > 0) CHECK_CUDA_RT(cudaMalloc(&d_workspace, workspace_size));

    auto launch = [&]() {
        CHECK_CUSPARSELT(cusparseLtMatmul(
            &handle, &plan, &alpha, dA_compressed, dB, &beta, dC, dC,
            d_workspace, nullptr, 0));
    };

    for (int i = 0; i < warmup_runs; ++i) launch();
    CHECK_CUDA_RT(cudaDeviceSynchronize());

    std::vector<float> samples;
    samples.reserve(measured_runs);
    for (int i = 0; i < measured_runs; ++i) {
        CHECK_CUDA_RT(cudaMemset(dC, 0, c_elems * sizeof(int32_t)));
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
    const double tops = compute_gemm_tops(M, N, K, ms);
    printf("cusparselt_igemm_sparse %7.3f ms  %10.2f TOPS\n", ms, tops);

    CHECK_CUDA_RT(cudaFree(d_workspace));
    CHECK_CUDA_RT(cudaFree(dA_compressed_buffer));
    CHECK_CUDA_RT(cudaFree(dA_compressed));
    CHECK_CUSPARSELT(cusparseLtMatmulPlanDestroy(&plan));
    CHECK_CUSPARSELT(cusparseLtMatmulAlgSelectionDestroy(&alg_sel));
    CHECK_CUSPARSELT(cusparseLtMatDescriptorDestroy(&matA));
    CHECK_CUSPARSELT(cusparseLtMatDescriptorDestroy(&matB));
    CHECK_CUSPARSELT(cusparseLtMatDescriptorDestroy(&matC));
    CHECK_CUSPARSELT(cusparseLtDestroy(&handle));
    CHECK_CUDA_RT(cudaFree(dA));
    CHECK_CUDA_RT(cudaFree(dB));
    CHECK_CUDA_RT(cudaFree(dC));
    CHECK_CUDA_RT(cudaFree(d_valid));
    return 0;
}
