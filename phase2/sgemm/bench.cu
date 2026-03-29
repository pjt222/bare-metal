/*
 * bench.cu — SGEMM benchmark harness
 *
 * Loads naive and tiled SGEMM cubins via the CUDA Driver API,
 * verifies correctness against a CPU reference, then benchmarks.
 *
 * Build:
 *   nvcc -arch=sm_86 -o bench bench.cu -lcuda -I../common
 *
 * Usage:
 *   ./bench [M] [N] [K]
 *   ./bench 1024 1024 1024    # default
 *
 * Expected output:
 *   Correctness:
 *     naive SGEMM              PASS  (max_abs=...  max_rel=...)
 *     tiled SGEMM              PASS  (max_abs=...  max_rel=...)
 *   Performance (avg of 20 runs):
 *     naive SGEMM               nn.nnn ms    nn.nn GFLOPS
 *     tiled SGEMM               nn.nnn ms   nnn.nn GFLOPS
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>

#include "../common/bench.h"
#include "../common/check.h"

// -----------------------------------------------------------------------
// Load a kernel function from a cubin file
// -----------------------------------------------------------------------
static void load_kernel(
    const char *cubin_path,
    const char *func_name,
    CUmodule   *out_module,
    CUfunction *out_func
) {
    CUresult load_result = cuModuleLoad(out_module, cubin_path);
    if (load_result != CUDA_SUCCESS) {
        const char *error_str = nullptr;
        cuGetErrorString(load_result, &error_str);
        fprintf(stderr, "Cannot load cubin '%s': %s\n",
                cubin_path, error_str ? error_str : "unknown");
        exit(EXIT_FAILURE);
    }
    CHECK_CU(cuModuleGetFunction(out_func, *out_module, func_name));
}

// -----------------------------------------------------------------------
// Run one SGEMM kernel and return timing (ms)
// -----------------------------------------------------------------------
static float run_sgemm(
    CUfunction kernel_func,
    CUdeviceptr device_a,    // M×K
    CUdeviceptr device_b,    // K×N
    CUdeviceptr device_c,    // M×N (output, zeroed before call)
    int M, int N, int K,
    int block_x, int block_y,
    int warmup_iters,
    int bench_iters,
    CUdeviceptr device_c_zeros,  // pre-zeroed C for reset between runs
    size_t c_bytes
) {
    void *args[] = { &device_a, &device_b, &device_c, &M, &N, &K };

    int grid_x = (N + block_x - 1) / block_x;
    int grid_y = (M + block_y - 1) / block_y;

    // Warmup
    for (int warmup_i = 0; warmup_i < warmup_iters; warmup_i++) {
        CHECK_CU(cuMemcpyDtoD(device_c, device_c_zeros, c_bytes));
        CHECK_CU(cuLaunchKernel(kernel_func,
            grid_x, grid_y, 1,
            block_x, block_y, 1,
            0, NULL, args, NULL));
    }
    CHECK_CU(cuCtxSynchronize());

    // Benchmark
    BenchTimer timer;
    timer.start();
    for (int bench_i = 0; bench_i < bench_iters; bench_i++) {
        CHECK_CU(cuMemcpyDtoD(device_c, device_c_zeros, c_bytes));
        CHECK_CU(cuLaunchKernel(kernel_func,
            grid_x, grid_y, 1,
            block_x, block_y, 1,
            0, NULL, args, NULL));
    }
    return timer.stop_ms() / (float)bench_iters;
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------
int main(int argc, char **argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int N = (argc > 2) ? atoi(argv[2]) : 1024;
    int K = (argc > 3) ? atoi(argv[3]) : 1024;

    printf("=== SGEMM Benchmark — RTX 3070 Ti (sm_86) ===\n");
    printf("Matrix: C[%d×%d] = A[%d×%d] * B[%d×%d]\n\n", M, N, M, K, K, N);

    // --- Init CUDA Driver ---
    CHECK_CU(cuInit(0));
    CUdevice cu_device;
    CHECK_CU(cuDeviceGet(&cu_device, 0));

    char device_name[256];
    CHECK_CU(cuDeviceGetName(device_name, sizeof(device_name), cu_device));
    printf("Device: %s\n\n", device_name);

    CUcontext cu_context;
    CHECK_CU(cuCtxCreate(&cu_context, 0, cu_device));

    // --- Load kernels ---
    CUmodule  naive_module, tiled_module;
    CUfunction naive_func,  tiled_func;

    CUmodule  regblk_module;
    CUfunction regblk_func;

    load_kernel("naive.sm_86.cubin",           "sgemm_naive",             &naive_module,  &naive_func);
    load_kernel("tiled.sm_86.cubin",           "sgemm_tiled",             &tiled_module,  &tiled_func);
    load_kernel("register_blocked.sm_86.cubin","sgemm_register_blocked",  &regblk_module, &regblk_func);
    printf("Kernels loaded.\n\n");

    // --- Allocate host memory ---
    size_t a_bytes = (size_t)M * K * sizeof(float);
    size_t b_bytes = (size_t)K * N * sizeof(float);
    size_t c_bytes = (size_t)M * N * sizeof(float);

    float *host_a   = (float *)malloc(a_bytes);
    float *host_b   = (float *)malloc(b_bytes);
    float *host_c   = (float *)malloc(c_bytes);
    float *host_ref = (float *)malloc(c_bytes);

    fill_random(host_a, M * K, 42);
    fill_random(host_b, K * N, 99);
    fill_zeros(host_ref, M * N);

    // --- CPU reference ---
    printf("Computing CPU reference (may take a moment for large matrices)...\n");
    cpu_sgemm(M, N, K, 1.0f, host_a, K, host_b, N, 0.0f, host_ref, N);

    // --- Allocate device memory ---
    CUdeviceptr device_a, device_b, device_c, device_c_zeros;
    CHECK_CU(cuMemAlloc(&device_a,      a_bytes));
    CHECK_CU(cuMemAlloc(&device_b,      b_bytes));
    CHECK_CU(cuMemAlloc(&device_c,      c_bytes));
    CHECK_CU(cuMemAlloc(&device_c_zeros, c_bytes));

    CHECK_CU(cuMemcpyHtoD(device_a, host_a, a_bytes));
    CHECK_CU(cuMemcpyHtoD(device_b, host_b, b_bytes));
    CHECK_CU(cuMemsetD32(device_c_zeros, 0, M * N));

    // --- Correctness checks ---
    printf("Correctness (tolerance 1e-3):\n");

    // Naive
    {
        CHECK_CU(cuMemcpyDtoD(device_c, device_c_zeros, c_bytes));
        void *args[] = { &device_a, &device_b, &device_c, &M, &N, &K };
        CHECK_CU(cuLaunchKernel(naive_func,
            (N + 15) / 16, (M + 15) / 16, 1,
            16, 16, 1,
            0, NULL, args, NULL));
        CHECK_CU(cuCtxSynchronize());
        CHECK_CU(cuMemcpyDtoH(host_c, device_c, c_bytes));
        auto result = check_fp32(host_c, host_ref, M * N);
        print_check_result("naive SGEMM", result);
    }

    // Tiled
    {
        CHECK_CU(cuMemcpyDtoD(device_c, device_c_zeros, c_bytes));
        void *args[] = { &device_a, &device_b, &device_c, &M, &N, &K };
        CHECK_CU(cuLaunchKernel(tiled_func,
            (N + 31) / 32, (M + 31) / 32, 1,
            32, 32, 1,
            0, NULL, args, NULL));
        CHECK_CU(cuCtxSynchronize());
        CHECK_CU(cuMemcpyDtoH(host_c, device_c, c_bytes));
        auto result = check_fp32(host_c, host_ref, M * N);
        print_check_result("tiled SGEMM", result);
    }

    // Register-blocked (64×64 block tile, 4×4 thread tile = 16×16 threads, BK=32)
    {
        CHECK_CU(cuMemcpyDtoD(device_c, device_c_zeros, c_bytes));
        void *args[] = { &device_a, &device_b, &device_c, &M, &N, &K };
        CHECK_CU(cuLaunchKernel(regblk_func,
            (N + 63) / 64, (M + 63) / 64, 1,
            16, 16, 1,   // 16×16 = 256 threads: BM/TM × BN/TN = 64/4 × 64/4
            0, NULL, args, NULL));
        CHECK_CU(cuCtxSynchronize());
        CHECK_CU(cuMemcpyDtoH(host_c, device_c, c_bytes));
        auto result = check_fp32(host_c, host_ref, M * N);
        print_check_result("register-blocked SGEMM", result);
    }

    printf("\n");

    // --- Performance benchmarks ---
    int warmup_iters = 3;
    int bench_iters  = 20;
    printf("Performance (avg of %d runs, %d warmup):\n", bench_iters, warmup_iters);

    double peak_gflops = 21700.0;  // RTX 3070 Ti theoretical FP32 peak (TFLOPS * 1000)

    // Naive
    {
        float avg_ms = run_sgemm(naive_func,
            device_a, device_b, device_c, M, N, K,
            16, 16, warmup_iters, bench_iters, device_c_zeros, c_bytes);
        double gflops = compute_gflops_gemm(M, N, K, avg_ms);
        print_gemm_result("naive SGEMM", M, N, K, avg_ms, gflops, peak_gflops);
    }

    // Tiled
    {
        float avg_ms = run_sgemm(tiled_func,
            device_a, device_b, device_c, M, N, K,
            32, 32, warmup_iters, bench_iters, device_c_zeros, c_bytes);
        double gflops = compute_gflops_gemm(M, N, K, avg_ms);
        print_gemm_result("tiled SGEMM", M, N, K, avg_ms, gflops, peak_gflops);
    }

    // Register-blocked (64×64 tile, 16×16 threads, BK=32)
    {
        float avg_ms = run_sgemm(regblk_func,
            device_a, device_b, device_c, M, N, K,
            16, 16, warmup_iters, bench_iters, device_c_zeros, c_bytes);
        double gflops = compute_gflops_gemm(M, N, K, avg_ms);
        print_gemm_result("register-blocked SGEMM", M, N, K, avg_ms, gflops, peak_gflops);
    }

    printf("\n  RTX 3070 Ti FP32 theoretical peak: %.0f GFLOPS\n", peak_gflops);

    // --- Cleanup ---
    cuMemFree(device_a);
    cuMemFree(device_b);
    cuMemFree(device_c);
    cuMemFree(device_c_zeros);
    cuModuleUnload(naive_module);
    cuModuleUnload(tiled_module);
    cuModuleUnload(regblk_module);
    cuCtxDestroy(cu_context);
    free(host_a);
    free(host_b);
    free(host_c);
    free(host_ref);

    return 0;
}
