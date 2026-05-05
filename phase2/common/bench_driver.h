#pragma once
/*
 * bench_driver.h — Shared benchmark driver for bare-metal GPU kernels
 *
 * Eliminates boilerplate across bench.cu files:
 *   - CUDA context init/destroy (RAII)
 *   - Typed device memory allocation (RAII)
 *   - Host allocation + random fill
 *   - Warmup + timed benchmark loop
 *   - CPU reference comparison
 *   - Standard result printing
 *
 * Usage:
 *   int main(int argc, char** argv) {
 *       BenchDriver driver(argc, argv);
 *       driver.init_context();
 *
 *       // Allocate buffers
 *       auto d_A = driver.device_alloc<float>(M*K);
 *       auto h_A = driver.host_alloc<float>(M*K);
 *       fill_random(h_A.get(), M*K);
 *       driver.copy_h2d(d_A, h_A, M*K * sizeof(float));
 *
 *       // Load kernel
 *       CUfunction kernel = driver.load_kernel("kernel.sm_86.cubin", "kernel_name");
 *
 *       // CPU reference
 *       auto h_ref = driver.host_alloc<float>(M*N);
 *       cpu_reference(h_A.get(), h_B.get(), h_ref.get(), M, N, K);
 *
 *       // Benchmark
 *       float ms = driver.benchmark_kernel(kernel, grid, block, smem, args);
 *       double gflops = compute_gflops_gemm(M, N, K, ms);
 *
 *       // Check
 *       driver.copy_d2h(h_out, d_C, M*N * sizeof(float));
 *       driver.check(h_out.get(), h_ref.get(), M*N, 1e-2f, 1e-2f, "MyKernel");
 *
 *       driver.print_result("MyKernel", ms, gflops);
 *   }
 */

#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>
#include "bench.h"
#include "check.h"

// -----------------------------------------------------------------------
// Typed device buffer (RAII)
// -----------------------------------------------------------------------
template<typename T>
struct DeviceBuffer {
    T *ptr = nullptr;
    size_t count = 0;

    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t n) { allocate(n); }

    void allocate(size_t n) {
        if (ptr) cuMemFree((CUdeviceptr)ptr);
        count = n;
        CUresult r = cuMemAlloc((CUdeviceptr*)&ptr, n * sizeof(T));
        if (r != CUDA_SUCCESS) {
            const char *err = nullptr;
            cuGetErrorString(r, &err);
            fprintf(stderr, "cuMemAlloc failed: %s\n", err ? err : "unknown");
            exit(1);
        }
    }

    ~DeviceBuffer() {
        if (ptr) cuMemFree((CUdeviceptr)ptr);
    }

    // Move-only
    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr(o.ptr), count(o.count) { o.ptr = nullptr; o.count = 0; }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (ptr) cuMemFree((CUdeviceptr)ptr);
        ptr = o.ptr; count = o.count;
        o.ptr = nullptr; o.count = 0;
        return *this;
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return ptr; }
    operator T*() const { return ptr; }
};

// -----------------------------------------------------------------------
// Typed host buffer (RAII)
// -----------------------------------------------------------------------
template<typename T>
struct HostBuffer {
    T *ptr = nullptr;
    size_t count = 0;

    HostBuffer() = default;
    explicit HostBuffer(size_t n) { allocate(n); }

    void allocate(size_t n) {
        if (ptr) free(ptr);
        count = n;
        ptr = (T*)malloc(n * sizeof(T));
        if (!ptr) { fprintf(stderr, "malloc failed\n"); exit(1); }
    }

    ~HostBuffer() { if (ptr) free(ptr); }

    HostBuffer(HostBuffer&& o) noexcept : ptr(o.ptr), count(o.count) { o.ptr = nullptr; o.count = 0; }
    HostBuffer& operator=(HostBuffer&& o) noexcept {
        if (ptr) free(ptr);
        ptr = o.ptr; count = o.count;
        o.ptr = nullptr; o.count = 0;
        return *this;
    }
    HostBuffer(const HostBuffer&) = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;

    T* get() const { return ptr; }
    T& operator[](size_t i) const { return ptr[i]; }
};

// -----------------------------------------------------------------------
// Benchmark driver
// -----------------------------------------------------------------------
struct BenchDriver {
    CUdevice   device = 0;
    CUcontext  context = nullptr;
    bool       context_owned = false;

    BenchDriver() = default;
    ~BenchDriver() {
        if (context_owned && context) {
            // Ensure no pending operations before destroy
            cuCtxSynchronize();
            cuDevicePrimaryCtxRelease(device);
        }
    }

    // Move-only
    BenchDriver(BenchDriver&&) = default;
    BenchDriver& operator=(BenchDriver&&) = default;
    BenchDriver(const BenchDriver&) = delete;
    BenchDriver& operator=(const BenchDriver&) = delete;

    // ------------------------------------------------------------------
    // Context management
    // ------------------------------------------------------------------
    void init_context(int dev_idx = 0) {
        CHECK_CU(cuInit(0));
        CHECK_CU(cuDeviceGet(&device, dev_idx));

        char name[256];
        CHECK_CU(cuDeviceGetName(name, sizeof(name), device));
        printf("Device: %s\n\n", name);

        CHECK_CU(cuDevicePrimaryCtxRetain(&context, device));
        CHECK_CU(cuCtxSetCurrent(context));
        context_owned = true;
    }

    // ------------------------------------------------------------------
    // Typed allocation helpers
    // ------------------------------------------------------------------
    template<typename T>
    DeviceBuffer<T> device_alloc(size_t n) {
        DeviceBuffer<T> buf;
        buf.allocate(n);
        return buf;
    }

    template<typename T>
    HostBuffer<T> host_alloc(size_t n) {
        HostBuffer<T> buf;
        buf.allocate(n);
        return buf;
    }

    // ------------------------------------------------------------------
    // Copy helpers
    // ------------------------------------------------------------------
    template<typename T>
    void copy_h2d(DeviceBuffer<T>& dst, const HostBuffer<T>& src, size_t bytes = 0) {
        if (bytes == 0) bytes = src.count * sizeof(T);
        CHECK_CU(cuMemcpyHtoD((CUdeviceptr)dst.get(), src.get(), bytes));
    }

    template<typename T>
    void copy_d2h(HostBuffer<T>& dst, const DeviceBuffer<T>& src, size_t bytes = 0) {
        if (bytes == 0) bytes = src.count * sizeof(T);
        CHECK_CU(cuMemcpyDtoH(dst.get(), (CUdeviceptr)src.get(), bytes));
    }

    void copy_d2d(CUdeviceptr dst, CUdeviceptr src, size_t bytes) {
        CHECK_CU(cuMemcpyDtoD(dst, src, bytes));
    }

    // ------------------------------------------------------------------
    // Kernel loading
    // ------------------------------------------------------------------
    CUfunction load_kernel(const char* cubin_path, const char* kernel_name, bool required = true) {
        CUmodule module = nullptr;
        CUfunction func = nullptr;
        CUresult r = cuModuleLoad(&module, cubin_path);
        if (r != CUDA_SUCCESS) {
            if (required) {
                const char *err = nullptr;
                cuGetErrorString(r, &err);
                fprintf(stderr, "ERROR: failed to load %s — %s\n", cubin_path, err ? err : "unknown");
                exit(1);
            }
            return nullptr;
        }
        CHECK_CU(cuModuleGetFunction(&func, module, kernel_name));
        return func;
    }

    // ------------------------------------------------------------------
    // Warmup + benchmark
    // ------------------------------------------------------------------
    float benchmark_kernel(CUfunction kernel,
                           dim3 grid, dim3 block, unsigned int smem,
                           void** args,
                           int warmup_runs = 3, int bench_runs = 10) {
        // Warmup
        for (int i = 0; i < warmup_runs; i++) {
            CHECK_CU(cuLaunchKernel(kernel, grid.x, grid.y, grid.z,
                                    block.x, block.y, block.z,
                                    smem, 0, args, nullptr));
        }
        CHECK_CU(cuCtxSynchronize());

        // Benchmark
        BenchTimer timer;
        timer.start();
        for (int i = 0; i < bench_runs; i++) {
            CHECK_CU(cuLaunchKernel(kernel, grid.x, grid.y, grid.z,
                                    block.x, block.y, block.z,
                                    smem, 0, args, nullptr));
        }
        CHECK_CU(cuCtxSynchronize());
        float total_ms = timer.stop_ms();
        return total_ms / (float)bench_runs;
    }

    // ------------------------------------------------------------------
    // Correctness check
    // ------------------------------------------------------------------
    template<typename T>
    bool check(const T* gpu, const T* ref, int n,
               T abs_tol, T rel_tol,
               const char* label, bool print_first = true) {
        CheckResult r = check_fp32(gpu, ref, n, abs_tol, rel_tol, print_first);
        print_check_result(label, r);
        return r.num_errors == 0;
    }

    // ------------------------------------------------------------------
    // Result printing
    // ------------------------------------------------------------------
    static void print_result(const char* label, int M, int N, int K,
                             float ms, double gflops,
                             double ref_gflops = 0.0) {
        print_gemm_result(label, M, N, K, ms, gflops, ref_gflops);
    }

    static void print_result_raw(const char* label, float ms, const char* extra = "") {
        printf("  %-30s %7.3f ms   %s\n", label, ms, extra);
    }
};
