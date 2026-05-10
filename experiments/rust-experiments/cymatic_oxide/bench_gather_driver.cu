// bench_gather_driver.cu
//
// Driver-API benchmark that loads gather_sum from an arbitrary cubin path,
// so the same harness can run nvcc-built and oxide-built kernels on
// identical data. Output is comparable to phase4/cymatic/bench_cymatic.cu
// but isolates the kernel under test.
//
// Usage:
//   ./bench_gather_driver <cubin> <perm.bin> <traces.bin>
//                         [iters_per_run=200] [warmup=5] [runs=5]
//
// Both cubins must export a function symbol named `gather_sum` with the
// same parameter ABI. nvcc's gather_nvcc.cu uses extern "C". Oxide's
// kernel #[kernel] also exports `gather_sum`. ABIs differ:
//   nvcc:  (data*, idx*, out*, int n, int iters)
//   oxide: (data*, data_len, idx*, idx_len, out*, out_len, u32 iters)
// Pass --abi nvcc | oxide on the command line so the launcher knows how
// to build the kernel-arg array.

#include <cuda.h>
#include <cuda_runtime.h>  // only for cudaEvent timing convenience
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <string>
#include <algorithm>

#define CHECK_CU(call) do { CUresult _e = (call); if (_e != CUDA_SUCCESS) { \
    const char *s; cuGetErrorString(_e, &s); \
    fprintf(stderr, "cu error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, s); \
    exit(1); } } while(0)

#define CHECK_RT(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) { \
    fprintf(stderr, "rt error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, \
            cudaGetErrorString(_e)); exit(1); } } while(0)

struct Trace {
    std::string name;
    std::vector<int> rmi;
};

static void read_perm(const char *path, int &grid_n, int &n_inside,
                       std::vector<int> &perm) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s\n", path); exit(1); }
    fread(&grid_n, 4, 1, f);
    fread(&n_inside, 4, 1, f);
    perm.resize(n_inside);
    fread(perm.data(), 4, n_inside, f);
    fclose(f);
}

static void read_traces(const char *path, std::vector<Trace> &traces) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s\n", path); exit(1); }
    int n_traces = 0;
    fread(&n_traces, 4, 1, f);
    traces.resize(n_traces);
    for (int i = 0; i < n_traces; ++i) {
        int name_len = 0, n = 0;
        fread(&name_len, 4, 1, f);
        std::string nm(name_len, '\0');
        fread(&nm[0], 1, name_len, f);
        fread(&n, 4, 1, f);
        traces[i].name = nm;
        traces[i].rmi.resize(n);
        fread(traces[i].rmi.data(), 4, n, f);
    }
    fclose(f);
}

enum Abi { ABI_NVCC, ABI_OXIDE };

// Launch helper. n_data and n_idx are slice lengths. n_idx == trace n.
static double bench_kernel(CUfunction fn, Abi abi,
                            CUdeviceptr d_data, size_t n_data,
                            CUdeviceptr d_idx,  size_t n_idx,
                            CUdeviceptr d_out,
                            int iters, int warmup, int runs) {
    int n = (int)n_idx;
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;

    if (n < 100000) iters *= 5;
    if (n < 10000)  iters *= 5;

    // Build kernel-arg arrays.
    // nvcc:  (data*, idx*, out*, int n, int iters)
    // oxide: (data*, data_len, idx*, idx_len, out*, out_len, u32 iters)
    void *args_nvcc[5];
    int   n_arg = n;
    int   iters_arg = iters;
    args_nvcc[0] = &d_data;
    args_nvcc[1] = &d_idx;
    args_nvcc[2] = &d_out;
    args_nvcc[3] = &n_arg;
    args_nvcc[4] = &iters_arg;

    uint64_t data_len = n_data, idx_len = n_idx, out_len = 1;
    uint32_t iters_u32 = (uint32_t)iters;
    void *args_oxide[7];
    args_oxide[0] = &d_data;
    args_oxide[1] = &data_len;
    args_oxide[2] = &d_idx;
    args_oxide[3] = &idx_len;
    args_oxide[4] = &d_out;
    args_oxide[5] = &out_len;
    args_oxide[6] = &iters_u32;

    void **args = (abi == ABI_NVCC) ? args_nvcc : args_oxide;

    cudaEvent_t e0, e1;
    CHECK_RT(cudaEventCreate(&e0));
    CHECK_RT(cudaEventCreate(&e1));

    float zero = 0.0f;
    for (int i = 0; i < warmup; ++i) {
        CHECK_CU(cuMemcpyHtoD(d_out, &zero, sizeof(float)));
        CHECK_CU(cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, 0, args, 0));
    }
    CHECK_RT(cudaDeviceSynchronize());

    std::vector<double> times;
    times.reserve(runs);
    for (int i = 0; i < runs; ++i) {
        CHECK_CU(cuMemcpyHtoD(d_out, &zero, sizeof(float)));
        CHECK_RT(cudaEventRecord(e0));
        CHECK_CU(cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, 0, args, 0));
        CHECK_RT(cudaEventRecord(e1));
        CHECK_RT(cudaEventSynchronize(e1));
        float ms = 0;
        CHECK_RT(cudaEventElapsedTime(&ms, e0, e1));
        times.push_back(ms / iters);
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <cubin> <abi=nvcc|oxide> <perm.bin> <traces.bin> "
                        "[iters=200] [warmup=5] [runs=5]\n", argv[0]);
        return 1;
    }
    const char *cubin_path = argv[1];
    Abi abi = (strcmp(argv[2], "oxide") == 0) ? ABI_OXIDE : ABI_NVCC;
    const char *perm_path  = argv[3];
    const char *trace_path = argv[4];
    int iters  = (argc > 5) ? atoi(argv[5]) : 200;
    int warmup = (argc > 6) ? atoi(argv[6]) : 5;
    int runs   = (argc > 7) ? atoi(argv[7]) : 5;

    CHECK_CU(cuInit(0));
    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK_CU(cuCtxCreate(&ctx, NULL, 0, dev));

    CUmodule mod;
    CHECK_CU(cuModuleLoad(&mod, cubin_path));
    CUfunction fn;
    CHECK_CU(cuModuleGetFunction(&fn, mod, "gather_sum"));

    int grid_n = 0, n_inside = 0;
    std::vector<int> perm;
    read_perm(perm_path, grid_n, n_inside, perm);

    std::vector<Trace> traces;
    read_traces(trace_path, traces);

    char dev_name[128]; cuDeviceGetName(dev_name, 128, dev);
    int sm_count = 0; cuDeviceGetAttribute(&sm_count,
        CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    printf("=== gather_sum runtime bench ===\n");
    printf("Device: %s  (%d SMs)\n", dev_name, sm_count);
    printf("Cubin:  %s   ABI: %s\n", cubin_path,
           abi == ABI_NVCC ? "nvcc" : "oxide");
    printf("Grid:   %dx%d  n_inside=%d  buffer=%.2f MB\n",
           grid_n, grid_n, n_inside, n_inside * 4.0 / 1e6);
    printf("Traces: %zu  iters=%d  warmup=%d  runs=%d\n\n",
           traces.size(), iters, warmup, runs);

    std::vector<float> h_row(n_inside), h_cym(n_inside);
    for (int k = 0; k < n_inside; ++k) {
        float v = sinf((float)k * 0.0001f) * 3.7f + (float)(k & 0xFF) * 0.013f;
        h_row[k]       = v;
        h_cym[perm[k]] = v;
    }

    CUdeviceptr d_row, d_cym, d_out;
    CHECK_CU(cuMemAlloc(&d_row, n_inside * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_cym, n_inside * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_out, sizeof(float)));
    CHECK_CU(cuMemcpyHtoD(d_row, h_row.data(), n_inside * sizeof(float)));
    CHECK_CU(cuMemcpyHtoD(d_cym, h_cym.data(), n_inside * sizeof(float)));

    printf("%-20s %10s   %12s %12s   %12s %12s   %8s\n",
           "trace", "n", "row_ms", "row_GB/s",
           "cym_ms", "cym_GB/s", "speedup");
    printf("%-20s %10s   %12s %12s   %12s %12s   %8s\n",
           "----------", "-------", "------", "--------",
           "------", "--------", "-------");

    for (auto &tr : traces) {
        int n = (int)tr.rmi.size();
        if (n == 0) continue;  // empty traces (cuMemAlloc rejects 0)
        std::vector<int> idx_row = tr.rmi;
        std::vector<int> idx_cym(n);
        for (int t = 0; t < n; ++t) idx_cym[t] = perm[tr.rmi[t]];

        CUdeviceptr d_idx_row, d_idx_cym;
        CHECK_CU(cuMemAlloc(&d_idx_row, n * sizeof(int)));
        CHECK_CU(cuMemAlloc(&d_idx_cym, n * sizeof(int)));
        CHECK_CU(cuMemcpyHtoD(d_idx_row, idx_row.data(), n * sizeof(int)));
        CHECK_CU(cuMemcpyHtoD(d_idx_cym, idx_cym.data(), n * sizeof(int)));

        double row_ms = bench_kernel(fn, abi, d_row, n_inside, d_idx_row, n, d_out,
                                      iters, warmup, runs);
        double cym_ms = bench_kernel(fn, abi, d_cym, n_inside, d_idx_cym, n, d_out,
                                      iters, warmup, runs);

        double bytes = (double)n * 4.0;
        double row_gbs = bytes / (row_ms * 1e-3) / 1e9;
        double cym_gbs = bytes / (cym_ms * 1e-3) / 1e9;

        printf("%-20s %10d   %12.4f %12.1f   %12.4f %12.1f   %7.2fx\n",
               tr.name.c_str(), n,
               row_ms, row_gbs,
               cym_ms, cym_gbs,
               row_ms / cym_ms);

        cuMemFree(d_idx_row);
        cuMemFree(d_idx_cym);
    }

    cuMemFree(d_row);
    cuMemFree(d_cym);
    cuMemFree(d_out);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);
    return 0;
}
