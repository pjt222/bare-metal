// bench.cu
//
// Measures effective gather bandwidth for two memory layouts of the same
// data: row-major-inside vs cymatic-permuted.
//
// Setup:
//   - Read perm.bin (grid_n, n_inside, perm[n_inside])
//   - Read traces.bin (list of access traces in row-major-inside index space)
//   - Allocate two device buffers of size n_inside × float, identical content
//     in different positions:
//       data_row[k]            = value(k)
//       data_cym[perm[k]]      = value(k)
//   - For each trace and each layout, build the index buffer:
//       idx_row[t] = trace[t]
//       idx_cym[t] = perm[trace[t]]
//   - Run a gather-sum kernel many iterations; report effective bandwidth.
//
// The two layouts touch the same logical data via the same trace, just at
// different physical addresses. Bandwidth difference ⇒ pure layout effect
// (warp coalescing, L1/L2 hit rate).
//
// Build: see Makefile.
// Run:   ./bench [iters_per_run=200] [warmup=5] [runs=5]

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <string>
#include <algorithm>

#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "cuda error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, \
            cudaGetErrorString(e)); exit(1); } } while(0)

struct Trace {
    std::string name;
    std::vector<int> rmi;   // row-major-inside indices
};

// ---- File parsing ----------------------------------------------------------

static void read_perm(const char *path, int &grid_n, int &n_inside,
                       std::vector<int> &perm) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    if (fread(&grid_n, 4, 1, f) != 1) { fprintf(stderr, "read grid_n\n"); exit(1); }
    if (fread(&n_inside, 4, 1, f) != 1) { fprintf(stderr, "read n_inside\n"); exit(1); }
    perm.resize(n_inside);
    if ((int)fread(perm.data(), 4, n_inside, f) != n_inside) {
        fprintf(stderr, "read perm\n"); exit(1);
    }
    fclose(f);
}

static void read_traces(const char *path, std::vector<Trace> &traces) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    int n_traces = 0;
    if (fread(&n_traces, 4, 1, f) != 1) { fprintf(stderr, "read n_traces\n"); exit(1); }
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

// ---- Kernels ---------------------------------------------------------------

// Gather + reduce. `iters` repeats over the same trace to amortize launch
// overhead and stress the cache hierarchy.
__global__ void gather_sum(const float *__restrict__ data,
                            const int   *__restrict__ idx,
                            float       *__restrict__ out,
                            int n, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float s = 0.0f;
    for (int it = 0; it < iters; ++it) {
        for (int k = tid; k < n; k += stride) {
            s += data[idx[k]];
        }
    }
    if (tid == 0) atomicAdd(out, s);  // single per-block reducer is fine for bench
}

// ---- Host helpers ----------------------------------------------------------

// Returns median ms across `runs` measurements. Auto-scales iters for tiny
// traces so each measured kernel runs at least ~1 ms (above timer noise).
static double bench_kernel(const float *d_data, const int *d_idx,
                            float *d_out, int n, int iters,
                            int warmup, int runs) {
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;

    // Scale iters up if trace is tiny so kernel time is well above ~10 us
    // event-timer noise. Target ~5 ms per kernel.
    if (n < 100000) iters *= 5;
    if (n < 10000)  iters *= 5;

    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));

    for (int i = 0; i < warmup; ++i) {
        CHECK(cudaMemset(d_out, 0, sizeof(float)));
        gather_sum<<<blocks, threads>>>(d_data, d_idx, d_out, n, iters);
    }
    CHECK(cudaDeviceSynchronize());

    std::vector<double> times;
    times.reserve(runs);
    for (int i = 0; i < runs; ++i) {
        CHECK(cudaMemset(d_out, 0, sizeof(float)));
        CHECK(cudaEventRecord(e0));
        gather_sum<<<blocks, threads>>>(d_data, d_idx, d_out, n, iters);
        CHECK(cudaEventRecord(e1));
        CHECK(cudaEventSynchronize(e1));
        float ms = 0.0f;
        CHECK(cudaEventElapsedTime(&ms, e0, e1));
        times.push_back(ms / iters);  // ms per single kernel iter (normalized)
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];   // median, normalized to 1 iter
}

// ---- main ------------------------------------------------------------------

int main(int argc, char **argv) {
    int iters  = (argc > 1) ? atoi(argv[1]) : 200;
    int warmup = (argc > 2) ? atoi(argv[2]) : 5;
    int runs   = (argc > 3) ? atoi(argv[3]) : 5;

    int grid_n = 0, n_inside = 0;
    std::vector<int> perm;
    read_perm("perm.bin", grid_n, n_inside, perm);

    std::vector<Trace> traces;
    read_traces("traces.bin", traces);

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== Cymatic memory-mapping benchmark ===\n");
    printf("Device: %s  (sm_%d%d, %d SMs)\n", prop.name,
           prop.major, prop.minor, prop.multiProcessorCount);
    printf("Grid:   %d × %d   n_inside=%d   buffer=%.2f MB\n",
           grid_n, grid_n, n_inside, n_inside * 4.0 / 1e6);
    printf("Traces: %zu   iters/run=%d   warmup=%d   runs=%d\n\n",
           traces.size(), iters, warmup, runs);

    // Build host data: row layout has value(k) at position k, cymatic layout has
    // value(k) at position perm[k]. Use a deterministic value so a checksum
    // would verify if needed.
    std::vector<float> h_row(n_inside), h_cym(n_inside);
    for (int k = 0; k < n_inside; ++k) {
        float v = sinf((float)k * 0.0001f) * 3.7f + (float)(k & 0xFF) * 0.013f;
        h_row[k]        = v;
        h_cym[perm[k]]  = v;
    }

    float *d_row = nullptr, *d_cym = nullptr, *d_out = nullptr;
    CHECK(cudaMalloc(&d_row, n_inside * sizeof(float)));
    CHECK(cudaMalloc(&d_cym, n_inside * sizeof(float)));
    CHECK(cudaMalloc(&d_out, sizeof(float)));
    CHECK(cudaMemcpy(d_row, h_row.data(), n_inside * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_cym, h_cym.data(), n_inside * sizeof(float), cudaMemcpyHostToDevice));

    printf("%-20s %10s   %12s %12s %8s   %12s %12s %8s   %8s\n",
           "trace", "n", "row_ms", "row_GB/s", "row_eff%",
           "cym_ms", "cym_GB/s", "cym_eff%", "speedup");
    printf("%-20s %10s   %12s %12s %8s   %12s %12s %8s   %8s\n",
           "----------", "-------", "------", "--------", "------",
           "------", "--------", "------", "-------");

    const double dram_peak_gbs = 608.0;  // RTX 3070 Ti

    for (auto &tr : traces) {
        int n = (int)tr.rmi.size();

        // Build idx buffers
        std::vector<int> idx_row = tr.rmi;
        std::vector<int> idx_cym(n);
        for (int t = 0; t < n; ++t) idx_cym[t] = perm[tr.rmi[t]];

        int *d_idx_row = nullptr, *d_idx_cym = nullptr;
        CHECK(cudaMalloc(&d_idx_row, n * sizeof(int)));
        CHECK(cudaMalloc(&d_idx_cym, n * sizeof(int)));
        CHECK(cudaMemcpy(d_idx_row, idx_row.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_idx_cym, idx_cym.data(), n * sizeof(int), cudaMemcpyHostToDevice));

        // bench_kernel returns median ms per single iter (scale-normalized).
        double row_ms = bench_kernel(d_row, d_idx_row, d_out, n, iters, warmup, runs);
        double cym_ms = bench_kernel(d_cym, d_idx_cym, d_out, n, iters, warmup, runs);

        // Bytes touched per single iter: n × 4 (data only — index is sequential
        // and amortized via L1; not counted to keep cym/row comparison honest).
        double bytes = (double)n * 4.0;
        double row_gbs = bytes / (row_ms * 1e-3) / 1e9;
        double cym_gbs = bytes / (cym_ms * 1e-3) / 1e9;
        double row_eff = 100.0 * row_gbs / dram_peak_gbs;
        double cym_eff = 100.0 * cym_gbs / dram_peak_gbs;

        printf("%-20s %10d   %12.3f %12.1f %7.1f%%   %12.3f %12.1f %7.1f%%   %7.2fx\n",
               tr.name.c_str(), n,
               row_ms, row_gbs, row_eff,
               cym_ms, cym_gbs, cym_eff,
               row_ms / cym_ms);

        cudaFree(d_idx_row);
        cudaFree(d_idx_cym);
    }

    cudaFree(d_row);
    cudaFree(d_cym);
    cudaFree(d_out);
    return 0;
}
