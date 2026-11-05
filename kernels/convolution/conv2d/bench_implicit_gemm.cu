/*
 * bench_implicit_gemm.cu — Implicit GEMM vs explicit im2col + GEMM for conv2d
 *
 * Tests the hypothesis: eliminating the 23.6 MB col buffer saves DRAM
 * traffic and speeds up small-channel and large-spatial convolutions.
 *
 * Kernels compared:
 *   Explicit:   im2col_nhwc_fp16 (writes col buffer) + wmma_gemm_conv (reads col buffer)
 *   Implicit:   implicit_gemm_conv (single kernel, no col buffer, indices on-the-fly)
 *
 * Build:
 *   nvcc -arch=sm_86 -O2 -o bench_implicit_gemm bench_implicit_gemm.cu \
 *        -lcuda -I../../kernels/_common
 *
 * Usage:
 *   bench_implicit_gemm                # default: the gate's measurement
 *   bench_implicit_gemm --per-launch   # also print per-launch timing (#223)
 *
 * Positional arguments are ignored (bench_regress passes the config key
 * as `1 64 64 320 320`); unknown options exit 2.
 *
 * --per-launch records a CUevent after every launch of the 20 warmup and
 * 100 timed launches, per config, and prints one raw line and one summary
 * line per kernel and phase. The default path creates no extra events and
 * prints no extra lines, so bench_regress and conv2d_variance.R see the
 * same output as before. See print_per_launch() for the line format.
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <unistd.h>
#include <vector>
#include <memory>
#include <algorithm>
#include <cuda.h>
#include <cuda_fp16.h>

#include "../../_common/bench.h"
#include "../../_common/check.h"

// -----------------------------------------------------------------------
// Per-launch timing (#223 step 1). Off unless --per-launch is given.
//
// A chain of events in the default stream: event 0 is recorded before the
// first launch of a phase, then one event after every launch. The time
// between consecutive events is the GPU time of the launch in between,
// so a clock ramp shows up as slow early launches, a throttle event as a
// slow run in the middle, and a uniformly slow window as neither.
//
// Event 0 of a phase is recorded while the GPU is idle, so the first
// interval also contains the host's submission latency for the first
// launch (microseconds against a ~1 ms kernel).
// -----------------------------------------------------------------------
struct LaunchEventChain {
    std::vector<CUevent> events;

    explicit LaunchEventChain(size_t event_count) : events(event_count) {
        for (CUevent &event : events)
            CHECK_CU(cuEventCreate(&event, CU_EVENT_DEFAULT));
    }
    ~LaunchEventChain() {
        for (CUevent event : events) cuEventDestroy(event);
    }
    LaunchEventChain(const LaunchEventChain &) = delete;
    LaunchEventChain &operator=(const LaunchEventChain &) = delete;

    void record(size_t event_index) {
        CHECK_CU(cuEventRecord(events[event_index], NULL));
    }
    // Caller must have synchronized past events[to_index].
    float elapsed_us(size_t from_index, size_t to_index) const {
        float elapsed_ms = 0.0f;
        CHECK_CU(cuEventElapsedTime(&elapsed_ms, events[from_index], events[to_index]));
        return elapsed_ms * 1000.0f;
    }
};

// Nearest-rank percentile: the smallest value with at least
// `fraction` of the sample at or below it.
static float nearest_rank(const std::vector<float> &sorted_values, double fraction) {
    size_t sample_count = sorted_values.size();
    size_t rank = (size_t)ceil(fraction * (double)sample_count);
    if (rank < 1) rank = 1;
    if (rank > sample_count) rank = sample_count;
    return sorted_values[rank - 1];
}

static double mean_of_range(const std::vector<float> &values, size_t begin, size_t end) {
    double total = 0.0;
    for (size_t index = begin; index < end; index++) total += values[index];
    return end > begin ? total / (double)(end - begin) : 0.0;
}

// Two lines per (kernel, phase), tab-free, `key=value` fields:
//
//   per-launch config=<N_H_W_Cin_Cout> kernel=<name> phase=<warmup|timed> n=<count> us=<v1,v2,...>
//   per-launch-summary config=... kernel=... phase=... n=... min_us=... p10_us=... p50_us=...
//       p90_us=... max_us=... first5_us=... last5_us=... sum_us=... [aggregate_us=...]
//
// Values are microseconds in launch order. aggregate_us, on timed phases
// only, is the BenchTimer window the default output reports (as ms per
// launch). sum_us telescopes to the chain's span (last event - event 0),
// so it matches aggregate_us up to the gaps at the window's edges: a
// consistency check on the chain, NOT a measure of what the extra events
// cost, which sits inside every interval. Whether --per-launch perturbs
// the timing is read only by comparing the default-mode aggregate with
// the --per-launch aggregate across runs. The lines avoid `ms`/`GFLOPS` and
// never start with `===`/`---`, so parse_throughput() cannot select them.
// `print_raw = false` prints only the summary (used for the explicit pair,
// whose raw values are im2col + gemm per index).
static void print_per_launch(const char *config_key, const char *kernel_name,
                             const char *phase_name, const std::vector<float> &values_us,
                             bool print_raw, double aggregate_us) {
    size_t sample_count = values_us.size();
    if (sample_count == 0) return;
    if (print_raw) {
        printf("  per-launch config=%s kernel=%s phase=%s n=%zu us=",
               config_key, kernel_name, phase_name, sample_count);
        for (size_t index = 0; index < sample_count; index++)
            printf(index == 0 ? "%.2f" : ",%.2f", (double)values_us[index]);
        printf("\n");
    }
    std::vector<float> sorted_values(values_us);
    std::sort(sorted_values.begin(), sorted_values.end());
    size_t edge_count = sample_count < 5 ? sample_count : 5;
    printf("  per-launch-summary config=%s kernel=%s phase=%s n=%zu"
           " min_us=%.2f p10_us=%.2f p50_us=%.2f p90_us=%.2f max_us=%.2f"
           " first5_us=%.2f last5_us=%.2f sum_us=%.2f",
           config_key, kernel_name, phase_name, sample_count,
           (double)sorted_values.front(),
           (double)nearest_rank(sorted_values, 0.10),
           (double)nearest_rank(sorted_values, 0.50),
           (double)nearest_rank(sorted_values, 0.90),
           (double)sorted_values.back(),
           mean_of_range(values_us, 0, edge_count),
           mean_of_range(values_us, sample_count - edge_count, sample_count),
           mean_of_range(values_us, 0, sample_count) * (double)sample_count);
    if (aggregate_us >= 0.0) printf(" aggregate_us=%.2f", aggregate_us);
    printf("\n");
}

// Helper: find cubin in CWD or in the binary's directory
static bool find_cubin(const char *name, char *out_path, size_t out_len) {
    if (access(name, R_OK) == 0) {
        strncpy(out_path, name, out_len - 1);
        out_path[out_len - 1] = '\0';
        return true;
    }
    char exe_path[4096];
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len > 0) {
        exe_path[len] = '\0';
        char *last_slash = strrchr(exe_path, '/');
        if (last_slash) {
            *last_slash = '\0';
            snprintf(out_path, out_len, "%s/%s", exe_path, name);
            if (access(out_path, R_OK) == 0) return true;
        }
    }
    return false;
}

// ---- Tile constants (must match kernel files) ----
#define BLOCK_M    64
#define BLOCK_N    64
#define BLOCK_K    16
#define WMMA_M     16
#define WMMA_N     16

// Explicit im2col smem: smem_A + smem_B (padded)
#define SMEM_A_STRIDE_EXPLICIT  (BLOCK_K + 8)
#define SMEM_B_STRIDE_EXPLICIT  (BLOCK_N + 8)
#define SMEM_BYTES_EXPLICIT     ((BLOCK_M * SMEM_A_STRIDE_EXPLICIT + BLOCK_K * SMEM_B_STRIDE_EXPLICIT) * 2)
// = (64*24 + 16*72) * 2 = (1536 + 1152) * 2 = 5376 bytes

// Implicit GEMM smem: smem_A + smem_B + coordinate tables
// smem_A: 64*24 halfs = 3072 B
// smem_B: 16*72 halfs = 2304 B
// M tables: 3*64 ints = 768 B
// K tables: 3*16 ints = 192 B
// Total: 6336 bytes
#define SMEM_BYTES_IMPLICIT     (SMEM_BYTES_EXPLICIT + (3 * BLOCK_M + 3 * BLOCK_K) * 4)

// -----------------------------------------------------------------------
// CPU reference: 3×3 NHWC convolution
// -----------------------------------------------------------------------
static void cpu_conv2d_nhwc_ref(
    const float *X, const float *W, float *Y,
    int N, int H, int W_dim, int Cin, int Cout
) {
    memset(Y, 0, (size_t)N * H * W_dim * Cout * sizeof(float));
    for (int n = 0; n < N; n++)
    for (int h = 0; h < H; h++)
    for (int w = 0; w < W_dim; w++)
    for (int cout_c = 0; cout_c < Cout; cout_c++) {
        double acc = 0.0;
        for (int kh = 0; kh < 3; kh++)
        for (int kw = 0; kw < 3; kw++) {
            int h_in = h + kh - 1, w_in = w + kw - 1;
            if (h_in < 0 || h_in >= H || w_in < 0 || w_in >= W_dim) continue;
            for (int cin = 0; cin < Cin; cin++) {
                size_t x_idx = ((size_t)n * H * W_dim + h_in * W_dim + w_in) * Cin + cin;
                size_t w_idx = (size_t)cout_c * 9 * Cin + (kh * 3 + kw) * Cin + cin;
                acc += (double)X[x_idx] * W[w_idx];
            }
        }
        size_t y_idx = ((size_t)n * H * W_dim + h * W_dim + w) * Cout + cout_c;
        Y[y_idx] = (float)acc;
    }
}

static unsigned short fp32_to_fp16_bits(float val) {
    __half h = __float2half(val);
    unsigned short bits;
    memcpy(&bits, &h, 2);
    return bits;
}

// Reshape: W_direct[Cout, kH, kW, Cin] FP32 → W_t[Cin*kH*kW, Cout] FP16
static void reshape_weights_to_col(
    const float *W_direct, unsigned short *W_t,
    int Cout, int Cin, int kH, int kW
) {
    int K_dim = Cin * kH * kW;
    for (int k = 0; k < K_dim; k++) {
        int cin    = k / (kH * kW);
        int k_pos  = k % (kH * kW);
        int kh_idx = k_pos / kW;
        int kw_idx = k_pos % kW;
        for (int cout_c = 0; cout_c < Cout; cout_c++) {
            float val = W_direct[(size_t)cout_c * kH * kW * Cin
                                  + (kh_idx * kW + kw_idx) * Cin + cin];
            W_t[(size_t)k * Cout + cout_c] = fp32_to_fp16_bits(val);
        }
    }
}

static void fill_rand(float *arr, size_t n, float scale = 0.1f) {
    for (size_t i = 0; i < n; i++)
        arr[i] = scale * (2.0f * (float)rand() / RAND_MAX - 1.0f);
}

static void print_usage(const char *program_name) {
    fprintf(stderr, "usage: %s [--per-launch] [ignored positional args]\n", program_name);
}

int main(int argc, char **argv) {
    // Positional arguments are accepted and ignored, as they always were:
    // bench_regress.R passes the data/baselines.json config key split on
    // '_' (`1 64 64 320 320`), and this bench runs every config regardless.
    // Its other runners pass none: bench_all.R (bench_all.yml `args: []`),
    // grid_measure.R (grid_sweep.yml `args: []`) and conv2d_variance.R. Unknown OPTIONS (anything starting with '-') are
    // refused, so a mistyped flag cannot silently run the default mode in
    // a campaign that meant to record per-launch.
    bool per_launch = false;
    for (int arg_index = 1; arg_index < argc; arg_index++) {
        const char *arg = argv[arg_index];
        if (strcmp(arg, "--per-launch") == 0) {
            per_launch = true;
        } else if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (arg[0] == '-') {
            fprintf(stderr, "unknown option: %s\n", arg);
            print_usage(argv[0]);
            return 2;
        }
    }

    const int warmup_launches = 20;
    const int timed_launches  = 100;

    // Test configurations: (N, H, W, Cin, Cout, label)
    struct Config {
        int N, H, W_dim, Cin, Cout;
        const char *label;
    } configs[] = {
        { 1, 64, 64, 320, 320, "SD 64×64  Cin=Cout=320  (baseline, large col buffer)" },
        { 1, 32, 32, 640, 640, "SD 32×32  Cin=Cout=640  (smaller spatial, larger channels)" },
        { 1,128,128, 160, 160, "SD 128×128 Cin=Cout=160 (large spatial, small col buffer penalty)"},
        { 1, 64, 64,  64,  64, "Small     64×64  Cin=Cout=64  (index decode overhead visible)"},
    };
    int num_configs = (int)(sizeof(configs) / sizeof(configs[0]));

    CHECK_CU(cuInit(0));
    CUdevice  cu_dev; CHECK_CU(cuDeviceGet(&cu_dev, 0));
    CUcontext cu_ctx; CHECK_CU(cuDevicePrimaryCtxRetain(&cu_ctx, cu_dev));
    CHECK_CU(cuCtxSetCurrent(cu_ctx));

    CUmodule mod_explicit, mod_implicit;
    char cubin_path[4096];
    if (!find_cubin("conv2d_im2col.sm_86.cubin", cubin_path, sizeof(cubin_path))) {
        fprintf(stderr, "Cannot find conv2d_im2col.sm_86.cubin\n"); return 1;
    }
    CHECK_CU(cuModuleLoad(&mod_explicit, cubin_path));
    if (!find_cubin("conv2d_implicit_gemm.sm_86.cubin", cubin_path, sizeof(cubin_path))) {
        fprintf(stderr, "Cannot find conv2d_implicit_gemm.sm_86.cubin\n"); return 1;
    }
    CHECK_CU(cuModuleLoad(&mod_implicit, cubin_path));

    CUfunction fn_im2col, fn_gemm_explicit, fn_gemm_implicit;
    CHECK_CU(cuModuleGetFunction(&fn_im2col,       mod_explicit, "im2col_nhwc_fp16"));
    CHECK_CU(cuModuleGetFunction(&fn_gemm_explicit, mod_explicit, "wmma_gemm_conv"));
    CHECK_CU(cuModuleGetFunction(&fn_gemm_implicit, mod_implicit, "implicit_gemm_conv"));

    CHECK_CU(cuFuncSetAttribute(fn_gemm_explicit, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, SMEM_BYTES_EXPLICIT));
    CHECK_CU(cuFuncSetAttribute(fn_gemm_implicit, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, SMEM_BYTES_IMPLICIT));

    printf("=== Implicit GEMM vs Explicit im2col + GEMM ===\n");
    printf("  Explicit: im2col_nhwc_fp16 → col buffer → wmma_gemm_conv\n");
    printf("  Implicit: implicit_gemm_conv (indices on-the-fly, no col buffer)\n");
    if (per_launch)
        printf("  Per-launch timing: on (--per-launch), %d warmup + %d timed launches per kernel\n",
               warmup_launches, timed_launches);
    printf("\n");

    for (int ci = 0; ci < num_configs; ci++) {
        int N     = configs[ci].N;
        int H     = configs[ci].H;
        int W_dim = configs[ci].W_dim;
        int Cin   = configs[ci].Cin;
        int Cout  = configs[ci].Cout;

        int kH = 3, kW = 3, pad = 1;
        int out_H = H, out_W = W_dim;
        int M_dim = N * out_H * out_W;
        int K_dim = Cin * kH * kW;

        float col_mb   = (float)M_dim * K_dim * 2 / 1e6f;
        double flops_g = 2.0 * M_dim * K_dim * Cout / 1e9;

        printf("--- %s ---\n", configs[ci].label);
        printf("  M=%d K=%d Cout=%d  col_buf=%.1f MB  GFLOPs=%.2f\n",
               M_dim, K_dim, Cout, col_mb, flops_g);

        if (M_dim % 16 != 0 || K_dim % 16 != 0 || Cout % 16 != 0) {
            printf("  SKIP: M, K, or Cout not divisible by 16 (WMMA constraint)\n\n");
            continue;
        }

        // ---- Allocate host memory ----
        size_t X_elems   = (size_t)N * H * W_dim * Cin;
        size_t W_elems   = (size_t)Cout * kH * kW * Cin;
        size_t Y_elems   = (size_t)M_dim * Cout;
        size_t col_elems = (size_t)M_dim * K_dim;
        size_t Wt_elems  = (size_t)K_dim * Cout;

        float          *host_X     = new float[X_elems];
        float          *host_W     = new float[W_elems];
        float          *host_Y_ref = new float[Y_elems];
        float          *host_Y_exp = new float[Y_elems];
        float          *host_Y_imp = new float[Y_elems];
        unsigned short *host_Wt    = new unsigned short[Wt_elems];

        srand(42 + ci);
        fill_rand(host_X, X_elems, 0.1f);
        fill_rand(host_W, W_elems, 0.1f / sqrtf((float)(Cin * kH * kW)));
        reshape_weights_to_col(host_W, host_Wt, Cout, Cin, kH, kW);

        // CPU reference (skip for large configs)
        bool ran_cpu = (X_elems < 16 * 1024 * 1024 / 4);  // skip if X > 16 MB
        if (ran_cpu) {
            cpu_conv2d_nhwc_ref(host_X, host_W, host_Y_ref, N, H, W_dim, Cin, Cout);
        }

        // ---- Device allocations ----
        CUdeviceptr d_X, d_col, d_Wt, d_Y;
        CHECK_CU(cuMemAlloc(&d_X,   X_elems   * sizeof(float)));
        CHECK_CU(cuMemAlloc(&d_col, col_elems * sizeof(unsigned short)));
        CHECK_CU(cuMemAlloc(&d_Wt,  Wt_elems  * sizeof(unsigned short)));
        CHECK_CU(cuMemAlloc(&d_Y,   Y_elems   * sizeof(float)));

        CHECK_CU(cuMemcpyHtoD(d_X,  host_X,  X_elems  * sizeof(float)));
        CHECK_CU(cuMemcpyHtoD(d_Wt, host_Wt, Wt_elems * sizeof(unsigned short)));

        int grid_m = (M_dim + BLOCK_M - 1) / BLOCK_M;
        int grid_n = (Cout  + BLOCK_N - 1) / BLOCK_N;

        int im2col_threads = 256;
        int im2col_blocks  = (int)((col_elems + im2col_threads - 1) / im2col_threads);
        if (im2col_blocks > 65535) im2col_blocks = 65535;

        void *im2col_args[] = { &d_X, &d_col, &N, &H, &W_dim, &Cin, &kH, &kW, &pad, &out_H, &out_W };
        void *gemm_exp_args[]= { &d_col, &d_Wt, &d_Y, &M_dim, &K_dim, &Cout };
        void *gemm_imp_args[]= { &d_X, &d_Wt, &d_Y,
                                  &N, &H, &W_dim, &Cin,
                                  &kH, &kW, &pad,
                                  &out_H, &out_W,
                                  &M_dim, &K_dim, &Cout };

        // ---- Correctness: explicit ----
        CHECK_CU(cuMemsetD32(d_Y, 0, Y_elems));
        CHECK_CU(cuLaunchKernel(fn_im2col,
                                im2col_blocks, 1, 1, im2col_threads, 1, 1,
                                0, 0, im2col_args, 0));
        CHECK_CU(cuLaunchKernel(fn_gemm_explicit,
                                grid_m, grid_n, 1, 128, 1, 1,
                                SMEM_BYTES_EXPLICIT, 0, gemm_exp_args, 0));
        CHECK_CU(cuCtxSynchronize());
        CHECK_CU(cuMemcpyDtoH(host_Y_exp, d_Y, Y_elems * sizeof(float)));

        // ---- Correctness: implicit ----
        CHECK_CU(cuMemsetD32(d_Y, 0, Y_elems));
        CHECK_CU(cuLaunchKernel(fn_gemm_implicit,
                                grid_m, grid_n, 1, 128, 1, 1,
                                SMEM_BYTES_IMPLICIT, 0, gemm_imp_args, 0));
        CHECK_CU(cuCtxSynchronize());
        CHECK_CU(cuMemcpyDtoH(host_Y_imp, d_Y, Y_elems * sizeof(float)));

        // Compare implicit vs explicit
        float max_abs_vs_exp = 0.0f;
        for (size_t i = 0; i < Y_elems; i++)
            max_abs_vs_exp = fmaxf(max_abs_vs_exp, fabsf(host_Y_imp[i] - host_Y_exp[i]));
        printf("  vs explicit: %s (max_abs=%.2e)\n",
               max_abs_vs_exp < 1e-4f ? "PASS" : "FAIL", (double)max_abs_vs_exp);

        if (ran_cpu) {
            float max_abs_vs_ref = 0.0f;
            for (size_t i = 0; i < Y_elems; i++)
                max_abs_vs_ref = fmaxf(max_abs_vs_ref, fabsf(host_Y_imp[i] - host_Y_ref[i]));
            printf("  vs CPU ref:  max_abs=%.2e\n", (double)max_abs_vs_ref);
        }

        // Per-launch event chains, created before any timed section so
        // event creation never falls inside a window. Explicit: event 0,
        // then one event after im2col and one after the GEMM per launch.
        // Implicit: event 0, then one event per launch. Default mode
        // creates none of them.
        char config_key[64];
        snprintf(config_key, sizeof(config_key), "%d_%d_%d_%d_%d", N, H, W_dim, Cin, Cout);
        std::unique_ptr<LaunchEventChain> explicit_warmup_chain, explicit_timed_chain;
        std::unique_ptr<LaunchEventChain> implicit_warmup_chain, implicit_timed_chain;
        if (per_launch) {
            explicit_warmup_chain.reset(new LaunchEventChain(1 + 2 * (size_t)warmup_launches));
            explicit_timed_chain.reset(new LaunchEventChain(1 + 2 * (size_t)timed_launches));
            implicit_warmup_chain.reset(new LaunchEventChain(1 + (size_t)warmup_launches));
            implicit_timed_chain.reset(new LaunchEventChain(1 + (size_t)timed_launches));
        }

        // ---- Timing: explicit (im2col + GEMM combined) ----
        float ms_explicit = 0.0f, explicit_window_ms = 0.0f;
        {
            if (per_launch) explicit_warmup_chain->record(0);
            for (int t = 0; t < warmup_launches; t++) {
                CHECK_CU(cuLaunchKernel(fn_im2col, im2col_blocks,1,1, im2col_threads,1,1, 0,0, im2col_args,0));
                if (per_launch) explicit_warmup_chain->record(1 + 2 * (size_t)t);
                CHECK_CU(cuLaunchKernel(fn_gemm_explicit, grid_m, grid_n,1, 128,1,1, SMEM_BYTES_EXPLICIT,0, gemm_exp_args,0));
                if (per_launch) explicit_warmup_chain->record(2 + 2 * (size_t)t);
            }
            CHECK_CU(cuCtxSynchronize());
            BenchTimer bt; bt.start();
            if (per_launch) explicit_timed_chain->record(0);
            for (int t = 0; t < timed_launches; t++) {
                CHECK_CU(cuLaunchKernel(fn_im2col, im2col_blocks,1,1, im2col_threads,1,1, 0,0, im2col_args,0));
                if (per_launch) explicit_timed_chain->record(1 + 2 * (size_t)t);
                CHECK_CU(cuLaunchKernel(fn_gemm_explicit, grid_m, grid_n,1, 128,1,1, SMEM_BYTES_EXPLICIT,0, gemm_exp_args,0));
                if (per_launch) explicit_timed_chain->record(2 + 2 * (size_t)t);
            }
            explicit_window_ms = bt.stop_ms();
            ms_explicit = explicit_window_ms / (float)timed_launches;
        }

        // ---- Timing: implicit GEMM (single kernel) ----
        float ms_implicit = 0.0f, implicit_window_ms = 0.0f;
        {
            if (per_launch) implicit_warmup_chain->record(0);
            for (int t = 0; t < warmup_launches; t++) {
                CHECK_CU(cuLaunchKernel(fn_gemm_implicit, grid_m, grid_n,1, 128,1,1, SMEM_BYTES_IMPLICIT,0, gemm_imp_args,0));
                if (per_launch) implicit_warmup_chain->record(1 + (size_t)t);
            }
            CHECK_CU(cuCtxSynchronize());
            BenchTimer bt; bt.start();
            if (per_launch) implicit_timed_chain->record(0);
            for (int t = 0; t < timed_launches; t++) {
                CHECK_CU(cuLaunchKernel(fn_gemm_implicit, grid_m, grid_n,1, 128,1,1, SMEM_BYTES_IMPLICIT,0, gemm_imp_args,0));
                if (per_launch) implicit_timed_chain->record(1 + (size_t)t);
            }
            implicit_window_ms = bt.stop_ms();
            ms_implicit = implicit_window_ms / (float)timed_launches;
        }

        double gflops_exp = flops_g / (ms_explicit * 1e-3);
        double gflops_imp = flops_g / (ms_implicit * 1e-3);

        printf("  Explicit (im2col+GEMM): %.3f ms  → %6.0f GFLOPS\n", ms_explicit, gflops_exp);
        printf("  Implicit (single kern): %.3f ms  → %6.0f GFLOPS  (%.2f× speedup)\n",
               ms_implicit, gflops_imp, ms_explicit / ms_implicit);

        // Every chain is complete: each phase ended in cuCtxSynchronize()
        // or BenchTimer::stop_ms(), which synchronizes on an event
        // recorded after the chain's last event.
        if (per_launch) {
            struct ExplicitPhase { const char *name; LaunchEventChain *chain; int launches; double aggregate_us; };
            ExplicitPhase explicit_phases[] = {
                { "warmup", explicit_warmup_chain.get(), warmup_launches, -1.0 },
                { "timed",  explicit_timed_chain.get(),  timed_launches,  (double)explicit_window_ms * 1000.0 },
            };
            for (const ExplicitPhase &phase : explicit_phases) {
                std::vector<float> im2col_us, gemm_us, pair_us;
                for (int t = 0; t < phase.launches; t++) {
                    size_t before_im2col = 2 * (size_t)t, after_im2col = before_im2col + 1, after_gemm = before_im2col + 2;
                    im2col_us.push_back(phase.chain->elapsed_us(before_im2col, after_im2col));
                    gemm_us.push_back(phase.chain->elapsed_us(after_im2col, after_gemm));
                    pair_us.push_back(phase.chain->elapsed_us(before_im2col, after_gemm));
                }
                print_per_launch(config_key, "im2col_nhwc_fp16", phase.name, im2col_us, true,  -1.0);
                print_per_launch(config_key, "wmma_gemm_conv",   phase.name, gemm_us,   true,  -1.0);
                print_per_launch(config_key, "explicit_pair",    phase.name, pair_us,   false, phase.aggregate_us);
            }
            struct ImplicitPhase { const char *name; LaunchEventChain *chain; int launches; double aggregate_us; };
            ImplicitPhase implicit_phases[] = {
                { "warmup", implicit_warmup_chain.get(), warmup_launches, -1.0 },
                { "timed",  implicit_timed_chain.get(),  timed_launches,  (double)implicit_window_ms * 1000.0 },
            };
            for (const ImplicitPhase &phase : implicit_phases) {
                std::vector<float> implicit_us;
                for (int t = 0; t < phase.launches; t++)
                    implicit_us.push_back(phase.chain->elapsed_us((size_t)t, (size_t)t + 1));
                print_per_launch(config_key, "implicit_gemm_conv", phase.name, implicit_us, true, phase.aggregate_us);
            }
        }
        printf("\n");

        cuMemFree(d_X); cuMemFree(d_col); cuMemFree(d_Wt); cuMemFree(d_Y);
        delete[] host_X; delete[] host_W; delete[] host_Y_ref;
        delete[] host_Y_exp; delete[] host_Y_imp; delete[] host_Wt;
    }

    cuModuleUnload(mod_explicit);
    cuModuleUnload(mod_implicit);
    cuDevicePrimaryCtxRelease(cu_dev);
    return 0;
}
