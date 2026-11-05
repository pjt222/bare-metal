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
 * Positional arguments must be all digits and are ignored (bench_regress
 * passes the config key as `1 64 64 320 320`); anything else exits 2.
 *
 * --per-launch records a CUevent pair around every launch of the 20 warmup
 * and 100 timed launches, per config, and prints each launch's span and the
 * gap before it, per kernel and phase (see LaunchTimeline). The default
 * path creates no extra events and prints no extra lines, so bench_regress
 * and conv2d_variance.R see the same output as before. See print_series()
 * for the line format.
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <unistd.h>
#include <vector>
#include <memory>
#include <algorithm>
#include <chrono>
#include <cuda.h>
#include <cuda_fp16.h>

#include "../../_common/bench.h"
#include "../../_common/check.h"

// -----------------------------------------------------------------------
// Per-launch timing (#223 step 1). Off unless --per-launch is given.
//
// A pair of events per launch in the default stream, as #223 step 1
// specifies: start[j] is recorded right before launch j and stop[j] right
// after it, and each phase records an origin event before its first
// launch. Two readings per launch, both spans on the GPU timeline:
//
//   span[j] = stop[j] - start[j]. The kernel's execution, plus any time
//             inside that window when the GPU was not running it: another
//             context's time slice, a GPU-wide stall such as a clock
//             switch, or idling because the launch command itself had not
//             arrived. start[j] is stamped as soon as it reaches the GPU,
//             so if the queue had drained (the host was late in
//             cuLaunchKernel, or a submission batch ended between the two
//             commands), the wait for the launch lands in span[j], not in
//             gap[j]. Only a host-side timestamp separates that case, and
//             step 1 records none. Clock telemetry (#223 step 5) is
//             needed to tell slow clocks from preemption.
//   gap[j]  = start[j] - stop[j-1] (j = 0: start[0] - origin). About zero
//             while the host keeps the queue fed; larger when the GPU
//             waited for the start event itself, or stalled between
//             commands. A small gap does not rule out a host wait: that
//             wait can sit in the span instead.
//
// Launch 0 of every phase follows an idle GPU, because each phase starts
// after a synchronization. Its post-idle latency is split three ways:
// - before the origin (in timed phases, between BenchTimer's start event
//   and the origin);
// - in gap[0] (origin to start[0], two back-to-back event commands);
// - in span[0], which by construction takes the delay of the launch
//   command, the last of the three to reach the GPU.
// The instrument cannot tell that latency from a slow first launch after
// idle.
//
// gap[j] + span[j] is what one event between consecutive launches would
// report. On the explicit path the launches alternate im2col, GEMM,
// im2col, ..., so a sustained change can be localized to the im2col window
// or the GEMM window. An isolated spike cannot be attributed to the kernel
// whose window it lands in, and neither can memory traffic that crosses
// the kernel boundary (dirty col-buffer lines written back from L2 during
// the GEMM).
// -----------------------------------------------------------------------
struct LaunchTimeline {
    CUevent origin = nullptr;
    std::vector<CUevent> start_events, stop_events;

    explicit LaunchTimeline(size_t launch_count)
        : start_events(launch_count), stop_events(launch_count) {
        CHECK_CU(cuEventCreate(&origin, CU_EVENT_DEFAULT));
        for (size_t j = 0; j < launch_count; j++) {
            CHECK_CU(cuEventCreate(&start_events[j], CU_EVENT_DEFAULT));
            CHECK_CU(cuEventCreate(&stop_events[j],  CU_EVENT_DEFAULT));
        }
    }
    ~LaunchTimeline() {
        cuEventDestroy(origin);
        for (CUevent event : start_events) cuEventDestroy(event);
        for (CUevent event : stop_events)  cuEventDestroy(event);
    }
    LaunchTimeline(const LaunchTimeline &) = delete;
    LaunchTimeline &operator=(const LaunchTimeline &) = delete;

    size_t launch_count() const { return start_events.size(); }
    void record_origin()        { CHECK_CU(cuEventRecord(origin, NULL)); }
    void record_start(size_t j) { CHECK_CU(cuEventRecord(start_events[j], NULL)); }
    void record_stop(size_t j)  { CHECK_CU(cuEventRecord(stop_events[j],  NULL)); }

    // Callers must have synchronized past the phase's last event.
    static float elapsed_us(CUevent from_event, CUevent to_event) {
        float elapsed_ms = 0.0f;
        CHECK_CU(cuEventElapsedTime(&elapsed_ms, from_event, to_event));
        return elapsed_ms * 1000.0f;
    }
    float span_us(size_t j) const { return elapsed_us(start_events[j], stop_events[j]); }
    float gap_us(size_t j) const  { return elapsed_us(j == 0 ? origin : stop_events[j - 1], start_events[j]); }
    float total_us() const        { return elapsed_us(origin, stop_events.back()); }
};

// No-ops when timeline is null, which is the default mode.
static inline void mark_start(LaunchTimeline *timeline, size_t j) { if (timeline) timeline->record_start(j); }
static inline void mark_stop(LaunchTimeline *timeline, size_t j)  { if (timeline) timeline->record_stop(j); }

// One event with RAII lifetime: the per-config time origin.
struct ScopedEvent {
    CUevent event = nullptr;
    ScopedEvent()  { CHECK_CU(cuEventCreate(&event, CU_EVENT_DEFAULT)); }
    ~ScopedEvent() { cuEventDestroy(event); }
    ScopedEvent(const ScopedEvent &) = delete;
    ScopedEvent &operator=(const ScopedEvent &) = delete;
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

// Median (nearest-rank) of values[begin, end); NAN for an empty range.
static double median_of_range(const std::vector<float> &values, size_t begin, size_t end) {
    if (end <= begin) return NAN;
    std::vector<float> window(values.begin() + begin, values.begin() + end);
    std::sort(window.begin(), window.end());
    return (double)nearest_rank(window, 0.50);
}

static double sum_of(const std::vector<float> &values) {
    double total = 0.0;
    for (float value : values) total += value;
    return total;
}

// Output lines, all `key=value` fields:
//
//   per-launch config=<N_H_W_Cin_Cout> kernel=<name> phase=<warmup|timed> metric=<span|gap> n=<count> us=<v1,v2,...>
//   per-launch-summary config=... kernel=... phase=... metric=... n=... min_us=... p10_us=... p50_us=...
//       p90_us=... max_us=... idx0_us=... head5_p50_us=... tail5_p50_us=... sum_us=...
//   per-launch-config config=... cpu_ref=<ran|skipped> cpu_ref_wall_us=...
//   per-launch-phase config=... path=<explicit|implicit> phase=... launches=<count> offset_us=... total_us=... [aggregate_us=...]
//
// One raw and one summary line per (kernel, phase, metric), except
// explicit_pair (start of im2col to end of the GEMM, per iteration), which
// gets a summary only. Values are microseconds in launch order.
//
// Percentiles are nearest-rank. idx0_us is the series' first value on its
// own, because launch 0 follows an idle GPU and both its gap and its span
// carry post-idle latency (see LaunchTimeline). For wmma_gemm_conv the
// first value is the phase's second launch (j = 1), which is kept apart
// only for uniformity. head5_p50_us is the median of indices 1..5 and
// tail5_p50_us the median of the last five, so a ramp has to show in
// several launches: one stall cannot move a median of five. Event timestamps step in about
// 1.02 us on this machine (most values sit on a 1.024 us grid). The %.2f
// format prints two decimals, but differences below about 1 us are not
// readable, which matters for the Small config's 60-140 us kernels.
//
// cpu_ref_wall_us is the host wall time of the CPU reference, the idle the
// ramp candidate is about (#223 steps 2-3). offset_us puts every phase of a
// config on one axis: GPU time from the config's origin, an event recorded
// as the CPU reference returns, to the phase's own origin. Read a ramp
// against offset_us + the launch's position, not against launch index.
// The first GPU work after the idle is NOT instrumented:
// - the device allocations and host-to-device copies;
// - one run of each kernel for the correctness check, each followed by a
//   synchronization and a device-to-host copy;
// - a host-only comparison, during which the GPU idles a second time.
// All of it comes before the explicit warmup's origin, so the window's
// length is that phase's offset_us. A ramp that starts and ends inside it
// is invisible here.
//
// total_us runs from the phase's origin to its last stop event, so it is
// the sum of all its spans and gaps. aggregate_us, on timed phases only,
// is the BenchTimer window the default output reports (as ms per launch).
// The two differ only by the edges outside the timeline, which makes them
// a consistency check, NOT a measure of what the extra events cost: that
// cost sits inside the spans and gaps. Whether --per-launch perturbs the
// timing can be read only by comparing the default-mode aggregate with the
// --per-launch aggregate across runs.
//
// The lines avoid `ms`/`GFLOPS` and never start with `===`/`---`, and
// neither parser match string ("Explicit (im2col+GEMM)", "Implicit
// (single kern)") appears in them, so parse_throughput() cannot select
// them. R/cuasmR/tests/testthat/test-parse_throughput.R pins that against
// a CAPTURED file, fixtures/conv2d_implicit_per_launch.txt, not against
// this bench: after changing any of these lines, recapture it
// (`bench_implicit_gemm --per-launch > <fixture>`) and rerun the test.
static void print_series(const char *config_key, const char *kernel_name,
                         const char *phase_name, const char *metric_name,
                         const std::vector<float> &values_us, bool print_raw) {
    size_t sample_count = values_us.size();
    if (sample_count == 0) return;
    if (print_raw) {
        printf("  per-launch config=%s kernel=%s phase=%s metric=%s n=%zu us=",
               config_key, kernel_name, phase_name, metric_name, sample_count);
        for (size_t index = 0; index < sample_count; index++)
            printf(index == 0 ? "%.2f" : ",%.2f", (double)values_us[index]);
        printf("\n");
    }
    std::vector<float> sorted_values(values_us);
    std::sort(sorted_values.begin(), sorted_values.end());
    size_t head_end   = sample_count < 6 ? sample_count : 6;               // indices 1..5
    size_t tail_begin = sample_count > 5 ? sample_count - 5 : 0;           // last five
    printf("  per-launch-summary config=%s kernel=%s phase=%s metric=%s n=%zu"
           " min_us=%.2f p10_us=%.2f p50_us=%.2f p90_us=%.2f max_us=%.2f"
           " idx0_us=%.2f head5_p50_us=%.2f tail5_p50_us=%.2f sum_us=%.2f\n",
           config_key, kernel_name, phase_name, metric_name, sample_count,
           (double)sorted_values.front(),
           (double)nearest_rank(sorted_values, 0.10),
           (double)nearest_rank(sorted_values, 0.50),
           (double)nearest_rank(sorted_values, 0.90),
           (double)sorted_values.back(),
           (double)values_us[0],
           median_of_range(values_us, 1, head_end),
           median_of_range(values_us, tail_begin, sample_count),
           sum_of(values_us));
}

static void print_phase_total(const char *config_key, const char *path_name,
                              const char *phase_name, const LaunchTimeline &timeline,
                              CUevent config_origin, double aggregate_us) {
    printf("  per-launch-phase config=%s path=%s phase=%s launches=%zu offset_us=%.2f total_us=%.2f",
           config_key, path_name, phase_name, timeline.launch_count(),
           (double)LaunchTimeline::elapsed_us(config_origin, timeline.origin),
           (double)timeline.total_us());
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
    fprintf(stderr, "usage: %s [--per-launch] [ignored numeric positional args]\n", program_name);
}

static bool is_all_digits(const char *text) {
    if (*text == '\0') return false;
    for (; *text; text++)
        if (*text < '0' || *text > '9') return false;
    return true;
}

int main(int argc, char **argv) {
    // Numeric positional arguments are accepted and ignored, as they always
    // were: bench_regress.R passes the data/baselines.json config key split
    // on '_' (`1 64 64 320 320`), and this bench runs every config
    // regardless. Its other runners pass none: bench_all.R (bench_all.yml
    // `args: []`), grid_measure.R (grid_sweep.yml `args: []`) and
    // conv2d_variance.R. Every other argument exits 2: an unknown option,
    // and also a flag that lost its dashes (`per-launch`) or was pasted with
    // a Unicode dash, which would otherwise run the default mode silently
    // in a campaign that meant to record per-launch.
    bool per_launch = false;
    for (int arg_index = 1; arg_index < argc; arg_index++) {
        const char *arg = argv[arg_index];
        if (strcmp(arg, "--per-launch") == 0) {
            per_launch = true;
        } else if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (!is_all_digits(arg)) {
            fprintf(stderr, "unknown argument: %s\n", arg);
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

        // Per-launch timelines and the config's time origin, created before
        // the CPU reference so that event creation falls outside the window
        // after the idle. Default mode creates none, and every mark_* call
        // below is then a no-op.
        char config_key[64];
        snprintf(config_key, sizeof(config_key), "%d_%d_%d_%d_%d", N, H, W_dim, Cin, Cout);
        std::unique_ptr<LaunchTimeline> explicit_warmup_timeline, explicit_timed_timeline;
        std::unique_ptr<LaunchTimeline> implicit_warmup_timeline, implicit_timed_timeline;
        std::unique_ptr<ScopedEvent> config_origin;
        if (per_launch) {
            explicit_warmup_timeline.reset(new LaunchTimeline(2 * (size_t)warmup_launches));
            explicit_timed_timeline.reset(new LaunchTimeline(2 * (size_t)timed_launches));
            implicit_warmup_timeline.reset(new LaunchTimeline((size_t)warmup_launches));
            implicit_timed_timeline.reset(new LaunchTimeline((size_t)timed_launches));
            config_origin.reset(new ScopedEvent());
        }

        // CPU reference (skip for large configs)
        bool ran_cpu = (X_elems < 16 * 1024 * 1024 / 4);  // skip if X > 16 MB
        std::chrono::steady_clock::time_point cpu_ref_begin;
        if (per_launch) cpu_ref_begin = std::chrono::steady_clock::now();
        if (ran_cpu) {
            cpu_conv2d_nhwc_ref(host_X, host_W, host_Y_ref, N, H, W_dim, Cin, Cout);
        }
        double cpu_ref_wall_us = 0.0;
        if (per_launch) {
            cpu_ref_wall_us = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - cpu_ref_begin).count();
            // The idle #223's ramp candidate is about ends here; every
            // phase's offset_us is measured from this event.
            CHECK_CU(cuEventRecord(config_origin->event, NULL));
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

        // ---- Timing: explicit (im2col + GEMM combined) ----
        // Launch j = 2t is im2col, j = 2t + 1 is the GEMM.
        float ms_explicit = 0.0f, explicit_window_ms = 0.0f;
        {
            LaunchTimeline *timeline = explicit_warmup_timeline.get();
            if (timeline) timeline->record_origin();
            for (int t = 0; t < warmup_launches; t++) {
                mark_start(timeline, 2 * (size_t)t);
                CHECK_CU(cuLaunchKernel(fn_im2col, im2col_blocks,1,1, im2col_threads,1,1, 0,0, im2col_args,0));
                mark_stop(timeline, 2 * (size_t)t);
                mark_start(timeline, 2 * (size_t)t + 1);
                CHECK_CU(cuLaunchKernel(fn_gemm_explicit, grid_m, grid_n,1, 128,1,1, SMEM_BYTES_EXPLICIT,0, gemm_exp_args,0));
                mark_stop(timeline, 2 * (size_t)t + 1);
            }
            CHECK_CU(cuCtxSynchronize());
            BenchTimer bt; bt.start();
            timeline = explicit_timed_timeline.get();
            if (timeline) timeline->record_origin();
            for (int t = 0; t < timed_launches; t++) {
                mark_start(timeline, 2 * (size_t)t);
                CHECK_CU(cuLaunchKernel(fn_im2col, im2col_blocks,1,1, im2col_threads,1,1, 0,0, im2col_args,0));
                mark_stop(timeline, 2 * (size_t)t);
                mark_start(timeline, 2 * (size_t)t + 1);
                CHECK_CU(cuLaunchKernel(fn_gemm_explicit, grid_m, grid_n,1, 128,1,1, SMEM_BYTES_EXPLICIT,0, gemm_exp_args,0));
                mark_stop(timeline, 2 * (size_t)t + 1);
            }
            explicit_window_ms = bt.stop_ms();
            ms_explicit = explicit_window_ms / (float)timed_launches;
        }

        // ---- Timing: implicit GEMM (single kernel) ----
        float ms_implicit = 0.0f, implicit_window_ms = 0.0f;
        {
            LaunchTimeline *timeline = implicit_warmup_timeline.get();
            if (timeline) timeline->record_origin();
            for (int t = 0; t < warmup_launches; t++) {
                mark_start(timeline, (size_t)t);
                CHECK_CU(cuLaunchKernel(fn_gemm_implicit, grid_m, grid_n,1, 128,1,1, SMEM_BYTES_IMPLICIT,0, gemm_imp_args,0));
                mark_stop(timeline, (size_t)t);
            }
            CHECK_CU(cuCtxSynchronize());
            BenchTimer bt; bt.start();
            timeline = implicit_timed_timeline.get();
            if (timeline) timeline->record_origin();
            for (int t = 0; t < timed_launches; t++) {
                mark_start(timeline, (size_t)t);
                CHECK_CU(cuLaunchKernel(fn_gemm_implicit, grid_m, grid_n,1, 128,1,1, SMEM_BYTES_IMPLICIT,0, gemm_imp_args,0));
                mark_stop(timeline, (size_t)t);
            }
            implicit_window_ms = bt.stop_ms();
            ms_implicit = implicit_window_ms / (float)timed_launches;
        }

        double gflops_exp = flops_g / (ms_explicit * 1e-3);
        double gflops_imp = flops_g / (ms_implicit * 1e-3);

        printf("  Explicit (im2col+GEMM): %.3f ms  → %6.0f GFLOPS\n", ms_explicit, gflops_exp);
        printf("  Implicit (single kern): %.3f ms  → %6.0f GFLOPS  (%.2f× speedup)\n",
               ms_implicit, gflops_imp, ms_explicit / ms_implicit);

        // Every timeline is complete: each phase ended in cuCtxSynchronize()
        // or BenchTimer::stop_ms(), which synchronizes on an event recorded
        // after the phase's last stop event.
        if (per_launch) {
            printf("  per-launch-config config=%s cpu_ref=%s cpu_ref_wall_us=%.2f\n",
                   config_key, ran_cpu ? "ran" : "skipped", cpu_ref_wall_us);
            struct Phase { const char *name; LaunchTimeline *timeline; double aggregate_us; };
            Phase explicit_phases[] = {
                { "warmup", explicit_warmup_timeline.get(), -1.0 },
                { "timed",  explicit_timed_timeline.get(),  (double)explicit_window_ms * 1000.0 },
            };
            for (const Phase &phase : explicit_phases) {
                const LaunchTimeline &timeline = *phase.timeline;
                size_t iterations = timeline.launch_count() / 2;
                std::vector<float> im2col_span, im2col_gap, gemm_span, gemm_gap, pair_span;
                for (size_t t = 0; t < iterations; t++) {
                    size_t im2col_j = 2 * t, gemm_j = 2 * t + 1;
                    im2col_span.push_back(timeline.span_us(im2col_j));
                    im2col_gap.push_back(timeline.gap_us(im2col_j));
                    gemm_span.push_back(timeline.span_us(gemm_j));
                    gemm_gap.push_back(timeline.gap_us(gemm_j));
                    pair_span.push_back(LaunchTimeline::elapsed_us(timeline.start_events[im2col_j],
                                                                   timeline.stop_events[gemm_j]));
                }
                print_series(config_key, "im2col_nhwc_fp16", phase.name, "span", im2col_span, true);
                print_series(config_key, "im2col_nhwc_fp16", phase.name, "gap",  im2col_gap,  true);
                print_series(config_key, "wmma_gemm_conv",   phase.name, "span", gemm_span,   true);
                print_series(config_key, "wmma_gemm_conv",   phase.name, "gap",  gemm_gap,    true);
                print_series(config_key, "explicit_pair",    phase.name, "span", pair_span,   false);
                print_phase_total(config_key, "explicit", phase.name, timeline, config_origin->event, phase.aggregate_us);
            }
            Phase implicit_phases[] = {
                { "warmup", implicit_warmup_timeline.get(), -1.0 },
                { "timed",  implicit_timed_timeline.get(),  (double)implicit_window_ms * 1000.0 },
            };
            for (const Phase &phase : implicit_phases) {
                const LaunchTimeline &timeline = *phase.timeline;
                std::vector<float> implicit_span, implicit_gap;
                for (size_t j = 0; j < timeline.launch_count(); j++) {
                    implicit_span.push_back(timeline.span_us(j));
                    implicit_gap.push_back(timeline.gap_us(j));
                }
                print_series(config_key, "implicit_gemm_conv", phase.name, "span", implicit_span, true);
                print_series(config_key, "implicit_gemm_conv", phase.name, "gap",  implicit_gap,  true);
                print_phase_total(config_key, "implicit", phase.name, timeline, config_origin->event, phase.aggregate_us);
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
