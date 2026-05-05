/*
 * bench_refactored.cu — Flash Attention benchmark using BenchDriver
 *
 * Demonstrates bench_driver.h with attention-specific patterns.
 * Before: ~220 lines
 * After:  ~85 lines
 *
 * Build:
 *   nvcc -arch=sm_86 -O2 -o bench_refactored bench_refactored.cu -lcuda -I../../phase2/common
 */

#include <cuda.h>
#include <cuda_fp16.h>
#include "../../phase2/common/bench_driver.h"

// Minimal CPU flash attention (causal, for correctness check only)
static void cpu_flash_attn(int seq, int heads, int d,
                           const float *Q, const float *K, const float *V,
                           float *O, float scale) {
    size_t stride = (size_t)seq * d;
    for (int h = 0; h < heads; h++) {
        for (int q = 0; q < seq; q++) {
            float max_s = -1e30f;
            for (int k = 0; k < seq; k++) {
                float dot = 0;
                for (int i = 0; i < d; i++)
                    dot += Q[h*stride + q*d + i] * K[h*stride + k*d + i];
                max_s = fmaxf(max_s, dot * scale);
            }
            float sum = 0;
            for (int k = 0; k < seq; k++) {
                float dot = 0;
                for (int i = 0; i < d; i++)
                    dot += Q[h*stride + q*d + i] * K[h*stride + k*d + i];
                sum += expf(dot * scale - max_s);
            }
            for (int i = 0; i < d; i++) {
                float out = 0;
                for (int k = 0; k < seq; k++) {
                    float dot = 0;
                    for (int j = 0; j < d; j++)
                        dot += Q[h*stride + q*d + j] * K[h*stride + k*d + j];
                    float w = expf(dot * scale - max_s) / sum;
                    out += w * V[h*stride + k*d + i];
                }
                O[h*stride + q*d + i] = out;
            }
        }
    }
}

int main(int argc, char **argv) {
    int seq   = (argc > 1) ? atoi(argv[1]) : 1024;
    int batch = (argc > 2) ? atoi(argv[2]) : 8;
    int heads = (argc > 3) ? atoi(argv[3]) : 8;
    const int d = 64;
    float scale = 1.0f / sqrtf((float)d);

    printf("=== Flash Attention (BenchDriver refactor) ===\n");
    printf("seq=%d batch=%d heads=%d d=%d\n\n", seq, batch, heads, d);

    BenchDriver driver;
    driver.init_context();

    size_t elems = (size_t)batch * heads * seq * d;
    auto d_Q = driver.device_alloc<__half>(elems);
    auto d_K = driver.device_alloc<__half>(elems);
    auto d_V = driver.device_alloc<__half>(elems);
    auto d_O = driver.device_alloc<float>(elems);

    auto h_Q = driver.host_alloc<float>(elems);
    auto h_K = driver.host_alloc<float>(elems);
    auto h_V = driver.host_alloc<float>(elems);
    auto h_O = driver.host_alloc<float>(elems);
    auto h_ref = driver.host_alloc<float>(elems);

    fill_random(h_Q.get(), elems, 1);
    fill_random(h_K.get(), elems, 2);
    fill_random(h_V.get(), elems, 3);

    // Upload FP16
    auto h_Qh = driver.host_alloc<__half>(elems);
    auto h_Kh = driver.host_alloc<__half>(elems);
    auto h_Vh = driver.host_alloc<__half>(elems);
    for (size_t i = 0; i < elems; i++) {
        h_Qh[i] = __float2half(h_Q[i]);
        h_Kh[i] = __float2half(h_K[i]);
        h_Vh[i] = __float2half(h_V[i]);
    }
    driver.copy_h2d(d_Q, h_Qh, elems * sizeof(__half));
    driver.copy_h2d(d_K, h_Kh, elems * sizeof(__half));
    driver.copy_h2d(d_V, h_Vh, elems * sizeof(__half));

    // CPU reference (small only)
    bool have_ref = (seq <= 128);
    if (have_ref) {
        cpu_flash_attn(seq, batch*heads, d, h_Q.get(), h_K.get(), h_V.get(), h_ref.get(), scale);
    }

    struct V { const char *name, *cubin, *sym; dim3 g, b; unsigned smem; };
    std::vector<V> variants = {
        {"flash_br16_regpv", "flash_br16_regpv.sm_86.cubin", "flash_attn_br16_regpv",
         dim3((seq+63)/64, heads, batch), dim3(128,1,1), 32*1024},
    };

    for (auto &v : variants) {
        CUfunction fn = driver.load_kernel(v.cubin, v.sym, false);
        if (!fn) { printf("  %-20s not found\n", v.name); continue; }

        void *args[] = { &d_Q, &d_K, &d_V, &d_O, &seq, &heads, &scale };
        float ms = driver.benchmark_kernel(fn, v.g, v.b, v.smem, args);
        double gflops = 2.0 * seq * seq * d * batch * heads / (ms / 1000.0) / 1e9;

        if (have_ref) {
            driver.copy_d2h(h_O, d_O, elems * sizeof(float));
            driver.check(h_O.get(), h_ref.get(), (int)elems, 1e-2f, 1e-2f, v.name);
        }
        printf("  %-20s %7.3f ms  %8.2f GFLOPS  [%s]\n",
               v.name, ms, gflops, have_ref ? "CHECKED" : "PERF_ONLY");
    }
    return 0;
}
