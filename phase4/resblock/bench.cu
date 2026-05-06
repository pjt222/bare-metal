/*
 * bench.cu — ResNet Block benchmark (BenchDriver refactor)
 *
 * Chains: GroupNorm+SiLU -> Conv2d(3x3) -> GroupNorm+SiLU -> Conv2d(3x3) -> +x
 *
 * Build:
 *   nvcc -arch=sm_86 -O2 -o bench bench.cu -lcuda -I../../phase2/common
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda.h>

#include "../../phase2/common/bench_driver.h"

static float cpu_silu(float x) { return x / (1.0f + expf(-x)); }

static void cpu_groupnorm_silu(
    const float *X, const float *g, const float *b,
    float *Y, int N, int C, int H, int W, int G, float eps
) {
    int cpg = C / G, spatial = H * W;
    for (int n = 0; n < N; n++) {
        for (int gi = 0; gi < G; gi++) {
            int cb = gi * cpg, gsize = cpg * spatial;
            double sum = 0.0;
            for (int s = 0; s < spatial; s++)
                for (int lc = 0; lc < cpg; lc++)
                    sum += X[(size_t)n * spatial * C + s * C + cb + lc];
            float mean = (float)(sum / gsize);
            double var = 0.0;
            for (int s = 0; s < spatial; s++)
                for (int lc = 0; lc < cpg; lc++) {
                    float d = X[(size_t)n * spatial * C + s * C + cb + lc] - mean;
                    var += d * d;
                }
            float inv_std = 1.0f / sqrtf((float)(var / gsize) + eps);
            for (int s = 0; s < spatial; s++)
                for (int lc = 0; lc < cpg; lc++) {
                    int gc = cb + lc;
                    size_t flat = (size_t)n * spatial * C + s * C + gc;
                    Y[flat] = cpu_silu(g[gc] * ((X[flat] - mean) * inv_std) + b[gc]);
                }
        }
    }
}

static void cpu_conv2d_nhwc(
    const float *X, const float *W, const float *bias, float *Y,
    int N, int H, int W_dim, int Cin, int Cout
) {
    for (int n = 0; n < N; n++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W_dim; w++)
                for (int co = 0; co < Cout; co++) {
                    double acc = bias ? (double)bias[co] : 0.0;
                    for (int kh = 0; kh < 3; kh++)
                        for (int kw = 0; kw < 3; kw++) {
                            int hi = h + kh - 1, wi = w + kw - 1;
                            if (hi < 0 || hi >= H || wi < 0 || wi >= W_dim) continue;
                            for (int ci = 0; ci < Cin; ci++) {
                                acc += (double)X[(size_t)n * H * W_dim * Cin + hi * W_dim * Cin + wi * Cin + ci]
                                     * (double)W[(size_t)co * 9 * Cin + (kh * 3 + kw) * Cin + ci];
                            }
                        }
                    Y[(size_t)n * H * W_dim * Cout + h * W_dim * Cout + w * Cout + co] = (float)acc;
                }
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1;
    int C = (argc > 2) ? atoi(argv[2]) : 64;
    int H = (argc > 3) ? atoi(argv[3]) : 32;
    int W = (argc > 4) ? atoi(argv[4]) : 32;
    int G = (argc > 5) ? atoi(argv[5]) : 8;
    float eps = 1e-5f;
    size_t elems = (size_t)N * H * W * C;
    size_t w3 = (size_t)C * 9 * C;

    if (C % G != 0) { fprintf(stderr, "C must be divisible by G\n"); return 1; }
    if ((C / G * H * W) % 32 != 0) { fprintf(stderr, "group_size must be divisible by 32\n"); return 1; }

    printf("=== ResNet Block Benchmark (BenchDriver refactor) ===\n");
    printf("N=%d C=%d H=%d W=%d G=%d\n", N, C, H, W, G);
    printf("Total FLOPs: %.3f GFLOPS\n\n", 2.0 * 2 * N * H * W * C * C * 9 / 1e9);

    BenchDriver driver;
    driver.init_context();

    auto h_X = driver.host_alloc<float>(elems);
    auto h_g = driver.host_alloc<float>(C);
    auto h_b = driver.host_alloc<float>(C);
    auto h_W1 = driver.host_alloc<float>(w3);
    auto h_W2 = driver.host_alloc<float>(w3);
    auto h_ref = driver.host_alloc<float>(elems);

    fill_random(h_X.get(), elems, 1);
    fill_random(h_g.get(), C, 2);
    fill_random(h_b.get(), C, 3);
    fill_random(h_W1.get(), w3, 4);
    fill_random(h_W2.get(), w3, 5);
    for (int i = 0; i < C; i++) h_g[i] = h_g[i] * 0.5f + 1.0f;
    for (size_t i = 0; i < w3; i++) { h_W1[i] *= 0.05f; h_W2[i] *= 0.05f; }

    // Device buffers
    auto d_X   = driver.device_alloc<float>(elems);
    auto d_g   = driver.device_alloc<float>(C);
    auto d_b   = driver.device_alloc<float>(C);
    auto d_W1  = driver.device_alloc<float>(w3);
    auto d_W2  = driver.device_alloc<float>(w3);
    auto d_t1  = driver.device_alloc<float>(elems);  // tmp1
    auto d_t2  = driver.device_alloc<float>(elems);  // tmp2
    auto d_Y   = driver.device_alloc<float>(elems);

    driver.copy_h2d(d_X, h_X, elems * sizeof(float));
    driver.copy_h2d(d_g, h_g, C * sizeof(float));
    driver.copy_h2d(d_b, h_b, C * sizeof(float));
    driver.copy_h2d(d_W1, h_W1, w3 * sizeof(float));
    driver.copy_h2d(d_W2, h_W2, w3 * sizeof(float));

    // Load kernels from two cubins
    CUmodule mod_rb, mod_conv;
    CHECK_CU(cuModuleLoad(&mod_rb, "resblock.sm_86.cubin"));
    CHECK_CU(cuModuleLoad(&mod_conv, "../conv2d/conv2d.sm_86.cubin"));
    CUfunction fn_gn_silu, fn_residual, fn_conv;
    CHECK_CU(cuModuleGetFunction(&fn_gn_silu, mod_rb, "groupnorm_silu_fused"));
    CHECK_CU(cuModuleGetFunction(&fn_residual, mod_rb, "residual_add"));
    CHECK_CU(cuModuleGetFunction(&fn_conv, mod_conv, "conv2d_nhwc"));

    int gn_grid  = N * G;
    int hw_tiles = (H * W + 15) / 16;
    int c_tiles  = (C + 7) / 8;
    int res_grid = ((int)elems + 255) / 256;
    CUdeviceptr dev_bias_null = 0;

    auto run_block = [&]() {
        void *a1[] = { &d_X.ptr, &d_g.ptr, &d_b.ptr, &d_t1.ptr, &N, &C, &H, &W, &G, &eps };
        CHECK_CU(cuLaunchKernel(fn_gn_silu, gn_grid, 1, 1, 32, 1, 1, 0, nullptr, a1, nullptr));
        void *c1[] = { &d_t1.ptr, &d_W1.ptr, &dev_bias_null, &d_t2.ptr,
                       &N, &H, &W, &C, &C };
        CHECK_CU(cuLaunchKernel(fn_conv, N, hw_tiles, c_tiles, 16, 8, 1, 0, nullptr, c1, nullptr));
        void *a2[] = { &d_t2.ptr, &d_g.ptr, &d_b.ptr, &d_t1.ptr, &N, &C, &H, &W, &G, &eps };
        CHECK_CU(cuLaunchKernel(fn_gn_silu, gn_grid, 1, 1, 32, 1, 1, 0, nullptr, a2, nullptr));
        void *c2[] = { &d_t1.ptr, &d_W2.ptr, &dev_bias_null, &d_t2.ptr,
                       &N, &H, &W, &C, &C };
        CHECK_CU(cuLaunchKernel(fn_conv, N, hw_tiles, c_tiles, 16, 8, 1, 0, nullptr, c2, nullptr));
        int total = (int)elems;
        void *r[] = { &d_t2.ptr, &d_X.ptr, &d_Y.ptr, &total };
        CHECK_CU(cuLaunchKernel(fn_residual, res_grid, 1, 1, 256, 1, 1, 0, nullptr, r, nullptr));
    };

    // =====================================================================
    // Correctness
    // =====================================================================
    printf("Correctness:\n");
    bool have_ref = ((double)N * H * W * C * C * 9 <= 5e8);
    if (have_ref) {
        auto h_tmp1 = driver.host_alloc<float>(elems);
        auto h_tmp2 = driver.host_alloc<float>(elems);
        cpu_groupnorm_silu(h_X.get(), h_g.get(), h_b.get(), h_tmp1.get(), N, C, H, W, G, eps);
        cpu_conv2d_nhwc(h_tmp1.get(), h_W1.get(), nullptr, h_tmp2.get(), N, H, W, C, C);
        cpu_groupnorm_silu(h_tmp2.get(), h_g.get(), h_b.get(), h_tmp1.get(), N, C, H, W, G, eps);
        cpu_conv2d_nhwc(h_tmp1.get(), h_W2.get(), nullptr, h_tmp2.get(), N, H, W, C, C);
        for (size_t i = 0; i < elems; i++) h_ref[i] = h_tmp2[i] + h_X[i];
    } else {
        printf("  [CPU ref skipped — config too large]\n");
    }

    run_block();
    CHECK_CU(cuCtxSynchronize());

    if (have_ref) {
        auto h_out = driver.host_alloc<float>(elems);
        driver.copy_d2h(h_out, d_Y, elems * sizeof(float));
        float abs_tol = 1e-2f * (float)sqrtf((float)C);
        driver.check(h_out.get(), h_ref.get(), (int)elems, abs_tol, 0.1f,
                     "ResNet Block (GN+SiLU x2 + Conv2d x2 + add)");
    }

    // =====================================================================
    // Performance
    // =====================================================================
    printf("\nPerformance:\n");
    for (int i = 0; i < 5; i++) run_block();
    CHECK_CU(cuCtxSynchronize());

    BenchTimer timer;
    timer.start();
    int bench_iters = 100;
    for (int i = 0; i < bench_iters; i++) run_block();
    CHECK_CU(cuCtxSynchronize());
    float avg_ms = timer.stop_ms() / bench_iters;

    double total_flops = 2.0 * 2.0 * N * H * W * C * C * 9;
    double gflops = total_flops / 1e9 / (avg_ms / 1000.0);
    double bytes = (double)elems * sizeof(float) * 3 * 5;
    double gb_s = bytes / 1e9 / (avg_ms / 1000.0);
    printf("  %-45s %7.3f ms  %7.1f GFLOPS  (~%.0f GB/s BW)\n",
           "Full ResNet Block (5 kernels)", avg_ms, gflops, gb_s);

    cuModuleUnload(mod_rb);
    cuModuleUnload(mod_conv);
    return 0;
}
