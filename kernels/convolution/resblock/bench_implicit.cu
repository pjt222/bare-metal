/*
 * bench_implicit.cu — ResNet Block benchmark using implicit_gemm_conv
 *
 * Variant of bench.cu: replaces conv2d_nhwc (236 GFLOPS) with
 * implicit_gemm_conv (4800-6800 GFLOPS, Tensor Cores). Issue #83.
 *
 * Build:
 *   nvcc -arch=sm_86 -O2 -o bench_implicit bench_implicit.cu \
 *        -lcuda -I../../kernels/_common
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda.h>
#include <cuda_fp16.h>

#include "../../_common/bench_driver.h"

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

// FP32 [Cout, kH, kW, Cin] → FP16 [Cin*kH*kW, Cout] reshape
static void reshape_weights_to_col(
    const float *W_direct, __half *W_t,
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
            W_t[(size_t)k * Cout + cout_c] = __float2half(val);
        }
    }
}

// Implicit GEMM smem layout from conv2d_implicit_gemm.cu:
//   BLOCK_M=64, BLOCK_N=64, BLOCK_K=16
//   smem_A: 64*24 halfs = 3072 B
//   smem_B: 16*72 halfs = 2304 B
//   coord tables: (3*64 + 3*16) floats * 4 = 960 B
//   total: ~6.3 KB
#define BLOCK_M_IGEMM 64
#define BLOCK_N_IGEMM 64
#define BLOCK_K_IGEMM 16
#define SMEM_A_STRIDE_IGEMM (BLOCK_K_IGEMM + 8)
#define SMEM_B_STRIDE_IGEMM (BLOCK_N_IGEMM + 8)
#define SMEM_BYTES_IGEMM (((BLOCK_M_IGEMM * SMEM_A_STRIDE_IGEMM + BLOCK_K_IGEMM * SMEM_B_STRIDE_IGEMM) * 2) + (3 * BLOCK_M_IGEMM + 3 * BLOCK_K_IGEMM) * 4)

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
    if (C % BLOCK_N_IGEMM != 0 && BLOCK_N_IGEMM % C != 0) {
        // implicit_gemm grid expects Cout multiple of BLOCK_N or padded; for ResBlock C=Cin=Cout
        // any C works because the kernel handles partial tiles. Continue.
    }

    printf("=== ResNet Block (implicit_gemm) Benchmark ===\n");
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

    // FP16 reshaped weights for implicit_gemm
    int K_dim = C * 9;
    auto h_W1f = driver.host_alloc<__half>((size_t)K_dim * C);
    auto h_W2f = driver.host_alloc<__half>((size_t)K_dim * C);
    reshape_weights_to_col(h_W1.get(), h_W1f.get(), C, C, 3, 3);
    reshape_weights_to_col(h_W2.get(), h_W2f.get(), C, C, 3, 3);

    auto d_X   = driver.device_alloc<float>(elems);
    auto d_g   = driver.device_alloc<float>(C);
    auto d_b   = driver.device_alloc<float>(C);
    auto d_W1f = driver.device_alloc<__half>((size_t)K_dim * C);
    auto d_W2f = driver.device_alloc<__half>((size_t)K_dim * C);
    auto d_t1  = driver.device_alloc<float>(elems);
    auto d_t2  = driver.device_alloc<float>(elems);
    auto d_Y   = driver.device_alloc<float>(elems);

    driver.copy_h2d(d_X, h_X, elems * sizeof(float));
    driver.copy_h2d(d_g, h_g, C * sizeof(float));
    driver.copy_h2d(d_b, h_b, C * sizeof(float));
    driver.copy_h2d(d_W1f, h_W1f, (size_t)K_dim * C * sizeof(__half));
    driver.copy_h2d(d_W2f, h_W2f, (size_t)K_dim * C * sizeof(__half));

    CUmodule mod_rb, mod_conv_ig;
    CHECK_CU(cuModuleLoad(&mod_rb, "resblock_fused.sm_86.cubin"));
    CHECK_CU(cuModuleLoad(&mod_conv_ig, "../conv2d/conv2d_implicit_gemm.sm_86.cubin"));
    CUfunction fn_gn_silu, fn_residual, fn_conv_ig;
    CHECK_CU(cuModuleGetFunction(&fn_gn_silu, mod_rb, "groupnorm_silu_fused"));
    CHECK_CU(cuModuleGetFunction(&fn_residual, mod_rb, "residual_add"));
    CHECK_CU(cuModuleGetFunction(&fn_conv_ig, mod_conv_ig, "implicit_gemm_conv"));

    // Set max dynamic smem for implicit_gemm
    CHECK_CU(cuFuncSetAttribute(fn_conv_ig,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, SMEM_BYTES_IGEMM));

    int gn_grid  = N * G;
    int res_grid = ((int)elems + 255) / 256;
    int M_dim    = N * H * W;
    int grid_m   = (M_dim + BLOCK_M_IGEMM - 1) / BLOCK_M_IGEMM;
    int grid_n   = (C     + BLOCK_N_IGEMM - 1) / BLOCK_N_IGEMM;
    int kH = 3, kW = 3, pad = 1;
    int out_H = H, out_W = W;

    auto run_block = [&]() {
        // GN+SiLU 1: X → t1
        void *a1[] = { &d_X.ptr, &d_g.ptr, &d_b.ptr, &d_t1.ptr, &N, &C, &H, &W, &G, &eps };
        CHECK_CU(cuLaunchKernel(fn_gn_silu, gn_grid, 1, 1, 32, 1, 1, 0, nullptr, a1, nullptr));

        // Conv2d 1 (implicit_gemm): t1 → t2
        void *c1[] = { &d_t1.ptr, &d_W1f.ptr, &d_t2.ptr,
                       &N, &H, &W, &C,            // N, H_in, W_in, Cin
                       &kH, &kW, &pad,
                       &out_H, &out_W,
                       &M_dim, &K_dim, &C };       // M, K_dim, Cout
        CHECK_CU(cuLaunchKernel(fn_conv_ig, grid_m, grid_n, 1, 128, 1, 1,
                                SMEM_BYTES_IGEMM, nullptr, c1, nullptr));

        // GN+SiLU 2: t2 → t1
        void *a2[] = { &d_t2.ptr, &d_g.ptr, &d_b.ptr, &d_t1.ptr, &N, &C, &H, &W, &G, &eps };
        CHECK_CU(cuLaunchKernel(fn_gn_silu, gn_grid, 1, 1, 32, 1, 1, 0, nullptr, a2, nullptr));

        // Conv2d 2 (implicit_gemm): t1 → t2
        void *c2[] = { &d_t1.ptr, &d_W2f.ptr, &d_t2.ptr,
                       &N, &H, &W, &C,
                       &kH, &kW, &pad,
                       &out_H, &out_W,
                       &M_dim, &K_dim, &C };
        CHECK_CU(cuLaunchKernel(fn_conv_ig, grid_m, grid_n, 1, 128, 1, 1,
                                SMEM_BYTES_IGEMM, nullptr, c2, nullptr));

        // Residual add: t2 + X → Y
        int total = (int)elems;
        void *r[] = { &d_t2.ptr, &d_X.ptr, &d_Y.ptr, &total };
        CHECK_CU(cuLaunchKernel(fn_residual, res_grid, 1, 1, 256, 1, 1, 0, nullptr, r, nullptr));
    };

    // Correctness
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
                     "ResNet Block (GN+SiLU x2 + implicit_gemm Conv x2 + add)");
    }

    // Performance
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
    printf("  %-45s %7.3f ms  %7.1f GFLOPS\n",
           "Full ResNet Block (5 kernels, implicit_gemm)", avg_ms, gflops);

    return 0;
}
