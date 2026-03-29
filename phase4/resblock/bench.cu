/*
 * bench.cu — ResNet Block integration benchmark
 *
 * Chains all Phase 4 primitives into a full ResNet block forward pass:
 *   x → GroupNorm+SiLU → Conv2d(3×3) → GroupNorm+SiLU → Conv2d(3×3) → + x
 *
 * Loads kernels from:
 *   resblock.sm_86.cubin   (groupnorm_silu_fused, residual_add)
 *   ../conv2d/conv2d.sm_86.cubin  (conv2d_nhwc)
 *
 * Build:
 *   # First build conv2d cubin (if not already built):
 *   nvcc --cubin -arch=sm_86 -O2 -o ../conv2d/conv2d.sm_86.cubin ../conv2d/conv2d.cu
 *   # Then:
 *   nvcc -arch=sm_86 -O2 -o bench bench.cu -lcuda -I../../phase2/common
 *
 * Usage:
 *   ./bench                      # N=1, C=64, H=32, W=32, G=8
 *   ./bench 1 320 16 16 32       # SD UNet block (320ch, 16×16 feature map)
 *   ./bench 2 128 64 64 8        # SD low-res block
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda.h>

#include "../../phase2/common/bench.h"
#include "../../phase2/common/check.h"

// -----------------------------------------------------------------------
// CPU SiLU activation
// -----------------------------------------------------------------------
static inline float cpu_silu(float x) {
    return x / (1.0f + expf(-x));
}

// -----------------------------------------------------------------------
// CPU GroupNorm + SiLU reference (NHWC)
// -----------------------------------------------------------------------
static void cpu_groupnorm_silu(
    const float *X, const float *gamma, const float *beta,
    float *Y, int N, int C, int H, int W, int G, float eps
) {
    int cpg = C / G;
    int spatial = H * W;
    for (int n = 0; n < N; n++) {
        for (int g = 0; g < G; g++) {
            int cb = g * cpg;
            int group_size = cpg * spatial;
            double sum = 0.0;
            for (int s = 0; s < spatial; s++)
                for (int lc = 0; lc < cpg; lc++)
                    sum += X[(size_t)n*H*W*C + s*C + cb+lc];
            float mean = (float)(sum / group_size);

            double var_sum = 0.0;
            for (int s = 0; s < spatial; s++)
                for (int lc = 0; lc < cpg; lc++) {
                    float d = X[(size_t)n*H*W*C + s*C + cb+lc] - mean;
                    var_sum += d*d;
                }
            float inv_std = 1.0f / sqrtf((float)(var_sum / group_size) + eps);

            for (int s = 0; s < spatial; s++) {
                for (int lc = 0; lc < cpg; lc++) {
                    int gc = cb + lc;
                    size_t flat = (size_t)n*H*W*C + s*C + gc;
                    float normalized = (X[flat] - mean) * inv_std;
                    float scaled = gamma[gc] * normalized + beta[gc];
                    Y[flat] = cpu_silu(scaled);
                }
            }
        }
    }
}

// -----------------------------------------------------------------------
// CPU 3×3 convolution reference (NHWC, stride=1, pad=1)
// -----------------------------------------------------------------------
static void cpu_conv2d_nhwc(
    const float *X, const float *W, const float *bias, float *Y,
    int N, int H, int W_dim, int Cin, int Cout
) {
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W_dim; w++) {
                for (int co = 0; co < Cout; co++) {
                    double acc = (bias != nullptr) ? (double)bias[co] : 0.0;
                    for (int kh = 0; kh < 3; kh++) {
                        for (int kw = 0; kw < 3; kw++) {
                            int h_in = h+kh-1, w_in = w+kw-1;
                            if (h_in < 0 || h_in >= H || w_in < 0 || w_in >= W_dim) continue;
                            for (int ci = 0; ci < Cin; ci++) {
                                size_t xi = (size_t)n*H*W_dim*Cin + h_in*W_dim*Cin + w_in*Cin + ci;
                                size_t wi = (size_t)co*9*Cin + (kh*3+kw)*Cin + ci;
                                acc += X[xi] * W[wi];
                            }
                        }
                    }
                    Y[(size_t)n*H*W_dim*Cout + h*W_dim*Cout + w*Cout + co] = (float)acc;
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    int N   = (argc > 1) ? atoi(argv[1]) : 1;
    int C   = (argc > 2) ? atoi(argv[2]) : 64;
    int H   = (argc > 3) ? atoi(argv[3]) : 32;
    int W   = (argc > 4) ? atoi(argv[4]) : 32;
    int G   = (argc > 5) ? atoi(argv[5]) : 8;

    if (C % G != 0) { fprintf(stderr, "C must be divisible by G\n"); return 1; }
    if ((C / G * H * W) % 32 != 0) {
        fprintf(stderr, "group_size must be divisible by 32\n"); return 1;
    }

    float eps = 1e-5f;
    size_t nhwc_elems = (size_t)N * H * W * C;
    size_t w3_elems   = (size_t)C * 9 * C;

    printf("=== ResNet Block: GroupNorm+SiLU → Conv2d(3×3) × 2 → Residual Add ===\n");
    printf("N=%d  C=%d  H=%d  W=%d  G=%d\n", N, C, H, W, G);
    printf("Total FLOPs per forward pass: %.3f GFLOPS\n\n",
           2.0 * 2 * N * H * W * C * C * 9 / 1e9);  // 2 convolutions

    CHECK_CU(cuInit(0));
    CUdevice cu_dev; CHECK_CU(cuDeviceGet(&cu_dev, 0));
    char devname[256]; CHECK_CU(cuDeviceGetName(devname, sizeof(devname), cu_dev));
    printf("Device: %s\n\n", devname);

    CUcontext ctx; CHECK_CU(cuCtxCreate(&ctx, 0, cu_dev));

    // Load both cubins
    CUmodule mod_rb, mod_conv;
    CUfunction fn_gn_silu, fn_residual, fn_conv;

    if (cuModuleLoad(&mod_rb, "resblock.sm_86.cubin") != CUDA_SUCCESS) {
        fprintf(stderr, "Cannot load resblock.sm_86.cubin\n"); return 1;
    }
    if (cuModuleLoad(&mod_conv, "../conv2d/conv2d.sm_86.cubin") != CUDA_SUCCESS) {
        fprintf(stderr, "Cannot load ../conv2d/conv2d.sm_86.cubin\n"); return 1;
    }
    CHECK_CU(cuModuleGetFunction(&fn_gn_silu,  mod_rb,   "groupnorm_silu_fused"));
    CHECK_CU(cuModuleGetFunction(&fn_residual, mod_rb,   "residual_add"));
    CHECK_CU(cuModuleGetFunction(&fn_conv,     mod_conv, "conv2d_nhwc"));
    printf("Kernels loaded.\n\n");

    // Host buffers
    float *host_X      = (float*)malloc(nhwc_elems * sizeof(float));
    float *host_gamma  = (float*)malloc(C * sizeof(float));
    float *host_beta   = (float*)malloc(C * sizeof(float));
    float *host_W1     = (float*)malloc(w3_elems * sizeof(float));
    float *host_W2     = (float*)malloc(w3_elems * sizeof(float));
    float *host_bias   = (float*)malloc(C * sizeof(float));
    float *host_Y_gpu  = (float*)malloc(nhwc_elems * sizeof(float));
    float *host_Y_ref  = (float*)malloc(nhwc_elems * sizeof(float));

    fill_random(host_X,     nhwc_elems, 1);
    fill_random(host_gamma, C,          2);
    fill_random(host_beta,  C,          3);
    fill_random(host_W1,    w3_elems,   4);
    fill_random(host_W2,    w3_elems,   5);
    fill_random(host_bias,  C,          6);
    // Scale gamma/W to keep values in range
    for (int i = 0; i < C; i++) { host_gamma[i] = host_gamma[i]*0.5f + 1.0f; }
    for (size_t i = 0; i < w3_elems; i++) { host_W1[i] *= 0.05f; host_W2[i] *= 0.05f; }
    for (int i = 0; i < C; i++) host_bias[i] = 0.0f;  // zero bias for simplicity

    // Device buffers
    CUdeviceptr dev_X, dev_gamma, dev_beta, dev_W1, dev_W2, dev_bias;
    CUdeviceptr dev_tmp1, dev_tmp2, dev_Y;  // intermediate buffers

    CHECK_CU(cuMemAlloc(&dev_X,     nhwc_elems * sizeof(float)));
    CHECK_CU(cuMemAlloc(&dev_gamma, C * sizeof(float)));
    CHECK_CU(cuMemAlloc(&dev_beta,  C * sizeof(float)));
    CHECK_CU(cuMemAlloc(&dev_W1,    w3_elems * sizeof(float)));
    CHECK_CU(cuMemAlloc(&dev_W2,    w3_elems * sizeof(float)));
    CHECK_CU(cuMemAlloc(&dev_bias,  C * sizeof(float)));
    CHECK_CU(cuMemAlloc(&dev_tmp1,  nhwc_elems * sizeof(float)));  // after GN+SiLU
    CHECK_CU(cuMemAlloc(&dev_tmp2,  nhwc_elems * sizeof(float)));  // after Conv2d
    CHECK_CU(cuMemAlloc(&dev_Y,     nhwc_elems * sizeof(float)));  // final output

    CHECK_CU(cuMemcpyHtoD(dev_X,     host_X,     nhwc_elems * sizeof(float)));
    CHECK_CU(cuMemcpyHtoD(dev_gamma, host_gamma, C * sizeof(float)));
    CHECK_CU(cuMemcpyHtoD(dev_beta,  host_beta,  C * sizeof(float)));
    CHECK_CU(cuMemcpyHtoD(dev_W1,    host_W1,    w3_elems * sizeof(float)));
    CHECK_CU(cuMemcpyHtoD(dev_W2,    host_W2,    w3_elems * sizeof(float)));
    CHECK_CU(cuMemcpyHtoD(dev_bias,  host_bias,  C * sizeof(float)));

    // Grid dimensions
    int gn_grid  = N * G;
    int hw_tiles = (H * W + 15) / 16;
    int c_tiles  = (C + 7) / 8;
    int res_grid = ((int)nhwc_elems + 255) / 256;
    CUdeviceptr dev_bias_null = 0;  // NULL bias for conv

    // Helper: run one full ResNet block forward pass
    auto run_resblock = [&]() {
        // Step 1: GroupNorm + SiLU (X → tmp1)
        void *gn1_args[] = { &dev_X, &dev_gamma, &dev_beta, &dev_tmp1,
                             &N, &C, &H, &W, &G, &eps };
        CHECK_CU(cuLaunchKernel(fn_gn_silu,
            gn_grid, 1, 1,   32, 1, 1,   0, NULL, gn1_args, NULL));

        // Step 2: Conv2d 3×3 (tmp1 → tmp2)
        void *conv1_args[] = { &dev_tmp1, &dev_W1, &dev_bias_null, &dev_tmp2,
                               &N, &H, &W, &C, &C };
        CHECK_CU(cuLaunchKernel(fn_conv,
            N, hw_tiles, c_tiles,
            16, 8, 1,
            0, NULL, conv1_args, NULL));

        // Step 3: GroupNorm + SiLU (tmp2 → tmp1, reuse buffer)
        void *gn2_args[] = { &dev_tmp2, &dev_gamma, &dev_beta, &dev_tmp1,
                             &N, &C, &H, &W, &G, &eps };
        CHECK_CU(cuLaunchKernel(fn_gn_silu,
            gn_grid, 1, 1,   32, 1, 1,   0, NULL, gn2_args, NULL));

        // Step 4: Conv2d 3×3 (tmp1 → tmp2)
        void *conv2_args[] = { &dev_tmp1, &dev_W2, &dev_bias_null, &dev_tmp2,
                               &N, &H, &W, &C, &C };
        CHECK_CU(cuLaunchKernel(fn_conv,
            N, hw_tiles, c_tiles,
            16, 8, 1,
            0, NULL, conv2_args, NULL));

        // Step 5: Residual add (tmp2 + X → Y)
        int total = (int)nhwc_elems;
        void *res_args[] = { &dev_tmp2, &dev_X, &dev_Y, &total };
        CHECK_CU(cuLaunchKernel(fn_residual,
            res_grid, 1, 1,   256, 1, 1,   0, NULL, res_args, NULL));
    };

    // =========================================================
    // Correctness
    // =========================================================
    printf("Correctness:\n");

    // CPU forward pass
    float *cpu_tmp1 = (float*)malloc(nhwc_elems * sizeof(float));
    float *cpu_tmp2 = (float*)malloc(nhwc_elems * sizeof(float));

    cpu_groupnorm_silu(host_X, host_gamma, host_beta,
                       cpu_tmp1, N, C, H, W, G, eps);

    // Only run CPU conv if problem is small enough
    bool run_cpu_ref = ((double)N * H * W * C * C * 9 <= 5e8);
    if (run_cpu_ref) {
        cpu_conv2d_nhwc(cpu_tmp1, host_W1, nullptr, cpu_tmp2, N, H, W, C, C);
        cpu_groupnorm_silu(cpu_tmp2, host_gamma, host_beta,
                           cpu_tmp1, N, C, H, W, G, eps);
        cpu_conv2d_nhwc(cpu_tmp1, host_W2, nullptr, cpu_tmp2, N, H, W, C, C);
        // Residual add
        for (size_t i = 0; i < nhwc_elems; i++) {
            host_Y_ref[i] = cpu_tmp2[i] + host_X[i];
        }
    } else {
        printf("  [CPU reference skipped — config too large for in-time CPU verification]\n");
    }

    // GPU forward pass
    run_resblock();
    CHECK_CU(cuCtxSynchronize());
    CHECK_CU(cuMemcpyDtoH(host_Y_gpu, dev_Y, nhwc_elems * sizeof(float)));

    if (run_cpu_ref) {
        // Conv2d accumulates many floats: 9 × Cin products per element → larger absolute error
        float abs_tol = 1e-2f * (float)sqrtf((float)C);  // scales with Cin depth
        auto result = check_fp32(host_Y_gpu, host_Y_ref, nhwc_elems, abs_tol, 0.1f);
        print_check_result("ResNet Block (GN+SiLU × 2 + Conv2d × 2 + add)", result);
    }

    // =========================================================
    // Performance
    // =========================================================
    printf("\nPerformance:\n");

    int warmup = 5, bench_iters = 100;
    for (int i = 0; i < warmup; i++) run_resblock();
    CHECK_CU(cuCtxSynchronize());

    float avg_ms;
    {
        BenchTimer timer;
        timer.start();
        for (int i = 0; i < bench_iters; i++) run_resblock();
        avg_ms = timer.stop_ms() / bench_iters;
    }

    double total_flops = 2.0 * 2.0 * N * H * W * C * C * 9;  // 2 conv passes
    double gflops = total_flops / 1e9 / (avg_ms / 1000.0);
    // Memory: 5 kernel launches × roughly 3× tensor size each
    double bytes = (double)nhwc_elems * sizeof(float) * 3 * 5;
    double gb_s  = bytes / 1e9 / (avg_ms / 1000.0);

    printf("  %-45s %7.3f ms  %7.1f GFLOPS  (~%.0f GB/s BW)\n",
           "Full ResNet Block (5 kernel launches)", avg_ms, gflops, gb_s);

    printf("\nKernel breakdown (estimated from individual runs):\n");
    printf("  groupnorm_silu_fused × 2  (SHFL.BFLY + MUFU.RSQ + MUFU.EX2)\n");
    printf("  conv2d_nhwc × 2           (FFMA × 310, 9×unrolled, weight-tiled)\n");
    printf("  residual_add × 1          (FADD, memory-bandwidth limited)\n");

    printf("\nSASS primitives in this block:\n");
    printf("  SHFL.BFLY  — Welford warp reduction (GroupNorm, 5 rounds each)\n");
    printf("  MUFU.RSQ   — rsqrtf(var + eps) (GroupNorm)\n");
    printf("  MUFU.EX2   — exp2f(-x * log2e) for SiLU sigmoid\n");
    printf("  MUFU.RCP   — 1/(1 + exp2f(...)) for SiLU sigmoid\n");
    printf("  FFMA       — Conv2d 3×3 inner loop accumulation\n");
    printf("  FADD       — Residual skip connection add\n");

    // Cleanup
    cuMemFree(dev_X); cuMemFree(dev_gamma); cuMemFree(dev_beta);
    cuMemFree(dev_W1); cuMemFree(dev_W2); cuMemFree(dev_bias);
    cuMemFree(dev_tmp1); cuMemFree(dev_tmp2); cuMemFree(dev_Y);
    cuModuleUnload(mod_rb); cuModuleUnload(mod_conv);
    cuCtxDestroy(ctx);

    free(host_X); free(host_gamma); free(host_beta);
    free(host_W1); free(host_W2); free(host_bias);
    free(host_Y_gpu); free(host_Y_ref);
    free(cpu_tmp1); free(cpu_tmp2);
    return 0;
}
