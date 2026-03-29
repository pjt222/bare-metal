/*
 * bench.cu — Group Normalization benchmark: correctness + throughput
 *
 * Tests both NHWC (groupnorm) and NCHW (groupnorm_nchw) kernels.
 *
 * Build:
 *   nvcc -arch=sm_86 -O2 -o bench bench.cu -lcuda -I../../phase2/common
 *
 * Usage:
 *   ./bench                      # default: N=4, C=32, H=64, W=64, G=8
 *   ./bench 1 320 16 16 32       # SD inference: N=1, C=320, H=16, W=16, G=32
 *   ./bench 4 512 8 8 16         # training batch with larger channels
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda.h>

#include "../../phase2/common/bench.h"
#include "../../phase2/common/check.h"

// -----------------------------------------------------------------------
// CPU reference: Group Normalization (NHWC layout)
// -----------------------------------------------------------------------
static void cpu_groupnorm_nhwc(
    const float *X, const float *gamma, const float *beta,
    float *Y,
    int N, int C, int H, int W, int num_groups, float epsilon
) {
    int channels_per_group = C / num_groups;
    int spatial_size = H * W;
    int group_size = channels_per_group * spatial_size;

    for (int n = 0; n < N; n++) {
        for (int g = 0; g < num_groups; g++) {
            int channel_base = g * channels_per_group;

            // Compute mean
            double sum = 0.0;
            for (int s = 0; s < spatial_size; s++) {
                for (int lc = 0; lc < channels_per_group; lc++) {
                    int gc = channel_base + lc;
                    float val = X[(size_t)n * H * W * C + (size_t)s * C + gc];
                    sum += val;
                }
            }
            float mean = (float)(sum / group_size);

            // Compute variance
            double var_sum = 0.0;
            for (int s = 0; s < spatial_size; s++) {
                for (int lc = 0; lc < channels_per_group; lc++) {
                    int gc = channel_base + lc;
                    float val = X[(size_t)n * H * W * C + (size_t)s * C + gc];
                    float diff = val - mean;
                    var_sum += (double)diff * diff;
                }
            }
            float variance = (float)(var_sum / group_size);
            float inv_std = 1.0f / sqrtf(variance + epsilon);

            // Normalize and apply gamma/beta
            for (int s = 0; s < spatial_size; s++) {
                for (int lc = 0; lc < channels_per_group; lc++) {
                    int gc = channel_base + lc;
                    size_t flat = (size_t)n * H * W * C + (size_t)s * C + gc;
                    float normalized = (X[flat] - mean) * inv_std;
                    Y[flat] = gamma[gc] * normalized + beta[gc];
                }
            }
        }
    }
}

// -----------------------------------------------------------------------
// CPU reference: Group Normalization (NCHW layout)
// -----------------------------------------------------------------------
static void cpu_groupnorm_nchw(
    const float *X, const float *gamma, const float *beta,
    float *Y,
    int N, int C, int H, int W, int num_groups, float epsilon
) {
    int channels_per_group = C / num_groups;
    int spatial_size = H * W;
    int group_size = channels_per_group * spatial_size;

    for (int n = 0; n < N; n++) {
        for (int g = 0; g < num_groups; g++) {
            int channel_base = g * channels_per_group;

            // Compute mean
            double sum = 0.0;
            for (int lc = 0; lc < channels_per_group; lc++) {
                int gc = channel_base + lc;
                for (int s = 0; s < spatial_size; s++) {
                    float val = X[(size_t)n * C * H * W + (size_t)gc * H * W + s];
                    sum += val;
                }
            }
            float mean = (float)(sum / group_size);

            // Compute variance
            double var_sum = 0.0;
            for (int lc = 0; lc < channels_per_group; lc++) {
                int gc = channel_base + lc;
                for (int s = 0; s < spatial_size; s++) {
                    float val = X[(size_t)n * C * H * W + (size_t)gc * H * W + s];
                    float diff = val - mean;
                    var_sum += (double)diff * diff;
                }
            }
            float variance = (float)(var_sum / group_size);
            float inv_std = 1.0f / sqrtf(variance + epsilon);

            // Normalize
            for (int lc = 0; lc < channels_per_group; lc++) {
                int gc = channel_base + lc;
                for (int s = 0; s < spatial_size; s++) {
                    size_t flat = (size_t)n * C * H * W + (size_t)gc * H * W + s;
                    float normalized = (X[flat] - mean) * inv_std;
                    Y[flat] = gamma[gc] * normalized + beta[gc];
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    int N          = (argc > 1) ? atoi(argv[1]) : 4;
    int C          = (argc > 2) ? atoi(argv[2]) : 32;
    int H          = (argc > 3) ? atoi(argv[3]) : 64;
    int W          = (argc > 4) ? atoi(argv[4]) : 64;
    int num_groups = (argc > 5) ? atoi(argv[5]) : 8;

    if (C % num_groups != 0) {
        fprintf(stderr, "C=%d must be divisible by num_groups=%d\n", C, num_groups);
        return 1;
    }
    if ((C / num_groups * H * W) % 32 != 0) {
        fprintf(stderr, "group_size=%d must be divisible by WARP_SIZE=32\n",
                C / num_groups * H * W);
        return 1;
    }

    float epsilon = 1e-5f;
    size_t total_elements = (size_t)N * C * H * W;
    int channels_per_group = C / num_groups;
    int group_size = channels_per_group * H * W;

    printf("=== Group Normalization — SHFL.BFLY + MUFU.RSQ + FFMA ===\n");
    printf("N=%d  C=%d  H=%d  W=%d  G=%d  (C/G=%d  group_size=%d)\n\n",
           N, C, H, W, num_groups, channels_per_group, group_size);

    CHECK_CU(cuInit(0));
    CUdevice cu_dev; CHECK_CU(cuDeviceGet(&cu_dev, 0));
    char devname[256]; CHECK_CU(cuDeviceGetName(devname, sizeof(devname), cu_dev));
    printf("Device: %s\n\n", devname);

    CUcontext ctx; CHECK_CU(cuCtxCreate(&ctx, 0, cu_dev));

    CUmodule mod;
    CUfunction fn_nhwc, fn_nchw;
    if (cuModuleLoad(&mod, "groupnorm.sm_86.cubin") != CUDA_SUCCESS) {
        fprintf(stderr, "Cannot load groupnorm.sm_86.cubin\n");
        return 1;
    }
    CHECK_CU(cuModuleGetFunction(&fn_nhwc, mod, "groupnorm"));
    CHECK_CU(cuModuleGetFunction(&fn_nchw, mod, "groupnorm_nchw"));
    printf("Kernels loaded.\n\n");

    // Host buffers
    float *host_X     = (float*)malloc(total_elements * sizeof(float));
    float *host_gamma = (float*)malloc(C * sizeof(float));
    float *host_beta  = (float*)malloc(C * sizeof(float));
    float *host_Y     = (float*)malloc(total_elements * sizeof(float));
    float *host_ref   = (float*)malloc(total_elements * sizeof(float));

    // Random input + learnable params
    fill_random(host_X,     total_elements, 42);
    fill_random(host_gamma, C, 43);
    fill_random(host_beta,  C, 44);
    // Scale gamma/beta to reasonable range (not just [-1,1])
    for (int c = 0; c < C; c++) {
        host_gamma[c] = host_gamma[c] * 0.5f + 1.0f;   // ~ [0.5, 1.5]
        host_beta[c]  = host_beta[c]  * 0.1f;            // ~ [-0.1, 0.1]
    }

    // Device buffers
    CUdeviceptr dev_X, dev_gamma, dev_beta, dev_Y;
    CHECK_CU(cuMemAlloc(&dev_X,     total_elements * sizeof(float)));
    CHECK_CU(cuMemAlloc(&dev_gamma, C * sizeof(float)));
    CHECK_CU(cuMemAlloc(&dev_beta,  C * sizeof(float)));
    CHECK_CU(cuMemAlloc(&dev_Y,     total_elements * sizeof(float)));

    CHECK_CU(cuMemcpyHtoD(dev_X,     host_X,     total_elements * sizeof(float)));
    CHECK_CU(cuMemcpyHtoD(dev_gamma, host_gamma, C * sizeof(float)));
    CHECK_CU(cuMemcpyHtoD(dev_beta,  host_beta,  C * sizeof(float)));

    // =========================================================
    // Correctness: groupnorm (NHWC)
    // =========================================================
    printf("Correctness (NHWC layout):\n");

    cpu_groupnorm_nhwc(host_X, host_gamma, host_beta, host_ref,
                       N, C, H, W, num_groups, epsilon);

    int grid_size = N * num_groups;
    void *args_nhwc[] = { &dev_X, &dev_gamma, &dev_beta, &dev_Y,
                          &N, &C, &H, &W, &num_groups, &epsilon };
    CHECK_CU(cuMemsetD32(dev_Y, 0, total_elements));
    CHECK_CU(cuLaunchKernel(fn_nhwc,
        grid_size, 1, 1,
        32,        1, 1,
        0, NULL, args_nhwc, NULL));
    CHECK_CU(cuCtxSynchronize());
    CHECK_CU(cuMemcpyDtoH(host_Y, dev_Y, total_elements * sizeof(float)));

    auto result_nhwc = check_fp32(host_Y, host_ref, total_elements, 1e-5f, 1e-4f);
    print_check_result("groupnorm NHWC", result_nhwc);

    // =========================================================
    // Correctness: groupnorm_nchw (NCHW)
    // =========================================================
    printf("\nCorrectness (NCHW layout):\n");

    // Need to convert X from NHWC to NCHW for nchw kernel
    // Allocation for NCHW test: X_nchw[n][c][h][w] = X_nhwc[n][h][w][c]
    float *host_X_nchw   = (float*)malloc(total_elements * sizeof(float));
    float *host_ref_nchw = (float*)malloc(total_elements * sizeof(float));

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    // NHWC: [n][h][w][c]
                    size_t nhwc_idx = (size_t)n * H * W * C + (size_t)h * W * C + w * C + c;
                    // NCHW: [n][c][h][w]
                    size_t nchw_idx = (size_t)n * C * H * W + (size_t)c * H * W + h * W + w;
                    host_X_nchw[nchw_idx] = host_X[nhwc_idx];
                }
            }
        }
    }

    cpu_groupnorm_nchw(host_X_nchw, host_gamma, host_beta, host_ref_nchw,
                       N, C, H, W, num_groups, epsilon);

    CUdeviceptr dev_X_nchw, dev_Y_nchw;
    CHECK_CU(cuMemAlloc(&dev_X_nchw, total_elements * sizeof(float)));
    CHECK_CU(cuMemAlloc(&dev_Y_nchw, total_elements * sizeof(float)));
    CHECK_CU(cuMemcpyHtoD(dev_X_nchw, host_X_nchw, total_elements * sizeof(float)));

    void *args_nchw[] = { &dev_X_nchw, &dev_gamma, &dev_beta, &dev_Y_nchw,
                          &N, &C, &H, &W, &num_groups, &epsilon };
    CHECK_CU(cuMemsetD32(dev_Y_nchw, 0, total_elements));
    CHECK_CU(cuLaunchKernel(fn_nchw,
        grid_size, 1, 1,
        32,        1, 1,
        0, NULL, args_nchw, NULL));
    CHECK_CU(cuCtxSynchronize());

    float *host_Y_nchw = (float*)malloc(total_elements * sizeof(float));
    CHECK_CU(cuMemcpyDtoH(host_Y_nchw, dev_Y_nchw, total_elements * sizeof(float)));

    auto result_nchw = check_fp32(host_Y_nchw, host_ref_nchw, total_elements, 1e-5f, 1e-4f);
    print_check_result("groupnorm NCHW", result_nchw);

    // =========================================================
    // Performance
    // =========================================================
    printf("\nPerformance:\n");

    int warmup = 10, bench_iters = 200;
    float avg_ms_nhwc, avg_ms_nchw;

    // --- NHWC throughput ---
    for (int i = 0; i < warmup; i++) {
        CHECK_CU(cuLaunchKernel(fn_nhwc,
            grid_size, 1, 1, 32, 1, 1, 0, NULL, args_nhwc, NULL));
    }
    CHECK_CU(cuCtxSynchronize());
    {
        BenchTimer timer;
        timer.start();
        for (int i = 0; i < bench_iters; i++) {
            CHECK_CU(cuLaunchKernel(fn_nhwc,
                grid_size, 1, 1, 32, 1, 1, 0, NULL, args_nhwc, NULL));
        }
        avg_ms_nhwc = timer.stop_ms() / bench_iters;
    }

    // GroupNorm reads X twice (Phase 1 + Phase 3) and writes Y once: 3 × total_elements × 4 bytes
    // Also reads gamma/beta once each: 2 × C × 4 bytes (negligible)
    double bytes_touched = 3.0 * total_elements * sizeof(float);
    double gb_per_sec_nhwc = bytes_touched / 1e9 / (avg_ms_nhwc / 1000.0);

    printf("  %-38s %7.3f ms   %6.1f GB/s\n",
           "groupnorm NHWC", avg_ms_nhwc, gb_per_sec_nhwc);

    // --- NCHW throughput ---
    for (int i = 0; i < warmup; i++) {
        CHECK_CU(cuLaunchKernel(fn_nchw,
            grid_size, 1, 1, 32, 1, 1, 0, NULL, args_nchw, NULL));
    }
    CHECK_CU(cuCtxSynchronize());
    {
        BenchTimer timer;
        timer.start();
        for (int i = 0; i < bench_iters; i++) {
            CHECK_CU(cuLaunchKernel(fn_nchw,
                grid_size, 1, 1, 32, 1, 1, 0, NULL, args_nchw, NULL));
        }
        avg_ms_nchw = timer.stop_ms() / bench_iters;
    }
    double gb_per_sec_nchw = bytes_touched / 1e9 / (avg_ms_nchw / 1000.0);
    printf("  %-38s %7.3f ms   %6.1f GB/s\n",
           "groupnorm NCHW", avg_ms_nchw, gb_per_sec_nchw);

    printf("\nMemory bandwidth ceiling: ~608 GB/s (RTX 3070 Ti)\n");
    printf("GroupNorm reads X twice (Welford + normalize) + writes Y once → 3x bandwidth limit ~202 GB/s\n");

    printf("\nSASS inspection:\n");
    printf("  cuobjdump -sass groupnorm.sm_86.cubin | grep -E 'SHFL|MUFU'\n");
    printf("  → SHFL.BFLY  (5 rounds: offset 16, 8, 4, 2, 1) — Welford warp reduction\n");
    printf("  → MUFU.RSQ   — rsqrtf(var + epsilon)\n");
    printf("  → MUFU.RCP   — delta / count in Welford online mean update\n");

    // Cleanup
    cuMemFree(dev_X); cuMemFree(dev_gamma); cuMemFree(dev_beta); cuMemFree(dev_Y);
    cuMemFree(dev_X_nchw); cuMemFree(dev_Y_nchw);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);

    free(host_X); free(host_gamma); free(host_beta);
    free(host_Y); free(host_ref);
    free(host_X_nchw); free(host_ref_nchw); free(host_Y_nchw);
    return 0;
}
