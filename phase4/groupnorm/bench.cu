/*
 * bench.cu — Group Normalization benchmark (BenchDriver refactor)
 *
 * Tests both NHWC (groupnorm) and NCHW (groupnorm_nchw) kernels.
 *
 * Build:
 *   nvcc -arch=sm_86 -O2 -o bench bench.cu -lcuda -I../../kernels/_common
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda.h>

#include "../../kernels/_common/bench_driver.h"

static void cpu_groupnorm_nhwc(
    const float *X, const float *gamma, const float *beta,
    float *Y, int N, int C, int H, int W, int G, float eps
) {
    int ch_per_g = C / G;
    int spatial = H * W;
    int gsize = ch_per_g * spatial;
    for (int n = 0; n < N; n++) {
        for (int g = 0; g < G; g++) {
            int cb = g * ch_per_g;
            double sum = 0.0;
            for (int s = 0; s < spatial; s++)
                for (int lc = 0; lc < ch_per_g; lc++)
                    sum += X[(size_t)n * spatial * C + s * C + cb + lc];
            float mean = (float)(sum / gsize);
            double var_sum = 0.0;
            for (int s = 0; s < spatial; s++) {
                for (int lc = 0; lc < ch_per_g; lc++) {
                    float d = X[(size_t)n * spatial * C + s * C + cb + lc] - mean;
                    var_sum += (double)d * d;
                }
            }
            float inv_std = 1.0f / sqrtf((float)(var_sum / gsize) + eps);
            for (int s = 0; s < spatial; s++) {
                for (int lc = 0; lc < ch_per_g; lc++) {
                    size_t flat = (size_t)n * spatial * C + s * C + cb + lc;
                    Y[flat] = gamma[cb + lc] * ((X[flat] - mean) * inv_std) + beta[cb + lc];
                }
            }
        }
    }
}

static void cpu_groupnorm_nchw(
    const float *X, const float *gamma, const float *beta,
    float *Y, int N, int C, int H, int W, int G, float eps
) {
    int ch_per_g = C / G;
    int spatial = H * W;
    int gsize = ch_per_g * spatial;
    for (int n = 0; n < N; n++) {
        for (int g = 0; g < G; g++) {
            int cb = g * ch_per_g;
            double sum = 0.0;
            for (int c = cb; c < cb + ch_per_g; c++)
                for (int s = 0; s < spatial; s++)
                    sum += X[(size_t)n * C * spatial + c * spatial + s];
            float mean = (float)(sum / gsize);
            double var_sum = 0.0;
            for (int c = cb; c < cb + ch_per_g; c++) {
                for (int s = 0; s < spatial; s++) {
                    float d = X[(size_t)n * C * spatial + c * spatial + s] - mean;
                    var_sum += (double)d * d;
                }
            }
            float inv_std = 1.0f / sqrtf((float)(var_sum / gsize) + eps);
            for (int c = cb; c < cb + ch_per_g; c++) {
                for (int s = 0; s < spatial; s++) {
                    size_t flat = (size_t)n * C * spatial + c * spatial + s;
                    Y[flat] = gamma[c] * ((X[flat] - mean) * inv_std) + beta[c];
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 4;
    int C = (argc > 2) ? atoi(argv[2]) : 32;
    int H = (argc > 3) ? atoi(argv[3]) : 64;
    int W = (argc > 4) ? atoi(argv[4]) : 64;
    int G = (argc > 5) ? atoi(argv[5]) : 8;
    float eps = 1e-5f;

    int total = N * C * H * W;
    int ch_per_g = C / G;
    int group_size = ch_per_g * H * W;
    printf("=== Group Normalization Benchmark (BenchDriver refactor) ===\n");
    printf("N=%d C=%d H=%d W=%d G=%d (C/G=%d group_size=%d)\n\n",
           N, C, H, W, G, ch_per_g, group_size);

    BenchDriver driver;
    driver.init_context();

    auto h_X = driver.host_alloc<float>(total);
    auto h_g = driver.host_alloc<float>(C);
    auto h_b = driver.host_alloc<float>(C);
    auto h_ref = driver.host_alloc<float>(total);
    auto h_out = driver.host_alloc<float>(total);

    fill_random(h_X.get(), total, 42);
    fill_random(h_g.get(), C, 43);
    fill_random(h_b.get(), C, 44);
    for (int c = 0; c < C; c++) {
        h_g[c] = h_g[c] * 0.5f + 1.0f;
        h_b[c] = h_b[c] * 0.1f;
    }

    auto d_X = driver.device_alloc<float>(total);
    auto d_g = driver.device_alloc<float>(C);
    auto d_b = driver.device_alloc<float>(C);
    auto d_Y = driver.device_alloc<float>(total);

    driver.copy_h2d(d_X, h_X, total * sizeof(float));
    driver.copy_h2d(d_g, h_g, C * sizeof(float));
    driver.copy_h2d(d_b, h_b, C * sizeof(float));

    CUmodule mod;
    CHECK_CU(cuModuleLoad(&mod, "groupnorm.sm_86.cubin"));
    CUfunction fn_nhwc, fn_nchw;
    CHECK_CU(cuModuleGetFunction(&fn_nhwc, mod, "groupnorm"));
    CHECK_CU(cuModuleGetFunction(&fn_nchw, mod, "groupnorm_nchw"));

    dim3 grid(N * G, 1, 1);
    dim3 block(32, 1, 1);

    // =====================================================================
    // NHWC correctness
    // =====================================================================
    cpu_groupnorm_nhwc(h_X.get(), h_g.get(), h_b.get(), h_ref.get(),
                       N, C, H, W, G, eps);
    void *args_nhwc[] = { &d_X.ptr, &d_g.ptr, &d_b.ptr, &d_Y.ptr,
                          &N, &C, &H, &W, &G, &eps };
    CHECK_CU(cuMemsetD32((CUdeviceptr)d_Y.ptr, 0, total));
    CHECK_CU(cuLaunchKernel(fn_nhwc, grid.x, grid.y, grid.z,
                            block.x, block.y, block.z,
                            0, nullptr, args_nhwc, nullptr));
    CHECK_CU(cuCtxSynchronize());
    driver.copy_d2h(h_out, d_Y, total * sizeof(float));
    driver.check(h_out.get(), h_ref.get(), total, 1e-5f, 1e-4f, "groupnorm NHWC");

    // =====================================================================
    // NCHW correctness (transpose X first)
    // =====================================================================
    auto h_X_nc = driver.host_alloc<float>(total);
    auto h_ref_nc = driver.host_alloc<float>(total);
    for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
            for (int h = 0; h < H; h++)
                for (int w = 0; w < W; w++) {
                    size_t nhwc = (size_t)n * H * W * C + (size_t)h * W * C + w * C + c;
                    size_t nchw = (size_t)n * C * H * W + (size_t)c * H * W + h * W + w;
                    h_X_nc[nchw] = h_X[nhwc];
                }

    cpu_groupnorm_nchw(h_X_nc.get(), h_g.get(), h_b.get(), h_ref_nc.get(),
                       N, C, H, W, G, eps);

    auto d_X_nc = driver.device_alloc<float>(total);
    auto d_Y_nc = driver.device_alloc<float>(total);
    driver.copy_h2d(d_X_nc, h_X_nc, total * sizeof(float));

    void *args_nchw[] = { &d_X_nc.ptr, &d_g.ptr, &d_b.ptr, &d_Y_nc.ptr,
                          &N, &C, &H, &W, &G, &eps };
    CHECK_CU(cuMemsetD32((CUdeviceptr)d_Y_nc.ptr, 0, total));
    CHECK_CU(cuLaunchKernel(fn_nchw, grid.x, grid.y, grid.z,
                            block.x, block.y, block.z,
                            0, nullptr, args_nchw, nullptr));
    CHECK_CU(cuCtxSynchronize());
    driver.copy_d2h(h_out, d_Y_nc, total * sizeof(float));
    driver.check(h_out.get(), h_ref_nc.get(), total, 1e-5f, 1e-4f, "groupnorm NCHW");

    // =====================================================================
    // Performance
    // =====================================================================
    printf("\nPerformance:\n");
    double bytes = 3.0 * total * sizeof(float);

    float ms_nhwc = driver.benchmark_kernel(fn_nhwc, grid, block, 0, args_nhwc, 10, 200);
    printf("  %-38s %7.3f ms   %6.1f GB/s\n",
           "groupnorm NHWC", ms_nhwc, bytes / 1e9 / (ms_nhwc / 1000.0));

    float ms_nchw = driver.benchmark_kernel(fn_nchw, grid, block, 0, args_nchw, 10, 200);
    printf("  %-38s %7.3f ms   %6.1f GB/s\n",
           "groupnorm NCHW", ms_nchw, bytes / 1e9 / (ms_nchw / 1000.0));

    printf("\nMemory bandwidth ceiling: ~608 GB/s (RTX 3070 Ti)\n");
    printf("GroupNorm reads X twice (Welford + normalize) + writes Y once -> 3x limit ~202 GB/s\n");

    cuModuleUnload(mod);
    return 0;
}
