/*
 * bench.cu — LayerNorm benchmark (BenchDriver refactor)
 *
 * Build: nvcc -arch=sm_86 -O2 -o bench bench.cu -lcuda -I../common
 */

#include <cuda.h>
#include <cstdio>
#include <cmath>
#include "../common/bench_driver.h"

static void cpu_layernorm(const float *input, const float *gamma,
                          const float *beta, float *output,
                          int num_rows, int row_width, float epsilon) {
    for (int row = 0; row < num_rows; row++) {
        const float *in = input + (size_t)row * row_width;
        float *out = output + (size_t)row * row_width;
        double sum = 0.0;
        for (int c = 0; c < row_width; c++) sum += in[c];
        float mean = (float)(sum / row_width);
        double sq = 0.0;
        for (int c = 0; c < row_width; c++) {
            float d = in[c] - mean;
            sq += d * d;
        }
        float var = (float)(sq / row_width);
        float rsqrt = 1.0f / sqrtf(var + epsilon);
        for (int c = 0; c < row_width; c++)
            out[c] = gamma[c] * (in[c] - mean) * rsqrt + beta[c];
    }
}

int main(int argc, char **argv) {
    int num_rows  = (argc > 1) ? atoi(argv[1]) : 65536;
    int row_width = (argc > 2) ? atoi(argv[2]) : 32;
    float epsilon = 1e-5f;

    printf("=== LayerNorm Benchmark ===\n");
    printf("Input: %d rows x %d cols  epsilon=%.0e\n\n", num_rows, row_width, (double)epsilon);

    BenchDriver driver;
    driver.init_context();

    size_t total = (size_t)num_rows * row_width;
    size_t bytes = total * sizeof(float);
    size_t pbytes = row_width * sizeof(float);

    auto d_in    = driver.device_alloc<float>(total);
    auto d_out   = driver.device_alloc<float>(total);
    auto d_gamma = driver.device_alloc<float>(row_width);
    auto d_beta  = driver.device_alloc<float>(row_width);
    auto h_in    = driver.host_alloc<float>(total);
    auto h_ref   = driver.host_alloc<float>(total);
    auto h_out   = driver.host_alloc<float>(total);
    auto h_gamma = driver.host_alloc<float>(row_width);
    auto h_beta  = driver.host_alloc<float>(row_width);

    fill_random(h_in.get(), total, 42);
    for (int c = 0; c < row_width; c++) { h_gamma[c] = 1.0f; h_beta[c] = 0.0f; }

    driver.copy_h2d(d_in, h_in, bytes);
    driver.copy_h2d(d_gamma, h_gamma, pbytes);
    driver.copy_h2d(d_beta, h_beta, pbytes);

    cpu_layernorm(h_in.get(), h_gamma.get(), h_beta.get(),
                  h_ref.get(), num_rows, row_width, epsilon);

    struct V {
        const char *name; dim3 g, b;
        bool use; float abs_tol, rel_tol;
    };
    std::vector<V> variants = {
        {"layernorm_warp",  dim3(num_rows,1,1), dim3(32,1,1),
         (row_width <= 32), 1e-4f, 1e-3f},
        {"layernorm_block", dim3(num_rows,1,1), dim3(128,1,1),
         (row_width <= 4*128), 1e-4f, 1e-3f},
    };

    double gbytes = (bytes + pbytes * 2 + bytes) / 1e9;

    printf("Correctness:\n");
    for (auto &v : variants) {
        if (!v.use) continue;
        CUfunction fn = driver.load_kernel("layernorm.sm_86.cubin", v.name);
        void *args[] = { &d_in, &d_gamma, &d_beta, &d_out,
                         &num_rows, &row_width, &epsilon };
        CHECK_CU(cuMemsetD32((CUdeviceptr)d_out.get(), 0, total));
        CHECK_CU(cuLaunchKernel(fn, v.g.x, v.g.y, v.g.z,
                                v.b.x, v.b.y, v.b.z,
                                0, NULL, args, NULL));
        CHECK_CU(cuCtxSynchronize());
        driver.copy_d2h(h_out, d_out, bytes);
        driver.check(h_out.get(), h_ref.get(), total, v.abs_tol, v.rel_tol, v.name);
    }

    printf("\nPerformance (avg of 100 runs, 5 warmup):\n");
    for (auto &v : variants) {
        if (!v.use) continue;
        CUfunction fn = driver.load_kernel("layernorm.sm_86.cubin", v.name);
        void *args[] = { &d_in, &d_gamma, &d_beta, &d_out,
                         &num_rows, &row_width, &epsilon };
        float ms = driver.benchmark_kernel(fn, v.g, v.b, 0, args, 5, 100);
        printf("  %-20s %7.3f ms   %8.2f GB/s\n", v.name, ms,
               gbytes / (ms / 1000.0));
    }
    printf("\nSASS: cuobjdump -sass layernorm.sm_86.cubin | grep -E 'SHFL|MUFU'\n");
    return 0;
}
