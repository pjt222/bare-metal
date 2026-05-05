/*
 * bench.cu — Softmax benchmark (BenchDriver refactor)
 *
 * Build: nvcc -arch=sm_86 -O2 -o bench bench.cu -lcuda -I../common
 */

#include <cuda.h>
#include <cstdio>
#include <cmath>
#include "../common/bench_driver.h"

static void cpu_softmax(const float *input, float *output,
                        int num_rows, int row_width) {
    for (int row = 0; row < num_rows; row++) {
        const float *in = input + (size_t)row * row_width;
        float *out = output + (size_t)row * row_width;
        float row_max = in[0];
        for (int c = 1; c < row_width; c++) row_max = fmaxf(row_max, in[c]);
        float exp_sum = 0.0f;
        for (int c = 0; c < row_width; c++) {
            out[c] = expf(in[c] - row_max);
            exp_sum += out[c];
        }
        float rcp = 1.0f / exp_sum;
        for (int c = 0; c < row_width; c++) out[c] *= rcp;
    }
}

int main(int argc, char **argv) {
    int num_rows  = (argc > 1) ? atoi(argv[1]) : 65536;
    int row_width = (argc > 2) ? atoi(argv[2]) : 32;

    printf("=== Softmax Benchmark ===\n");
    printf("Input: %d rows × %d cols\n\n", num_rows, row_width);

    BenchDriver driver;
    driver.init_context();

    size_t total = (size_t)num_rows * row_width;
    size_t bytes = total * sizeof(float);
    auto d_in  = driver.device_alloc<float>(total);
    auto d_out = driver.device_alloc<float>(total);
    auto h_in  = driver.host_alloc<float>(total);
    auto h_ref = driver.host_alloc<float>(total);
    auto h_out = driver.host_alloc<float>(total);

    fill_random(h_in.get(), total, 42);
    driver.copy_h2d(d_in, h_in, bytes);
    cpu_softmax(h_in.get(), h_ref.get(), num_rows, row_width);

    struct V {
        const char *name; dim3 g, b;
        bool use; float abs_tol, rel_tol;
    };
    std::vector<V> variants = {
        {"softmax_warp",  dim3(num_rows,1,1), dim3(32,1,1),
         (row_width <= 32), 1e-4f, 1e-4f},
        {"softmax_block", dim3(num_rows,1,1), dim3(128,1,1),
         (row_width <= 4*128), 1e-4f, 1e-4f},
    };

    double gbytes = 2.0 * bytes / 1e9;

    printf("Correctness:\n");
    for (auto &v : variants) {
        if (!v.use) continue;
        CUfunction fn = driver.load_kernel("softmax.sm_86.cubin", v.name);
        void *args[] = { &d_in, &d_out, &num_rows, &row_width };
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
        CUfunction fn = driver.load_kernel("softmax.sm_86.cubin", v.name);
        void *args[] = { &d_in, &d_out, &num_rows, &row_width };
        float ms = driver.benchmark_kernel(fn, v.g, v.b, 0, args, 5, 100);
        printf("  %-20s %7.3f ms   %8.2f GB/s\n", v.name, ms,
               gbytes / (ms / 1000.0));
    }
    printf("\nSASS: cuobjdump -sass softmax.sm_86.cubin | grep -E 'SHFL|MUFU'\n");
    return 0;
}
