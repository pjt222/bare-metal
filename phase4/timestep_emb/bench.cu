/*
 * bench.cu — Timestep Embedding benchmark (BenchDriver refactor)
 *
 * Build: nvcc -arch=sm_86 -O2 -o bench bench.cu -lcuda -I../../kernels/_common
 */

#include <cuda.h>
#include <cstdio>
#include <cmath>
#include "../../kernels/_common/bench_driver.h"

#define LOG_MAX_PERIOD 9.210340371976183f

static void cpu_timestep_emb(const float *timesteps, float *output,
                              int batch_size, int d_model) {
    int half_dim = d_model / 2;
    for (int b = 0; b < batch_size; b++) {
        float t = timesteps[b];
        float *out = output + (size_t)b * d_model;
        for (int i = 0; i < half_dim; i++) {
            float freq = expf(-LOG_MAX_PERIOD * (float)i / (float)half_dim);
            float angle = t * freq;
            out[i]            = sinf(angle);
            out[i + half_dim] = cosf(angle);
        }
    }
}

int main(int argc, char **argv) {
    int d_model = (argc > 1) ? atoi(argv[1]) : 512;
    int batch   = (argc > 2) ? atoi(argv[2]) : 1024;
    if (d_model % 2 != 0) { fprintf(stderr, "d_model must be even\n"); return 1; }
    int half_dim = d_model / 2;

    printf("=== Timestep Embedding Benchmark ===\n");
    printf("d_model=%d  batch=%d\n\n", d_model, batch);

    BenchDriver driver;
    driver.init_context();

    size_t emb_elems = (size_t)batch * d_model;
    size_t emb_bytes = emb_elems * sizeof(float);

    auto d_ts   = driver.device_alloc<float>(batch);
    auto d_out  = driver.device_alloc<float>(emb_elems);
    auto h_ts   = driver.host_alloc<float>(batch);
    auto h_ref  = driver.host_alloc<float>(emb_elems);
    auto h_out  = driver.host_alloc<float>(emb_elems);

    for (int i = 0; i < batch; i++)
        h_ts[i] = (float)(rand() % 1000) + (float)rand() / RAND_MAX;

    driver.copy_h2d(d_ts, h_ts, batch * sizeof(float));
    cpu_timestep_emb(h_ts.get(), h_ref.get(), batch, d_model);

    CUfunction fn = driver.load_kernel("timestep_emb.sm_86.cubin", "timestep_emb_batch");
    dim3 grid(batch, 1, 1);
    dim3 block(half_dim, 1, 1);
    void *args[] = { &d_ts, &d_out, &d_model, &batch };

    printf("Correctness:\n");
    CHECK_CU(cuMemsetD32((CUdeviceptr)d_out.get(), 0, emb_elems));
    CHECK_CU(cuLaunchKernel(fn, grid.x, grid.y, grid.z,
                            block.x, block.y, block.z,
                            0, NULL, args, NULL));
    CHECK_CU(cuCtxSynchronize());
    driver.copy_d2h(h_out, d_out, emb_bytes);
    driver.check(h_out.get(), h_ref.get(), emb_elems, 5e-4f, 5e-4f,
                 "timestep_emb_batch (--use_fast_math)");

    printf("\nPerformance (avg of 200 runs, 5 warmup):\n");
    float ms = driver.benchmark_kernel(fn, grid, block, 0, args, 5, 200);
    double elems_per_sec = (double)emb_elems / (ms / 1000.0);
    double gb_per_sec    = (double)emb_elems * sizeof(float) / 1e9 / (ms / 1000.0);
    printf("  %-40s %6.3f ms   %8.2f M emb/s   %6.2f GB/s\n",
           "timestep_emb_batch", ms, elems_per_sec / 1e6, gb_per_sec);
    return 0;
}
