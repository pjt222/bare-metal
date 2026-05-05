/*
 * bench.cu — Timestep Embedding benchmark: correctness + throughput
 *
 * Build:
 *   nvcc -arch=sm_86 -O2 -o bench bench.cu -lcuda -I../../phase2/common
 *
 * Usage:
 *   ./bench              # default: d_model=512, batch=1024
 *   ./bench 320 2        # d_model=320 (SD), batch=2 (CFG inference)
 *   ./bench 1024 512     # d_model=1024, batch=512 (training)
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda.h>

#include "../../phase2/common/bench.h"
#include "../../phase2/common/check.h"

#define LOG_MAX_PERIOD  9.210340371976183f
#define LOG2E           1.4426950408889634f

// CPU reference: exact sinusoidal embedding using standard math
static void cpu_timestep_emb(
    const float *timesteps, float *output,
    int batch_size, int d_model
) {
    int half_dim = d_model / 2;
    for (int b = 0; b < batch_size; b++) {
        float t = timesteps[b];
        float *out_row = output + (size_t)b * d_model;
        for (int i = 0; i < half_dim; i++) {
            float freq = expf(-LOG_MAX_PERIOD * (float)i / (float)half_dim);
            float angle = t * freq;
            out_row[i]            = sinf(angle);
            out_row[i + half_dim] = cosf(angle);
        }
    }
}

int main(int argc, char **argv) {
    int d_model    = (argc > 1) ? atoi(argv[1]) : 512;
    int batch_size = (argc > 2) ? atoi(argv[2]) : 1024;

    if (d_model % 2 != 0) {
        fprintf(stderr, "d_model must be even\n");
        return 1;
    }
    int half_dim = d_model / 2;

    printf("=== Timestep Embedding — MUFU.SIN + MUFU.COS + MUFU.EX2 ===\n");
    printf("d_model=%d  batch=%d  (SASS: --use_fast_math required)\n\n",
           d_model, batch_size);

    CHECK_CU(cuInit(0));
    CUdevice cu_dev; CHECK_CU(cuDeviceGet(&cu_dev, 0));
    char devname[256]; CHECK_CU(cuDeviceGetName(devname, sizeof(devname), cu_dev));
    printf("Device: %s\n\n", devname);

    CUcontext ctx; CHECK_CU(cuCtxCreate(&ctx, 0, cu_dev));

    CUmodule mod; CUfunction fn_batch, fn_single;
    if (cuModuleLoad(&mod, "timestep_emb.sm_86.cubin") != CUDA_SUCCESS) {
        fprintf(stderr, "Cannot load timestep_emb.sm_86.cubin\n");
        return 1;
    }
    CHECK_CU(cuModuleGetFunction(&fn_batch,  mod, "timestep_emb_batch"));
    CHECK_CU(cuModuleGetFunction(&fn_single, mod, "timestep_emb_single"));
    printf("Kernels loaded.\n\n");

    // Host buffers
    size_t emb_elements = (size_t)batch_size * d_model;
    float *host_ts  = (float*)malloc(batch_size * sizeof(float));
    float *host_out = (float*)malloc(emb_elements * sizeof(float));
    float *host_ref = (float*)malloc(emb_elements * sizeof(float));

    // Generate random timesteps in [0, 1000)
    for (int i = 0; i < batch_size; i++) {
        host_ts[i] = (float)(rand() % 1000) + (float)rand() / RAND_MAX;
    }

    // CPU reference
    cpu_timestep_emb(host_ts, host_ref, batch_size, d_model);

    // Device buffers
    CUdeviceptr dev_ts, dev_out;
    CHECK_CU(cuMemAlloc(&dev_ts,  batch_size * sizeof(float)));
    CHECK_CU(cuMemAlloc(&dev_out, emb_elements * sizeof(float)));
    CHECK_CU(cuMemcpyHtoD(dev_ts, host_ts, batch_size * sizeof(float)));

    // Correctness
    printf("Correctness:\n");

    void *args_batch[] = { &dev_ts, &dev_out, &d_model, &batch_size };
    CHECK_CU(cuMemsetD32(dev_out, 0, emb_elements));
    CHECK_CU(cuLaunchKernel(fn_batch,
        batch_size, 1, 1,   // grid: one block per timestep
        half_dim,  1, 1,    // block: one thread per (sin,cos) pair
        0, NULL, args_batch, NULL));
    CHECK_CU(cuCtxSynchronize());
    CHECK_CU(cuMemcpyDtoH(host_out, dev_out, emb_elements * sizeof(float)));

    // fast_math sin/cos (MUFU.SIN/COS) has ~2 ULP error vs libm reference.
    // For values near ±1, 2 ULP ≈ 2.4e-7; but argument-reduction error for large angles
    // can push max_abs to ~5e-4. Use 5e-4 for both thresholds.
    auto correctness = check_fp32(host_out, host_ref, emb_elements, 5e-4f, 5e-4f);
    print_check_result("timestep_emb_batch (--use_fast_math)", correctness);

    printf("\nPerformance:\n");
    int warmup = 5, bench_iters = 200;

    for (int i = 0; i < warmup; i++) {
        CHECK_CU(cuLaunchKernel(fn_batch,
            batch_size, 1, 1, half_dim, 1, 1, 0, NULL, args_batch, NULL));
    }
    CHECK_CU(cuCtxSynchronize());

    float avg_ms;
    {
        BenchTimer timer;
        timer.start();
        for (int i = 0; i < bench_iters; i++) {
            CHECK_CU(cuLaunchKernel(fn_batch,
                batch_size, 1, 1, half_dim, 1, 1, 0, NULL, args_batch, NULL));
        }
        avg_ms = timer.stop_ms() / bench_iters;
    }

    // Throughput: batch_size × d_model float output values
    double elems_per_sec = (double)emb_elements / (avg_ms / 1000.0);
    double gb_per_sec    = (double)emb_elements * sizeof(float) / 1e9 / (avg_ms / 1000.0);

    printf("  %-40s %6.3f ms   %8.2f M emb/s   %6.2f GB/s\n",
           "timestep_emb_batch", avg_ms, elems_per_sec / 1e6, gb_per_sec);
    printf("\nNote: kernel is compute-bound (MUFU units) not memory-bound.\n");
    printf("      Each element requires: EX2 + SIN or COS = ~3 MUFU calls.\n");
    printf("      At ~128 MUFU ops/cycle/SM × 46 SMs = 5888 ops/cycle.\n");

    printf("\nSASS inspection:\n");
    printf("  cuobjdump -sass timestep_emb.sm_86.cubin | grep MUFU\n");
    printf("  → MUFU.SIN  (sinf with --use_fast_math)\n");
    printf("  → MUFU.COS  (cosf with --use_fast_math)\n");
    printf("  → MUFU.EX2  (exp2f for frequency computation)\n");
    printf("  NOTE: without --use_fast_math, sinf emits polynomial FFMA sequence!\n");

    cuMemFree(dev_ts); cuMemFree(dev_out);
    cuModuleUnload(mod); cuCtxDestroy(ctx);
    free(host_ts); free(host_out); free(host_ref);
    return 0;
}
