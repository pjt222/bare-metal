/*
 * test_mma_sp.cu — Minimal mma.sp verification (no ldmatrix, constant operands)
 *
 * Tests the mma.sp instruction in isolation:
 *   A = all 1.0 (packed FP16), B = all 1.0, meta = 0x44444444
 *   Expected: every output element = 8.0
 *   (k=16 logical, 4 groups of 4, 2 nonzero per group → 8 multiply-adds × 1.0 × 1.0 = 8.0)
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o test_mma_sp.sm_86.cubin test_mma_sp.cu
 *   nvcc -arch=sm_86 -O2 -o test_mma_sp test_mma_sp.cu -lcuda
 */

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cuda.h>
#include <cuda_fp16.h>

// One warp, constant operands, outputs 16×8 via one mma.sp.m16n8k16
extern "C" __global__ void test_sparse_constant(float *output) {
    int lane = threadIdx.x;

    // All-ones FP16: 1.0 = 0x3C00. Two packed = 0x3C003C00.
    uint32_t fa0 = 0x3C003C00;  // A sparse fragment: 2 regs × (2 FP16)
    uint32_t fa1 = 0x3C003C00;

    uint32_t fb0 = 0x3C003C00;  // B dense fragment: 2 regs × (2 FP16)
    uint32_t fb1 = 0x3C003C00;

    uint32_t meta = 0x44444444; // positions {0,1} per group

    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;

    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11}, %12, 0x0;\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(fa0), "r"(fa1), "r"(fb0), "r"(fb1),
          "f"(d0), "f"(d1), "f"(d2), "f"(d3),
          "r"(meta));

    // Store output — PTX layout
    int groupID = lane >> 2;
    int row_even = (groupID < 4) ? (groupID * 2) : ((groupID - 4) * 2 + 8);
    int row_odd = row_even + 1;
    int col0 = (lane & 3) * 2;
    int col1 = col0 + 1;

    output[row_even * 8 + col0] = d0;
    output[row_even * 8 + col1] = d1;
    output[row_odd  * 8 + col0] = d2;
    output[row_odd  * 8 + col1] = d3;
}

int main() {
    printf("=== mma.sp constant-operand test ===\n\n");

    CUresult err = cuInit(0);
    CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuCtxCreate(&ctx, 0, dev);

    CUmodule mod;
    if (cuModuleLoad(&mod, "test_mma_sp.sm_86.cubin") != CUDA_SUCCESS) {
        fprintf(stderr, "Build first: nvcc --cubin -arch=sm_86 -O2 -o test_mma_sp.sm_86.cubin test_mma_sp.cu\n");
        return 1;
    }
    CUfunction fn;
    cuModuleGetFunction(&fn, mod, "test_sparse_constant");

    CUdeviceptr d_out;
    cuMemAlloc(&d_out, 16 * 8 * sizeof(float));
    cuMemsetD32(d_out, 0, 16 * 8);

    void *args[] = { &d_out };
    cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, NULL, args, NULL);
    cuCtxSynchronize();

    float host_out[16 * 8];
    cuMemcpyDtoH(host_out, d_out, sizeof(host_out));

    printf("Output (16×8, expected all 8.0):\n");
    int errors = 0;
    for (int r = 0; r < 16; r++) {
        printf("  row %2d:", r);
        for (int c = 0; c < 8; c++) {
            float v = host_out[r * 8 + c];
            printf(" %5.1f", v);
            if (fabsf(v - 8.0f) > 0.01f) errors++;
        }
        printf("\n");
    }
    printf("\n%s (%d/128 wrong)\n", errors == 0 ? "PASS" : "FAIL", errors);

    cuMemFree(d_out);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);
    return 0;
}
