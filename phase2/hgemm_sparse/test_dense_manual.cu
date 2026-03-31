/*
 * test_dense_manual.cu — Bridge WMMA→PTX fragment layout for mma.sync.m16n8k16
 *
 * Two paths compute the same C = A[16×16] × B[16×8]:
 *   Path 1 (WMMA): load_matrix_sync → mma_sync → store_matrix_sync (known correct)
 *   Path 2 (PTX):  Extract raw uint32_t from WMMA fragments → PTX mma.sync.m16n8k16
 *
 * If both paths match, WMMA and PTX share the same physical register layout.
 * The register mapping discovered by verify_wmma_ab_layout.cu:
 *
 *   matrix_a (row_major, m16n16k16): 8 uint32_t per thread, regs 4-7 = dup of 0-3
 *     reg 0: {A[gid][tid*2],     A[gid][tid*2+1]}       K-left,  row gid
 *     reg 1: {A[gid+8][tid*2],   A[gid+8][tid*2+1]}     K-left,  row gid+8
 *     reg 2: {A[gid][tid*2+8],   A[gid][tid*2+9]}       K-right, row gid
 *     reg 3: {A[gid+8][tid*2+8], A[gid+8][tid*2+9]}     K-right, row gid+8
 *
 *   matrix_b (row_major, m16n16k16): 8 uint32_t, regs 4-7 = dup of 0-3
 *     reg 0: {B[tid*2][gid],     B[tid*2+1][gid]}       K-left,  N col gid     (left sub-tile)
 *     reg 1: {B[tid*2+8][gid],   B[tid*2+9][gid]}       K-right, N col gid     (left sub-tile)
 *     reg 2: {B[tid*2][gid+8],   B[tid*2+1][gid+8]}     K-left,  N col gid+8   (right sub-tile)
 *     reg 3: {B[tid*2+8][gid+8], B[tid*2+9][gid+8]}     K-right, N col gid+8   (right sub-tile)
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o test_dense_manual.sm_86.cubin test_dense_manual.cu
 *   nvcc -arch=sm_86 -O2 -o test_dense_manual test_dense_manual.cu -lcuda -I../common
 */

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "../common/bench.h"
#include "../common/check.h"

// Dense mma.sync bridging test: WMMA fragments → PTX mma.sync.m16n8k16
extern "C" __global__ void dense_manual_m16n8(
    const __half * __restrict__ A,      // [16 × 16] row-major
    const __half * __restrict__ B,      // [16 × 8]  row-major
    float        * __restrict__ C_wmma, // [16 × 16] WMMA output (first 8 cols valid)
    float        * __restrict__ C_ptx   // [16 × 8]  PTX output
) {
    int lane = threadIdx.x;
    __shared__ __align__(16) __half smem_a[16 * 16];
    __shared__ __align__(16) __half smem_b[16 * 8];

    // Load to smem
    for (int i = lane; i < 256; i += 32) smem_a[i] = A[i];
    for (int i = lane; i < 128; i += 32) smem_b[i] = B[i];
    __syncwarp();

    // ---- Path 1: WMMA reference ----
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> frag_a;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::row_major> frag_b;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> frag_c;
    nvcuda::wmma::fill_fragment(frag_c, 0.0f);

    // WMMA needs B[16×16] but we only have B[16×8]. Pad to 16×16.
    __shared__ __half smem_b16[16 * 16];
    for (int i = lane; i < 256; i += 32) {
        int r = i / 16, c = i % 16;
        smem_b16[i] = (c < 8) ? smem_b[r * 8 + c] : __float2half(0.0f);
    }
    __syncwarp();

    nvcuda::wmma::load_matrix_sync(frag_a, smem_a, 16);
    nvcuda::wmma::load_matrix_sync(frag_b, smem_b16, 16);
    nvcuda::wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
    nvcuda::wmma::store_matrix_sync(C_wmma, frag_c, 16, nvcuda::wmma::mem_row_major);

    // ---- Path 2: PTX using WMMA-extracted registers ----
    // Extract raw uint32_t from WMMA fragments
    uint32_t *a_regs = reinterpret_cast<uint32_t*>(&frag_a.x[0]);  // 8 regs, 0-3 unique
    uint32_t *b_regs = reinterpret_cast<uint32_t*>(&frag_b.x[0]);  // 8 regs, 0-3 unique

    // PTX mma.sync.m16n8k16 for left sub-tile (N=0..7):
    //   A: 4 regs = a_regs[0..3]
    //   B: 2 regs = b_regs[0..1] (left sub-tile: N col = gid, K-left and K-right)
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
          "r"(b_regs[0]), "r"(b_regs[1]),
          "f"(d0), "f"(d1), "f"(d2), "f"(d3));

    // Store PTX result using PTX accumulator layout:
    //   gid = lane >> 2, tid = lane & 3
    //   row_lo = gid, row_hi = gid + 8   (actual sm_86 layout, NOT PTX docs!)
    //   col0 = tid * 2, col1 = tid * 2 + 1
    //   d0 = C[row_lo][col0], d1 = C[row_lo][col1]
    //   d2 = C[row_hi][col0], d3 = C[row_hi][col1]
    //
    // Wait — the accumulator output layout might differ from what the probe showed
    // for input fragments. The accumulator layout was verified in verify_wmma_layout.cu
    // as (gid, gid+8). But the PTX mma.sync instruction's OUTPUT registers might use
    // a different convention. test_mma_sp.cu uses the PTX-documented layout for the
    // accumulator store and it works for constant inputs. Let's try BOTH layouts.

    int gid = lane >> 2;
    int tid = lane & 3;

    // Try the actual (gid, gid+8) layout discovered by our probes:
    int row_lo = gid;
    int row_hi = gid + 8;
    int col0 = tid * 2;
    int col1 = col0 + 1;

    C_ptx[row_lo * 8 + col0] = d0;
    C_ptx[row_lo * 8 + col1] = d1;
    C_ptx[row_hi * 8 + col0] = d2;
    C_ptx[row_hi * 8 + col1] = d3;
}

int main() {
    printf("=== Dense mma.sync WMMA→PTX bridging test ===\n\n");

    CUresult err = cuInit(0);
    CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuCtxCreate(&ctx, 0, dev);

    CUmodule mod;
    if (cuModuleLoad(&mod, "test_dense_manual.sm_86.cubin") != CUDA_SUCCESS) {
        fprintf(stderr, "Build cubin first\n"); return 1;
    }
    CUfunction fn;
    cuModuleGetFunction(&fn, mod, "dense_manual_m16n8");

    // Create test matrices: A[16×16], B[16×8]
    float host_a_f32[256], host_b_f32[128];
    fill_random(host_a_f32, 256, 42);
    fill_random(host_b_f32, 128, 137);

    __half host_a[256], host_b[128];
    for (int i = 0; i < 256; i++) host_a[i] = __float2half(host_a_f32[i]);
    for (int i = 0; i < 128; i++) host_b[i] = __float2half(host_b_f32[i]);

    // CPU reference: C = A × B (FP32 accumulation, FP16 inputs)
    float host_ref[128] = {0};
    for (int m = 0; m < 16; m++)
        for (int n = 0; n < 8; n++)
            for (int k = 0; k < 16; k++)
                host_ref[m*8+n] += (float)host_a[m*16+k] * (float)host_b[k*8+n];

    CUdeviceptr d_a, d_b, d_c_wmma, d_c_ptx;
    cuMemAlloc(&d_a, 256 * sizeof(__half));
    cuMemAlloc(&d_b, 128 * sizeof(__half));
    cuMemAlloc(&d_c_wmma, 256 * sizeof(float));  // 16×16 for WMMA
    cuMemAlloc(&d_c_ptx, 128 * sizeof(float));   // 16×8 for PTX
    cuMemcpyHtoD(d_a, host_a, 256 * sizeof(__half));
    cuMemcpyHtoD(d_b, host_b, 128 * sizeof(__half));
    cuMemsetD32(d_c_wmma, 0, 256);
    cuMemsetD32(d_c_ptx, 0, 128);

    int smem_needed = (16*16 + 16*8 + 16*16) * sizeof(__half);

    void *args[] = { &d_a, &d_b, &d_c_wmma, &d_c_ptx };
    cuLaunchKernel(fn, 1,1,1, 32,1,1, smem_needed, NULL, args, NULL);
    cuCtxSynchronize();

    // Extract [16×8] from [16×16] WMMA output
    float host_wmma_full[256];
    cuMemcpyDtoH(host_wmma_full, d_c_wmma, sizeof(host_wmma_full));
    float host_wmma[128];
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 8; c++)
            host_wmma[r*8+c] = host_wmma_full[r*16+c];

    float host_ptx[128];
    cuMemcpyDtoH(host_ptx, d_c_ptx, sizeof(host_ptx));

    // Check WMMA vs CPU reference
    auto result_wmma = check_fp32(host_wmma, host_ref, 128, 5e-2f, 5e-2f);
    print_check_result("WMMA m16n16k16 (reference)", result_wmma);

    // Check PTX vs CPU reference
    auto result_ptx = check_fp32(host_ptx, host_ref, 128, 5e-2f, 5e-2f);
    print_check_result("PTX m16n8k16 (WMMA regs, gid/gid+8 accum)", result_ptx);

    // Check PTX vs WMMA (should be bit-identical)
    auto result_bridge = check_fp32(host_ptx, host_wmma, 128, 0.0f, 0.0f);
    print_check_result("PTX vs WMMA (bit-exact bridge)", result_bridge);

    if (result_ptx.num_errors > 0) {
        printf("\n  Sample outputs (PTX vs REF):\n");
        for (int i = 0; i < 8 && i < 128; i++) {
            printf("    C[%d][%d] PTX=%.4f REF=%.4f WMMA=%.4f\n",
                   i/8, i%8, host_ptx[i], host_ref[i], host_wmma[i]);
        }

        // Also try PTX-documented accumulator layout for comparison
        printf("\n  If PTX accum layout is wrong, try rearranging...\n");
    }

    cuMemFree(d_a); cuMemFree(d_b); cuMemFree(d_c_wmma); cuMemFree(d_c_ptx);
    cuModuleUnload(mod); cuCtxDestroy(ctx);
    return 0;
}
