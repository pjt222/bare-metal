/*
 * bench_implicit_gemm.cu — Implicit GEMM vs explicit im2col + GEMM for conv2d
 *
 * Tests the hypothesis: eliminating the 23.6 MB col buffer saves DRAM
 * traffic and speeds up small-channel and large-spatial convolutions.
 *
 * Kernels compared:
 *   Explicit:   im2col_nhwc_fp16 (writes col buffer) + wmma_gemm_conv (reads col buffer)
 *   Implicit:   implicit_gemm_conv (single kernel, no col buffer, indices on-the-fly)
 *
 * Build:
 *   nvcc -arch=sm_86 -O2 -o bench_implicit_gemm bench_implicit_gemm.cu \
 *        -lcuda -I../../phase2/common
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda.h>
#include <cuda_fp16.h>

#include "../../phase2/common/bench.h"
#include "../../phase2/common/check.h"

// ---- Tile constants (must match kernel files) ----
#define BLOCK_M    64
#define BLOCK_N    64
#define BLOCK_K    16
#define WMMA_M     16
#define WMMA_N     16

// Explicit im2col smem: smem_A + smem_B (padded)
#define SMEM_A_STRIDE_EXPLICIT  (BLOCK_K + 8)
#define SMEM_B_STRIDE_EXPLICIT  (BLOCK_N + 8)
#define SMEM_BYTES_EXPLICIT     ((BLOCK_M * SMEM_A_STRIDE_EXPLICIT + BLOCK_K * SMEM_B_STRIDE_EXPLICIT) * 2)
// = (64*24 + 16*72) * 2 = (1536 + 1152) * 2 = 5376 bytes

// Implicit GEMM smem: smem_A + smem_B + coordinate tables
// smem_A: 64*24 halfs = 3072 B
// smem_B: 16*72 halfs = 2304 B
// M tables: 3*64 ints = 768 B
// K tables: 3*16 ints = 192 B
// Total: 6336 bytes
#define SMEM_BYTES_IMPLICIT     (SMEM_BYTES_EXPLICIT + (3 * BLOCK_M + 3 * BLOCK_K) * 4)

// -----------------------------------------------------------------------
// CPU reference: 3×3 NHWC convolution
// -----------------------------------------------------------------------
static void cpu_conv2d_nhwc_ref(
    const float *X, const float *W, float *Y,
    int N, int H, int W_dim, int Cin, int Cout
) {
    memset(Y, 0, (size_t)N * H * W_dim * Cout * sizeof(float));
    for (int n = 0; n < N; n++)
    for (int h = 0; h < H; h++)
    for (int w = 0; w < W_dim; w++)
    for (int cout_c = 0; cout_c < Cout; cout_c++) {
        double acc = 0.0;
        for (int kh = 0; kh < 3; kh++)
        for (int kw = 0; kw < 3; kw++) {
            int h_in = h + kh - 1, w_in = w + kw - 1;
            if (h_in < 0 || h_in >= H || w_in < 0 || w_in >= W_dim) continue;
            for (int cin = 0; cin < Cin; cin++) {
                size_t x_idx = ((size_t)n * H * W_dim + h_in * W_dim + w_in) * Cin + cin;
                size_t w_idx = (size_t)cout_c * 9 * Cin + (kh * 3 + kw) * Cin + cin;
                acc += (double)X[x_idx] * W[w_idx];
            }
        }
        size_t y_idx = ((size_t)n * H * W_dim + h * W_dim + w) * Cout + cout_c;
        Y[y_idx] = (float)acc;
    }
}

static unsigned short fp32_to_fp16_bits(float val) {
    __half h = __float2half(val);
    unsigned short bits;
    memcpy(&bits, &h, 2);
    return bits;
}

// Reshape: W_direct[Cout, kH, kW, Cin] FP32 → W_t[Cin*kH*kW, Cout] FP16
static void reshape_weights_to_col(
    const float *W_direct, unsigned short *W_t,
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
            W_t[(size_t)k * Cout + cout_c] = fp32_to_fp16_bits(val);
        }
    }
}

static void fill_rand(float *arr, size_t n, float scale = 0.1f) {
    for (size_t i = 0; i < n; i++)
        arr[i] = scale * (2.0f * (float)rand() / RAND_MAX - 1.0f);
}

int main(void) {
    // Test configurations: (N, H, W, Cin, Cout, label)
    struct Config {
        int N, H, W_dim, Cin, Cout;
        const char *label;
    } configs[] = {
        { 1, 64, 64, 320, 320, "SD 64×64  Cin=Cout=320  (baseline, large col buffer)" },
        { 1, 32, 32, 640, 640, "SD 32×32  Cin=Cout=640  (smaller spatial, larger channels)" },
        { 1,128,128, 160, 160, "SD 128×128 Cin=Cout=160 (large spatial, small col buffer penalty)"},
        { 1, 64, 64,  64,  64, "Small     64×64  Cin=Cout=64  (index decode overhead visible)"},
    };
    int num_configs = (int)(sizeof(configs) / sizeof(configs[0]));

    CHECK_CU(cuInit(0));
    CUdevice  cu_dev; CHECK_CU(cuDeviceGet(&cu_dev, 0));
    CUcontext cu_ctx; CHECK_CU(cuCtxCreate(&cu_ctx, 0, cu_dev));

    CUmodule mod_explicit, mod_implicit;
    CHECK_CU(cuModuleLoad(&mod_explicit, "conv2d_im2col.sm_86.cubin"));
    CHECK_CU(cuModuleLoad(&mod_implicit, "conv2d_implicit_gemm.sm_86.cubin"));

    CUfunction fn_im2col, fn_gemm_explicit, fn_gemm_implicit;
    CHECK_CU(cuModuleGetFunction(&fn_im2col,       mod_explicit, "im2col_nhwc_fp16"));
    CHECK_CU(cuModuleGetFunction(&fn_gemm_explicit, mod_explicit, "wmma_gemm_conv"));
    CHECK_CU(cuModuleGetFunction(&fn_gemm_implicit, mod_implicit, "implicit_gemm_conv"));

    CHECK_CU(cuFuncSetAttribute(fn_gemm_explicit, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, SMEM_BYTES_EXPLICIT));
    CHECK_CU(cuFuncSetAttribute(fn_gemm_implicit, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, SMEM_BYTES_IMPLICIT));

    printf("=== Implicit GEMM vs Explicit im2col + GEMM ===\n");
    printf("  Explicit: im2col_nhwc_fp16 → col buffer → wmma_gemm_conv\n");
    printf("  Implicit: implicit_gemm_conv (indices on-the-fly, no col buffer)\n\n");

    for (int ci = 0; ci < num_configs; ci++) {
        int N     = configs[ci].N;
        int H     = configs[ci].H;
        int W_dim = configs[ci].W_dim;
        int Cin   = configs[ci].Cin;
        int Cout  = configs[ci].Cout;

        int kH = 3, kW = 3, pad = 1;
        int out_H = H, out_W = W_dim;
        int M_dim = N * out_H * out_W;
        int K_dim = Cin * kH * kW;

        float col_mb   = (float)M_dim * K_dim * 2 / 1e6f;
        double flops_g = 2.0 * M_dim * K_dim * Cout / 1e9;

        printf("--- %s ---\n", configs[ci].label);
        printf("  M=%d K=%d Cout=%d  col_buf=%.1f MB  GFLOPs=%.2f\n",
               M_dim, K_dim, Cout, col_mb, flops_g);

        if (M_dim % 16 != 0 || K_dim % 16 != 0 || Cout % 16 != 0) {
            printf("  SKIP: M, K, or Cout not divisible by 16 (WMMA constraint)\n\n");
            continue;
        }

        // ---- Allocate host memory ----
        size_t X_elems   = (size_t)N * H * W_dim * Cin;
        size_t W_elems   = (size_t)Cout * kH * kW * Cin;
        size_t Y_elems   = (size_t)M_dim * Cout;
        size_t col_elems = (size_t)M_dim * K_dim;
        size_t Wt_elems  = (size_t)K_dim * Cout;

        float          *host_X     = new float[X_elems];
        float          *host_W     = new float[W_elems];
        float          *host_Y_ref = new float[Y_elems];
        float          *host_Y_exp = new float[Y_elems];
        float          *host_Y_imp = new float[Y_elems];
        unsigned short *host_Wt    = new unsigned short[Wt_elems];

        srand(42 + ci);
        fill_rand(host_X, X_elems, 0.1f);
        fill_rand(host_W, W_elems, 0.1f / sqrtf((float)(Cin * kH * kW)));
        reshape_weights_to_col(host_W, host_Wt, Cout, Cin, kH, kW);

        // CPU reference (skip for large configs)
        bool ran_cpu = (X_elems < 16 * 1024 * 1024 / 4);  // skip if X > 16 MB
        if (ran_cpu) {
            cpu_conv2d_nhwc_ref(host_X, host_W, host_Y_ref, N, H, W_dim, Cin, Cout);
        }

        // ---- Device allocations ----
        CUdeviceptr d_X, d_col, d_Wt, d_Y;
        CHECK_CU(cuMemAlloc(&d_X,   X_elems   * sizeof(float)));
        CHECK_CU(cuMemAlloc(&d_col, col_elems * sizeof(unsigned short)));
        CHECK_CU(cuMemAlloc(&d_Wt,  Wt_elems  * sizeof(unsigned short)));
        CHECK_CU(cuMemAlloc(&d_Y,   Y_elems   * sizeof(float)));

        CHECK_CU(cuMemcpyHtoD(d_X,  host_X,  X_elems  * sizeof(float)));
        CHECK_CU(cuMemcpyHtoD(d_Wt, host_Wt, Wt_elems * sizeof(unsigned short)));

        int grid_m = (M_dim + BLOCK_M - 1) / BLOCK_M;
        int grid_n = (Cout  + BLOCK_N - 1) / BLOCK_N;

        int im2col_threads = 256;
        int im2col_blocks  = (int)((col_elems + im2col_threads - 1) / im2col_threads);
        if (im2col_blocks > 65535) im2col_blocks = 65535;

        void *im2col_args[] = { &d_X, &d_col, &N, &H, &W_dim, &Cin, &kH, &kW, &pad, &out_H, &out_W };
        void *gemm_exp_args[]= { &d_col, &d_Wt, &d_Y, &M_dim, &K_dim, &Cout };
        void *gemm_imp_args[]= { &d_X, &d_Wt, &d_Y,
                                  &N, &H, &W_dim, &Cin,
                                  &kH, &kW, &pad,
                                  &out_H, &out_W,
                                  &M_dim, &K_dim, &Cout };

        // ---- Correctness: explicit ----
        CHECK_CU(cuMemsetD32(d_Y, 0, Y_elems));
        CHECK_CU(cuLaunchKernel(fn_im2col,
                                im2col_blocks, 1, 1, im2col_threads, 1, 1,
                                0, 0, im2col_args, 0));
        CHECK_CU(cuLaunchKernel(fn_gemm_explicit,
                                grid_m, grid_n, 1, 128, 1, 1,
                                SMEM_BYTES_EXPLICIT, 0, gemm_exp_args, 0));
        CHECK_CU(cuCtxSynchronize());
        CHECK_CU(cuMemcpyDtoH(host_Y_exp, d_Y, Y_elems * sizeof(float)));

        // ---- Correctness: implicit ----
        CHECK_CU(cuMemsetD32(d_Y, 0, Y_elems));
        CHECK_CU(cuLaunchKernel(fn_gemm_implicit,
                                grid_m, grid_n, 1, 128, 1, 1,
                                SMEM_BYTES_IMPLICIT, 0, gemm_imp_args, 0));
        CHECK_CU(cuCtxSynchronize());
        CHECK_CU(cuMemcpyDtoH(host_Y_imp, d_Y, Y_elems * sizeof(float)));

        // Compare implicit vs explicit
        float max_abs_vs_exp = 0.0f;
        for (size_t i = 0; i < Y_elems; i++)
            max_abs_vs_exp = fmaxf(max_abs_vs_exp, fabsf(host_Y_imp[i] - host_Y_exp[i]));
        printf("  vs explicit: %s (max_abs=%.2e)\n",
               max_abs_vs_exp < 1e-4f ? "PASS" : "FAIL", (double)max_abs_vs_exp);

        if (ran_cpu) {
            float max_abs_vs_ref = 0.0f;
            for (size_t i = 0; i < Y_elems; i++)
                max_abs_vs_ref = fmaxf(max_abs_vs_ref, fabsf(host_Y_imp[i] - host_Y_ref[i]));
            printf("  vs CPU ref:  max_abs=%.2e\n", (double)max_abs_vs_ref);
        }

        // ---- Timing: explicit (im2col + GEMM combined) ----
        float ms_explicit = 0.0f;
        {
            for (int t = 0; t < 20; t++) {
                CHECK_CU(cuLaunchKernel(fn_im2col, im2col_blocks,1,1, im2col_threads,1,1, 0,0, im2col_args,0));
                CHECK_CU(cuLaunchKernel(fn_gemm_explicit, grid_m, grid_n,1, 128,1,1, SMEM_BYTES_EXPLICIT,0, gemm_exp_args,0));
            }
            CHECK_CU(cuCtxSynchronize());
            BenchTimer bt; bt.start();
            for (int t = 0; t < 100; t++) {
                CHECK_CU(cuLaunchKernel(fn_im2col, im2col_blocks,1,1, im2col_threads,1,1, 0,0, im2col_args,0));
                CHECK_CU(cuLaunchKernel(fn_gemm_explicit, grid_m, grid_n,1, 128,1,1, SMEM_BYTES_EXPLICIT,0, gemm_exp_args,0));
            }
            ms_explicit = bt.stop_ms() / 100.0f;
        }

        // ---- Timing: implicit GEMM (single kernel) ----
        float ms_implicit = 0.0f;
        {
            for (int t = 0; t < 20; t++)
                CHECK_CU(cuLaunchKernel(fn_gemm_implicit, grid_m, grid_n,1, 128,1,1, SMEM_BYTES_IMPLICIT,0, gemm_imp_args,0));
            CHECK_CU(cuCtxSynchronize());
            BenchTimer bt; bt.start();
            for (int t = 0; t < 100; t++)
                CHECK_CU(cuLaunchKernel(fn_gemm_implicit, grid_m, grid_n,1, 128,1,1, SMEM_BYTES_IMPLICIT,0, gemm_imp_args,0));
            ms_implicit = bt.stop_ms() / 100.0f;
        }

        double gflops_exp = flops_g / (ms_explicit * 1e-3);
        double gflops_imp = flops_g / (ms_implicit * 1e-3);

        printf("  Explicit (im2col+GEMM): %.3f ms  → %6.0f GFLOPS\n", ms_explicit, gflops_exp);
        printf("  Implicit (single kern): %.3f ms  → %6.0f GFLOPS  (%.2f× speedup)\n\n",
               ms_implicit, gflops_imp, ms_explicit / ms_implicit);

        cuMemFree(d_X); cuMemFree(d_col); cuMemFree(d_Wt); cuMemFree(d_Y);
        delete[] host_X; delete[] host_W; delete[] host_Y_ref;
        delete[] host_Y_exp; delete[] host_Y_imp; delete[] host_Wt;
    }

    cuModuleUnload(mod_explicit);
    cuModuleUnload(mod_implicit);
    cuCtxDestroy(cu_ctx);
    return 0;
}
