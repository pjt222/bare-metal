/*
 * bench.cu — IGEMM benchmark: INT8 Tensor Core vs FP16 HGEMM vs theoretical peak
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o igemm.sm_86.cubin igemm.cu
 *   nvcc -arch=sm_86 -O2 -o bench bench.cu -lcuda -I../common
 *
 * Usage:
 *   ./bench [M] [N] [K]
 *   ./bench 4096 4096 4096   # default — large enough to saturate Tensor Cores
 *
 * Expected: ~4× improvement over FP16 HGEMM in throughput (TOPS).
 * Theoretical: 696 TOPS INT8 / 174 TFLOPS FP16 = 4×
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda.h>

#include "../common/bench.h"
#include "../common/check.h"

// -----------------------------------------------------------------------
// Symmetric per-tensor quantization: FP32 → INT8
// -----------------------------------------------------------------------
static float compute_scale(const float *data, int num_elements) {
    float max_abs = 0.0f;
    for (int i = 0; i < num_elements; i++) {
        float abs_val = fabsf(data[i]);
        if (abs_val > max_abs) max_abs = abs_val;
    }
    if (max_abs == 0.0f) return 1.0f;  // avoid division by zero
    return max_abs / 127.0f;
}

static void quantize_symmetric(
    const float *fp32_data,
    signed char *int8_data,
    int num_elements,
    float scale
) {
    float inv_scale = 1.0f / scale;
    for (int i = 0; i < num_elements; i++) {
        int quantized = (int)roundf(fp32_data[i] * inv_scale);
        if (quantized >  127) quantized =  127;
        if (quantized < -128) quantized = -128;
        int8_data[i] = (signed char)quantized;
    }
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------
int main(int argc, char **argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 4096;
    int N = (argc > 2) ? atoi(argv[2]) : 4096;
    int K = (argc > 3) ? atoi(argv[3]) : 4096;

    // Round to nearest multiple of 16 (WMMA requirement)
    M = (M + 15) / 16 * 16;
    N = (N + 15) / 16 * 16;
    K = (K + 15) / 16 * 16;

    printf("=== IGEMM Benchmark — INT8 Tensor Cores (sm_86) ===\n");
    printf("Matrix: C[%d×%d] = A[%d×%d] * B[%d×%d]  (INT8 in, INT32 accum, FP32 out)\n\n",
           M, N, M, K, K, N);

    CHECK_CU(cuInit(0));
    CUdevice cu_device;
    CHECK_CU(cuDeviceGet(&cu_device, 0));

    char device_name[256];
    CHECK_CU(cuDeviceGetName(device_name, sizeof(device_name), cu_device));
    printf("Device: %s\n\n", device_name);

    CUcontext cu_context;
    CHECK_CU(cuCtxCreate(&cu_context, 0, cu_device));

    // --- Load IGEMM kernel ---
    CUmodule   igemm_module;
    CUfunction igemm_func;
    CUresult load_result = cuModuleLoad(&igemm_module, "igemm.sm_86.cubin");
    if (load_result != CUDA_SUCCESS) {
        const char *err = nullptr;
        cuGetErrorString(load_result, &err);
        fprintf(stderr, "Cannot load igemm.sm_86.cubin: %s\n", err);
        fprintf(stderr, "Build with: nvcc --cubin -arch=sm_86 -O2 -o igemm.sm_86.cubin igemm.cu\n");
        return EXIT_FAILURE;
    }
    CHECK_CU(cuModuleGetFunction(&igemm_func, igemm_module, "igemm_wmma"));

    // --- Load tiled IGEMM kernel ---
    CUmodule   tiled_module = NULL;
    CUfunction tiled_func   = NULL;
    bool have_tiled = (cuModuleLoad(&tiled_module, "igemm_tiled.sm_86.cubin") == CUDA_SUCCESS);
    if (have_tiled) {
        CHECK_CU(cuModuleGetFunction(&tiled_func, tiled_module, "igemm_tiled"));
    }
    // --- Load register-blocked IGEMM kernel ---
    CUmodule   regblk_module = NULL;
    CUfunction regblk_func   = NULL;
    bool have_regblk = (cuModuleLoad(&regblk_module, "igemm_register_blocked.sm_86.cubin") == CUDA_SUCCESS);
    if (have_regblk) {
        CHECK_CU(cuModuleGetFunction(&regblk_func, regblk_module, "igemm_register_blocked"));
    }
    // --- Load hand-tuned tiled IGEMM kernel ---
    CUmodule   handtuned_module = NULL;
    CUfunction handtuned_func   = NULL;
    bool have_handtuned = (cuModuleLoad(&handtuned_module, "igemm_tiled_handtuned.sm_86.cubin") == CUDA_SUCCESS);
    if (have_handtuned) {
        CHECK_CU(cuModuleGetFunction(&handtuned_func, handtuned_module, "igemm_tiled"));
    }
    // --- Load aggressive hand-tuned IGEMM kernel ---
    CUmodule   aggressive_module = NULL;
    CUfunction aggressive_func   = NULL;
    bool have_aggressive = (cuModuleLoad(&aggressive_module, "igemm_tiled_aggressive.sm_86.cubin") == CUDA_SUCCESS);
    if (have_aggressive) {
        CHECK_CU(cuModuleGetFunction(&aggressive_func, aggressive_module, "igemm_tiled"));
    }
    // --- Load software-pipelined IGEMM kernel (LDG double-buffer) ---
    CUmodule   pipelined_module = NULL;
    CUfunction pipelined_func   = NULL;
    bool have_pipelined = (cuModuleLoad(&pipelined_module, "igemm_pipelined.sm_86.cubin") == CUDA_SUCCESS);
    if (have_pipelined) {
        CHECK_CU(cuModuleGetFunction(&pipelined_func, pipelined_module, "igemm_pipelined"));
    }
    // --- Load cp.async pipelined IGEMM kernel ---
    CUmodule   cpasync_module = NULL;
    CUfunction cpasync_func   = NULL;
    bool have_cpasync = (cuModuleLoad(&cpasync_module, "igemm_pipelined_cpasync.sm_86.cubin") == CUDA_SUCCESS);
    if (have_cpasync) {
        CHECK_CU(cuModuleGetFunction(&cpasync_func, cpasync_module, "igemm_pipelined_cpasync"));
    }
    // --- Load cp.async BK=64 IGEMM kernel ---
    CUmodule   cpasync_bk64_module = NULL;
    CUfunction cpasync_bk64_func   = NULL;
    bool have_cpasync_bk64 = (cuModuleLoad(&cpasync_bk64_module, "igemm_pipelined_cpasync_bk64.sm_86.cubin") == CUDA_SUCCESS);
    if (have_cpasync_bk64) {
        CHECK_CU(cuModuleGetFunction(&cpasync_bk64_func, cpasync_bk64_module, "igemm_pipelined_cpasync_bk64"));
    }
    // --- Load 8-warp 128×128 IGEMM kernel ---
    CUmodule   warp8_module = NULL;
    CUfunction warp8_func   = NULL;
    bool have_warp8 = (cuModuleLoad(&warp8_module, "igemm_8warp.sm_86.cubin") == CUDA_SUCCESS);
    if (have_warp8) {
        CHECK_CU(cuModuleGetFunction(&warp8_func, warp8_module, "igemm_8warp"));
    }
    // --- Load 8-warp 128×256 IGEMM kernel ---
    CUmodule   warp8_256_module = NULL;
    CUfunction warp8_256_func   = NULL;
    bool have_warp8_256 = (cuModuleLoad(&warp8_256_module, "igemm_8warp_256.sm_86.cubin") == CUDA_SUCCESS);
    if (have_warp8_256) {
        CHECK_CU(cuModuleGetFunction(&warp8_256_func, warp8_256_module, "igemm_8warp_256"));
    }
    // --- Load 8-warp 256×256 IGEMM kernel ---
    CUmodule   warp8_256x256_module = NULL;
    CUfunction warp8_256x256_func   = NULL;
    bool have_warp8_256x256 = (cuModuleLoad(&warp8_256x256_module, "igemm_8warp_256x256.sm_86.cubin") == CUDA_SUCCESS);
    if (have_warp8_256x256) {
        CHECK_CU(cuModuleGetFunction(&warp8_256x256_func, warp8_256x256_module, "igemm_8warp_256x256"));
    }
    // --- Load 8-warp triple-buffer IGEMM kernel ---
    CUmodule   tribuf_module = NULL;
    CUfunction tribuf_func   = NULL;
    bool have_tribuf = (cuModuleLoad(&tribuf_module, "igemm_8warp_tribuf.sm_86.cubin") == CUDA_SUCCESS);
    if (have_tribuf) {
        CHECK_CU(cuModuleGetFunction(&tribuf_func, tribuf_module, "igemm_8warp_tribuf"));
    }
    // --- Load online-quant FP16→INT8 IGEMM kernel ---
    CUmodule   onlinequant_module = NULL;
    CUfunction onlinequant_func   = NULL;
    bool have_onlinequant = (cuModuleLoad(&onlinequant_module, "igemm_online_quant.sm_86.cubin") == CUDA_SUCCESS);
    if (have_onlinequant) {
        CHECK_CU(cuModuleGetFunction(&onlinequant_func, onlinequant_module, "igemm_online_quant"));
    }
    // --- Load in-place online-quant FP16→INT8 IGEMM kernel ---
    CUmodule   inplace_module = NULL;
    CUfunction inplace_func   = NULL;
    bool have_inplace = (cuModuleLoad(&inplace_module, "igemm_online_quant_inplace.sm_86.cubin") == CUDA_SUCCESS);
    if (have_inplace) {
        CHECK_CU(cuModuleGetFunction(&inplace_func, inplace_module, "igemm_online_quant_inplace"));
    }
    // --- Load HGEMM baseline for FP16 comparison ---
    CUmodule   hgemm_module = NULL;
    CUfunction hgemm_func   = NULL;
    bool have_hgemm = (cuModuleLoad(&hgemm_module, "../hgemm/hgemm.sm_86.cubin") == CUDA_SUCCESS);
    if (have_hgemm) {
        CHECK_CU(cuModuleGetFunction(&hgemm_func, hgemm_module, "hgemm_wmma"));
    }
    // --- Load persistent IGEMM kernel ---
    CUmodule   persistent_module = NULL;
    CUfunction persistent_func   = NULL;
    bool have_persistent = (cuModuleLoad(&persistent_module, "igemm_persistent.sm_86.cubin") == CUDA_SUCCESS);
    if (have_persistent) {
        CHECK_CU(cuModuleGetFunction(&persistent_func, persistent_module, "igemm_persistent"));
    }
    // --- Load per-channel asymmetric IGEMM kernel ---
    CUmodule   perchannel_module = NULL;
    CUfunction perchannel_func   = NULL;
    bool have_perchannel = (cuModuleLoad(&perchannel_module, "igemm_pipelined_cpasync_perchannel.sm_86.cubin") == CUDA_SUCCESS);
    if (have_perchannel) {
        CHECK_CU(cuModuleGetFunction(&perchannel_func, perchannel_module, "igemm_pipelined_cpasync_perchannel"));
    }
    // --- Load bank-conflict-free online-quant IGEMM kernel ---
    CUmodule   bankfree_module = NULL;
    CUfunction bankfree_func   = NULL;
    bool have_bankfree = (cuModuleLoad(&bankfree_module, "igemm_online_quant_bankfree.sm_86.cubin") == CUDA_SUCCESS);
    if (have_bankfree) {
        CHECK_CU(cuModuleGetFunction(&bankfree_func, bankfree_module, "igemm_online_quant_bankfree"));
    }
    // --- Load warp-specialized online-quant IGEMM kernel ---
    CUmodule   warpspec_module = NULL;
    CUfunction warpspec_func   = NULL;
    bool have_warpspec = (cuModuleLoad(&warpspec_module, "igemm_warp_specialized.sm_86.cubin") == CUDA_SUCCESS);
    if (have_warpspec) {
        CHECK_CU(cuModuleGetFunction(&warpspec_func, warpspec_module, "igemm_warp_specialized"));
    }
    printf("IGEMM kernels loaded: naive%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s.\n\n",
           have_tiled ? " + tiled" : "",
           have_regblk ? " + register-blocked" : "",
           have_handtuned ? " + handtuned" : "",
           have_aggressive ? " + aggressive" : "",
           have_pipelined ? " + pipelined" : "",
           have_cpasync ? " + cpasync" : "",
           have_cpasync_bk64 ? " + cpasync_bk64" : "",
           have_perchannel ? " + perchannel" : "",
           have_warp8 ? " + 8warp" : "",
           have_warp8_256 ? " + 8warp256" : "",
           have_tribuf ? " + tribuf" : "",
           have_persistent ? " + persistent" : "",
           have_onlinequant ? " + online-quant" : "",
           have_inplace ? " + inplace" : "",
           have_bankfree ? " + bankfree" : "",
           have_warpspec ? " + warpspec" : "",
           have_hgemm ? " + hgemm" : "");

    // --- Allocate host memory ---
    size_t a_elems = (size_t)M * K;
    size_t b_elems = (size_t)K * N;
    size_t c_elems = (size_t)M * N;

    float       *host_a_fp32 = (float *)      malloc(a_elems * sizeof(float));
    float       *host_b_fp32 = (float *)      malloc(b_elems * sizeof(float));
    float       *host_c_fp32 = (float *)      malloc(c_elems * sizeof(float));
    float       *host_ref    = (float *)      malloc(c_elems * sizeof(float));
    signed char *host_a_int8 = (signed char *)malloc(a_elems);
    signed char *host_b_int8 = (signed char *)malloc(b_elems);

    fill_random(host_a_fp32, a_elems, 42);
    fill_random(host_b_fp32, b_elems, 99);
    fill_zeros(host_ref, c_elems);

    // --- Quantize ---
    float scale_a = compute_scale(host_a_fp32, a_elems);
    float scale_b = compute_scale(host_b_fp32, b_elems);
    quantize_symmetric(host_a_fp32, host_a_int8, a_elems, scale_a);
    quantize_symmetric(host_b_fp32, host_b_int8, b_elems, scale_b);
    printf("Quantization: scale_a=%.6f  scale_b=%.6f\n", scale_a, scale_b);

    // --- Per-channel quantization parameters ---
    float *host_pc_scale = (float *)malloc(N * sizeof(float));
    int   *host_pc_zp    = (int *)  malloc(N * sizeof(int));
    float dequant_scale_sym = scale_a * scale_b;
    srand(123);
    for (int n = 0; n < N; n++) {
        // Vary per-channel scale around the symmetric value (±20%)
        host_pc_scale[n] = dequant_scale_sym * (0.8f + 0.4f * ((float)rand() / RAND_MAX));
        // Small random zero-points
        host_pc_zp[n] = (rand() % 5) - 2;  // range [-2, 2]
    }

    // CPU FP32 reference (only for small matrices)
    bool run_cpu_ref = (M <= 512);
    if (run_cpu_ref) {
        printf("Computing CPU reference...\n");
        cpu_sgemm(M, N, K, 1.0f, host_a_fp32, K, host_b_fp32, N, 0.0f, host_ref, N);
    } else {
        printf("CPU reference skipped (matrix too large — use M<=512 for correctness check)\n");
    }

    // --- Allocate device memory ---
    CUdeviceptr dev_a_int8, dev_b_int8, dev_c;
    CHECK_CU(cuMemAlloc(&dev_a_int8, a_elems));           // 1 byte per int8
    CHECK_CU(cuMemAlloc(&dev_b_int8, b_elems));
    CHECK_CU(cuMemAlloc(&dev_c,      c_elems * sizeof(float)));

    CHECK_CU(cuMemcpyHtoD(dev_a_int8, host_a_int8, a_elems));
    CHECK_CU(cuMemcpyHtoD(dev_b_int8, host_b_int8, b_elems));

    CUdeviceptr dev_pc_scale_bench, dev_pc_zp_bench;
    CHECK_CU(cuMemAlloc(&dev_pc_scale_bench, N * sizeof(float)));
    CHECK_CU(cuMemAlloc(&dev_pc_zp_bench,    N * sizeof(int)));
    CHECK_CU(cuMemcpyHtoD(dev_pc_scale_bench, host_pc_scale, N * sizeof(float)));
    CHECK_CU(cuMemcpyHtoD(dev_pc_zp_bench,    host_pc_zp,    N * sizeof(int)));

    // --- FP16 device memory for online-quant and HGEMM ---
    CUdeviceptr dev_a_fp16 = 0, dev_b_fp16 = 0;
    if (have_onlinequant || have_hgemm) {
        // Convert FP32 host data to FP16 and upload
        unsigned short *host_a_fp16 = (unsigned short *)malloc(a_elems * sizeof(unsigned short));
        unsigned short *host_b_fp16 = (unsigned short *)malloc(b_elems * sizeof(unsigned short));
        for (size_t i = 0; i < a_elems; i++) {
            // Simple FP32→FP16 via half intrinsics at runtime
            float v = host_a_fp32[i];
            unsigned int f = *(unsigned int*)&v;
            unsigned int sign = (f >> 16) & 0x8000;
            int exp = ((f >> 23) & 0xFF) - 127 + 15;
            unsigned int mant = (f >> 13) & 0x03FF;
            if (exp <= 0) host_a_fp16[i] = sign;
            else if (exp >= 31) host_a_fp16[i] = sign | 0x7C00;
            else host_a_fp16[i] = sign | (exp << 10) | mant;
        }
        for (size_t i = 0; i < b_elems; i++) {
            float v = host_b_fp32[i];
            unsigned int f = *(unsigned int*)&v;
            unsigned int sign = (f >> 16) & 0x8000;
            int exp = ((f >> 23) & 0xFF) - 127 + 15;
            unsigned int mant = (f >> 13) & 0x03FF;
            if (exp <= 0) host_b_fp16[i] = sign;
            else if (exp >= 31) host_b_fp16[i] = sign | 0x7C00;
            else host_b_fp16[i] = sign | (exp << 10) | mant;
        }
        CHECK_CU(cuMemAlloc(&dev_a_fp16, a_elems * sizeof(unsigned short)));
        CHECK_CU(cuMemAlloc(&dev_b_fp16, b_elems * sizeof(unsigned short)));
        CHECK_CU(cuMemcpyHtoD(dev_a_fp16, host_a_fp16, a_elems * sizeof(unsigned short)));
        CHECK_CU(cuMemcpyHtoD(dev_b_fp16, host_b_fp16, b_elems * sizeof(unsigned short)));
        free(host_a_fp16);
        free(host_b_fp16);
    }

    // --- Persistent kernel: tile counter + config ---
    CUdeviceptr dev_tile_counter = 0;
    if (have_persistent) {
        CHECK_CU(cuMemAlloc(&dev_tile_counter, sizeof(int)));
    }
    int persistent_num_sms  = 48;  // GA104

    // --- Launch configs ---
    int grid_naive_x = (N + 31) / 32;  // naive: 32×32 per block
    int grid_naive_y = (M + 31) / 32;
    int grid_tiled_x = (N + 63) / 64;    // tiled: 64×64 per block
    int grid_tiled_y = (M + 63) / 64;
    int grid_regblk_x = (N + 127) / 128; // register-blocked: 128×128 per block
    int grid_regblk_y = (M + 127) / 128;
    int grid_8warp_x  = (N + 127) / 128; // 8-warp: 128×128 per block
    int grid_8warp_y  = (M + 127) / 128;
    int grid_8w256_x  = (N + 255) / 256; // 8-warp 128×256 per block
    int grid_8w256_y  = (M + 127) / 128;

    // --- Correctness check ---
    if (run_cpu_ref) {
        printf("\nCorrectness (INT8 input, INT32 accum, FP32 dequantized output):\n");

        // Naive
        CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
        void *args[] = { &dev_a_int8, &dev_b_int8, &dev_c, &M, &N, &K, &scale_a, &scale_b };
        CHECK_CU(cuLaunchKernel(igemm_func,
            grid_naive_x, grid_naive_y, 1,   64, 2, 1,
            0, NULL, args, NULL));
        CHECK_CU(cuCtxSynchronize());
        CHECK_CU(cuMemcpyDtoH(host_c_fp32, dev_c, c_elems * sizeof(float)));
        auto r1 = check_fp32(host_c_fp32, host_ref, c_elems, 0.5f, 0.1f);
        print_check_result("igemm_wmma  (naive)", r1);

        // Tiled
        if (have_tiled) {
            CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
            void *args_t[] = { &dev_a_int8, &dev_b_int8, &dev_c, &M, &N, &K, &scale_a, &scale_b };
            CHECK_CU(cuLaunchKernel(tiled_func,
                grid_tiled_x, grid_tiled_y, 1,   64, 2, 1,
                0, NULL, args_t, NULL));
            CHECK_CU(cuCtxSynchronize());
            CHECK_CU(cuMemcpyDtoH(host_c_fp32, dev_c, c_elems * sizeof(float)));
            auto r2 = check_fp32(host_c_fp32, host_ref, c_elems, 0.5f, 0.1f);
            print_check_result("igemm_tiled (smem)", r2);
        }
        // Hand-tuned tiled
        if (have_handtuned) {
            CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
            void *args_h[] = { &dev_a_int8, &dev_b_int8, &dev_c, &M, &N, &K, &scale_a, &scale_b };
            CHECK_CU(cuLaunchKernel(handtuned_func,
                grid_tiled_x, grid_tiled_y, 1,   64, 2, 1,
                0, NULL, args_h, NULL));
            CHECK_CU(cuCtxSynchronize());
            CHECK_CU(cuMemcpyDtoH(host_c_fp32, dev_c, c_elems * sizeof(float)));
            auto r_ht = check_fp32(host_c_fp32, host_ref, c_elems, 0.5f, 0.1f);
            print_check_result("igemm_handtuned (S04->S02)", r_ht);
        }
        // Aggressive hand-tuned
        if (have_aggressive) {
            CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
            void *args_ag[] = { &dev_a_int8, &dev_b_int8, &dev_c, &M, &N, &K, &scale_a, &scale_b };
            CHECK_CU(cuLaunchKernel(aggressive_func,
                grid_tiled_x, grid_tiled_y, 1,   64, 2, 1,
                0, NULL, args_ag, NULL));
            CHECK_CU(cuCtxSynchronize());
            CHECK_CU(cuMemcpyDtoH(host_c_fp32, dev_c, c_elems * sizeof(float)));
            auto r_ag = check_fp32(host_c_fp32, host_ref, c_elems, 0.5f, 0.1f);
            print_check_result("igemm_aggressive (S04->S01)", r_ag);
        }
        // Register-blocked
        if (have_regblk) {
            CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
            void *args_r[] = { &dev_a_int8, &dev_b_int8, &dev_c, &M, &N, &K, &scale_a, &scale_b };
            CHECK_CU(cuLaunchKernel(regblk_func,
                grid_regblk_x, grid_regblk_y, 1,   128, 1, 1,
                0, NULL, args_r, NULL));
            CHECK_CU(cuCtxSynchronize());
            CHECK_CU(cuMemcpyDtoH(host_c_fp32, dev_c, c_elems * sizeof(float)));
            auto r3 = check_fp32(host_c_fp32, host_ref, c_elems, 0.5f, 0.1f);
            print_check_result("igemm_regblk (128x128)", r3);
        }
        // Pipelined (LDG double-buffer)
        if (have_pipelined) {
            CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
            void *args_p[] = { &dev_a_int8, &dev_b_int8, &dev_c, &M, &N, &K, &scale_a, &scale_b };
            CHECK_CU(cuLaunchKernel(pipelined_func,
                grid_tiled_x, grid_tiled_y, 1,   64, 2, 1,
                0, NULL, args_p, NULL));
            CHECK_CU(cuCtxSynchronize());
            CHECK_CU(cuMemcpyDtoH(host_c_fp32, dev_c, c_elems * sizeof(float)));
            auto r_pipe = check_fp32(host_c_fp32, host_ref, c_elems, 0.5f, 0.1f);
            print_check_result("igemm_pipelined (LDG)", r_pipe);
        }
        // Pipelined (cp.async)
        if (have_cpasync) {
            CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
            void *args_ca[] = { &dev_a_int8, &dev_b_int8, &dev_c, &M, &N, &K, &scale_a, &scale_b };
            CHECK_CU(cuLaunchKernel(cpasync_func,
                grid_tiled_x, grid_tiled_y, 1,   64, 2, 1,
                0, NULL, args_ca, NULL));
            CHECK_CU(cuCtxSynchronize());
            CHECK_CU(cuMemcpyDtoH(host_c_fp32, dev_c, c_elems * sizeof(float)));
            auto r_ca = check_fp32(host_c_fp32, host_ref, c_elems, 0.5f, 0.1f);
            print_check_result("igemm_cpasync (LDGSTS)", r_ca);
        }
        // Per-channel asymmetric
        if (have_perchannel) {
            // CPU reference for per-channel dequantization
            float *host_ref_pc = (float *)malloc(c_elems * sizeof(float));
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    int acc_val = 0;
                    for (int k = 0; k < K; k++)
                        acc_val += (int)host_a_int8[m * K + k] * (int)host_b_int8[k * N + n];
                    host_ref_pc[m * N + n] = ((float)acc_val - (float)host_pc_zp[n]) * host_pc_scale[n];
                }
            }

            CUdeviceptr dev_pc_scale, dev_pc_zp;
            CHECK_CU(cuMemAlloc(&dev_pc_scale, N * sizeof(float)));
            CHECK_CU(cuMemAlloc(&dev_pc_zp,    N * sizeof(int)));
            CHECK_CU(cuMemcpyHtoD(dev_pc_scale, host_pc_scale, N * sizeof(float)));
            CHECK_CU(cuMemcpyHtoD(dev_pc_zp,    host_pc_zp,    N * sizeof(int)));

            CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
            void *args_pc[] = { &dev_a_int8, &dev_b_int8, &dev_c, &M, &N, &K,
                                &dev_pc_scale, &dev_pc_zp };
            CHECK_CU(cuLaunchKernel(perchannel_func,
                grid_tiled_x, grid_tiled_y, 1,   64, 2, 1,
                0, NULL, args_pc, NULL));
            CHECK_CU(cuCtxSynchronize());
            CHECK_CU(cuMemcpyDtoH(host_c_fp32, dev_c, c_elems * sizeof(float)));
            auto r_pc = check_fp32(host_c_fp32, host_ref_pc, c_elems, 0.5f, 0.1f);
            print_check_result("igemm_perchannel (asym)", r_pc);

            cuMemFree(dev_pc_scale);
            cuMemFree(dev_pc_zp);
            free(host_ref_pc);
        }
        // 8-warp 128×128
        if (have_warp8) {
            CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
            void *args_w8[] = { &dev_a_int8, &dev_b_int8, &dev_c, &M, &N, &K, &scale_a, &scale_b };
            CHECK_CU(cuLaunchKernel(warp8_func,
                grid_8warp_x, grid_8warp_y, 1,   256, 1, 1,
                0, NULL, args_w8, NULL));
            CHECK_CU(cuCtxSynchronize());
            CHECK_CU(cuMemcpyDtoH(host_c_fp32, dev_c, c_elems * sizeof(float)));
            auto r_w8 = check_fp32(host_c_fp32, host_ref, c_elems, 0.5f, 0.1f);
            print_check_result("igemm_8warp (128x128)", r_w8);
        }
        // 8-warp 128×256
        if (have_warp8_256) {
            CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
            void *args_w256[] = { &dev_a_int8, &dev_b_int8, &dev_c, &M, &N, &K, &scale_a, &scale_b };
            CHECK_CU(cuLaunchKernel(warp8_256_func,
                grid_8w256_x, grid_8w256_y, 1,   256, 1, 1,
                0, NULL, args_w256, NULL));
            CHECK_CU(cuCtxSynchronize());
            CHECK_CU(cuMemcpyDtoH(host_c_fp32, dev_c, c_elems * sizeof(float)));
            auto r_w256 = check_fp32(host_c_fp32, host_ref, c_elems, 0.5f, 0.1f);
            print_check_result("igemm_8warp_256 (128x256)", r_w256);
        }
        // Persistent 128×256
        if (have_persistent) {
            CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
            CHECK_CU(cuMemsetD32(dev_tile_counter, 0, 1));
            void *args_ps[] = { &dev_a_int8, &dev_b_int8, &dev_c, &M, &N, &K,
                                &scale_a, &scale_b, &dev_tile_counter };
            CHECK_CU(cuLaunchKernel(persistent_func,
                persistent_num_sms, 1, 1,   256, 1, 1,
                0, NULL, args_ps, NULL));
            CHECK_CU(cuCtxSynchronize());
            CHECK_CU(cuMemcpyDtoH(host_c_fp32, dev_c, c_elems * sizeof(float)));
            auto r_ps = check_fp32(host_c_fp32, host_ref, c_elems, 0.5f, 0.1f);
            print_check_result("igemm_persistent (128x256)", r_ps);
        }
        // Online-quant FP16→INT8 128×128
        if (have_onlinequant) {
            CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
            void *args_oq[] = { &dev_a_fp16, &dev_b_fp16, &dev_c, &M, &N, &K };
            int grid_oq_x = (N + 127) / 128;
            int grid_oq_y = (M + 127) / 128;
            CHECK_CU(cuLaunchKernel(onlinequant_func,
                grid_oq_x, grid_oq_y, 1,   256, 1, 1,
                0, NULL, args_oq, NULL));
            CHECK_CU(cuCtxSynchronize());
            CHECK_CU(cuMemcpyDtoH(host_c_fp32, dev_c, c_elems * sizeof(float)));
            // Wider tolerance: FP16 input + per-tile INT8 quantization
            auto r_oq = check_fp32(host_c_fp32, host_ref, c_elems, 0.5f, 0.1f);
            print_check_result("igemm_online_quant (FP16→INT8)", r_oq);
        }
        // In-place online-quant FP16→INT8 128×128
        if (have_inplace) {
            CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
            void *args_ip[] = { &dev_a_fp16, &dev_b_fp16, &dev_c, &M, &N, &K };
            int grid_ip_x = (N + 127) / 128;
            int grid_ip_y = (M + 127) / 128;
            CHECK_CU(cuLaunchKernel(inplace_func,
                grid_ip_x, grid_ip_y, 1,   256, 1, 1,
                0, NULL, args_ip, NULL));
            CHECK_CU(cuCtxSynchronize());
            CHECK_CU(cuMemcpyDtoH(host_c_fp32, dev_c, c_elems * sizeof(float)));
            auto r_ip = check_fp32(host_c_fp32, host_ref, c_elems, 0.5f, 0.1f);
            print_check_result("igemm_inplace (FP16→INT8 in-place)", r_ip);
        }
        // Bank-conflict-free online-quant FP16→INT8 128×128
        if (have_bankfree) {
            CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
            void *args_bf[] = { &dev_a_fp16, &dev_b_fp16, &dev_c, &M, &N, &K };
            int grid_bf_x = (N + 127) / 128;
            int grid_bf_y = (M + 127) / 128;
            CHECK_CU(cuLaunchKernel(bankfree_func,
                grid_bf_x, grid_bf_y, 1,   256, 1, 1,
                0, NULL, args_bf, NULL));
            CHECK_CU(cuCtxSynchronize());
            CHECK_CU(cuMemcpyDtoH(host_c_fp32, dev_c, c_elems * sizeof(float)));
            auto r_bf = check_fp32(host_c_fp32, host_ref, c_elems, 0.5f, 0.1f);
            print_check_result("igemm_bankfree (FP16→INT8 bank-free)", r_bf);
        }
        // Warp-specialized online-quant FP16→INT8 128×128
        if (have_warpspec) {
            CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
            void *args_ws[] = { &dev_a_fp16, &dev_b_fp16, &dev_c, &M, &N, &K };
            int grid_ws_x = (N + 127) / 128;
            int grid_ws_y = (M + 127) / 128;
            CHECK_CU(cuLaunchKernel(warpspec_func,
                grid_ws_x, grid_ws_y, 1,   256, 1, 1,
                0, NULL, args_ws, NULL));
            CHECK_CU(cuCtxSynchronize());
            CHECK_CU(cuMemcpyDtoH(host_c_fp32, dev_c, c_elems * sizeof(float)));
            auto r_ws = check_fp32(host_c_fp32, host_ref, c_elems, 0.5f, 0.1f);
            print_check_result("igemm_warpspec (FP16→INT8 warp-spec)", r_ws);
        }
        // HGEMM baseline (FP16)
        if (have_hgemm) {
            CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
            void *args_hg[] = { &dev_a_fp16, &dev_b_fp16, &dev_c, &M, &N, &K };
            int grid_hg_x = (N + 31) / 32;
            int grid_hg_y = (M + 31) / 32;
            CHECK_CU(cuLaunchKernel(hgemm_func,
                grid_hg_x, grid_hg_y, 1,   64, 2, 1,
                0, NULL, args_hg, NULL));
            CHECK_CU(cuCtxSynchronize());
            CHECK_CU(cuMemcpyDtoH(host_c_fp32, dev_c, c_elems * sizeof(float)));
            auto r_hg = check_fp32(host_c_fp32, host_ref, c_elems, 1e-2f, 1e-2f);
            print_check_result("hgemm_wmma (FP16 baseline)", r_hg);
        }
        // 8-warp triple-buffer 128×128
        if (have_tribuf) {
            CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
            void *args_tb[] = { &dev_a_int8, &dev_b_int8, &dev_c, &M, &N, &K, &scale_a, &scale_b };
            CHECK_CU(cuLaunchKernel(tribuf_func,
                grid_8warp_x, grid_8warp_y, 1,   256, 1, 1,
                0, NULL, args_tb, NULL));
            CHECK_CU(cuCtxSynchronize());
            CHECK_CU(cuMemcpyDtoH(host_c_fp32, dev_c, c_elems * sizeof(float)));
            auto r_tb = check_fp32(host_c_fp32, host_ref, c_elems, 0.5f, 0.1f);
            print_check_result("igemm_tribuf (128x128 3buf)", r_tb);
        }
        // Pipelined cp.async BK=64
        if (have_cpasync_bk64) {
            CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
            void *args_bk64[] = { &dev_a_int8, &dev_b_int8, &dev_c, &M, &N, &K, &scale_a, &scale_b };
            CHECK_CU(cuLaunchKernel(cpasync_bk64_func,
                grid_tiled_x, grid_tiled_y, 1,   64, 2, 1,
                0, NULL, args_bk64, NULL));
            CHECK_CU(cuCtxSynchronize());
            CHECK_CU(cuMemcpyDtoH(host_c_fp32, dev_c, c_elems * sizeof(float)));
            auto r_bk64 = check_fp32(host_c_fp32, host_ref, c_elems, 0.5f, 0.1f);
            print_check_result("igemm_cpasync_bk64 (LDGSTS)", r_bk64);
        }
        printf("  Note: quantization error expected — symmetric per-tensor, scale=max_abs/127\n");
    }

    // --- Performance benchmark ---
    int warmup_iters = 5;
    int bench_iters  = 50;
    printf("\nPerformance (avg of %d runs, %d warmup):\n", bench_iters, warmup_iters);

    double int8_peak_tops  = 696000.0;  // RTX 3070 Ti INT8 Tensor Core peak
    double fp16_peak_gflops = 174000.0;

    void *bench_args[] = { &dev_a_int8, &dev_b_int8, &dev_c, &M, &N, &K, &scale_a, &scale_b };

    // Helper to benchmark a kernel (flexible block dims)
    auto run_bench = [&](CUfunction fn, int gx, int gy, int bx, int by,
                         const char *label) {
        for (int i = 0; i < warmup_iters; i++) {
            CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
            CHECK_CU(cuLaunchKernel(fn, gx, gy, 1, bx, by, 1, 0, NULL, bench_args, NULL));
        }
        CHECK_CU(cuCtxSynchronize());

        float ms;
        {
            BenchTimer timer;
            timer.start();
            for (int i = 0; i < bench_iters; i++) {
                CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
                CHECK_CU(cuLaunchKernel(fn, gx, gy, 1, bx, by, 1, 0, NULL, bench_args, NULL));
            }
            ms = timer.stop_ms() / bench_iters;
        }
        double t = compute_gflops_gemm(M, N, K, ms);
        printf("  %-35s %7.3f ms   %8.2f TOPS\n", label, ms, t);
        return t;
    };

    double naive_tops  = run_bench(igemm_func, grid_naive_x, grid_naive_y, 64, 2,
                                   "igemm_wmma  (naive)");
    double tiled_tops  = 0.0;
    double regblk_tops = 0.0;
    if (have_tiled)
        tiled_tops = run_bench(tiled_func, grid_tiled_x, grid_tiled_y, 64, 2,
                               "igemm_tiled (64x64)");
    if (have_regblk)
        regblk_tops = run_bench(regblk_func, grid_regblk_x, grid_regblk_y, 128, 1,
                                "igemm_regblk (128x128)");
    double handtuned_tops = 0.0;
    if (have_handtuned)
        handtuned_tops = run_bench(handtuned_func, grid_tiled_x, grid_tiled_y, 64, 2,
                                   "igemm_handtuned (S04->S02)");
    double aggressive_tops = 0.0;
    if (have_aggressive)
        aggressive_tops = run_bench(aggressive_func, grid_tiled_x, grid_tiled_y, 64, 2,
                                    "igemm_aggressive (S04->S01)");
    double pipelined_tops = 0.0;
    if (have_pipelined)
        pipelined_tops = run_bench(pipelined_func, grid_tiled_x, grid_tiled_y, 64, 2,
                                   "igemm_pipelined (LDG dbuf)");
    double cpasync_tops = 0.0;
    if (have_cpasync)
        cpasync_tops = run_bench(cpasync_func, grid_tiled_x, grid_tiled_y, 64, 2,
                                 "igemm_cpasync (LDGSTS dbuf)");
    double warp8_tops = 0.0;
    if (have_warp8)
        warp8_tops = run_bench(warp8_func, grid_8warp_x, grid_8warp_y, 256, 1,
                               "igemm_8warp (128x128 cp.async)");
    double warp8_256_tops = 0.0;
    if (have_warp8_256)
        warp8_256_tops = run_bench(warp8_256_func, grid_8w256_x, grid_8w256_y, 256, 1,
                                   "igemm_8warp_256 (128x256 cp.async)");
    double warp8_256x256_tops = 0.0;
    if (have_warp8_256x256) {
        int gx256 = (N + 255) / 256, gy256 = (M + 255) / 256;
        warp8_256x256_tops = run_bench(warp8_256x256_func, gx256, gy256, 256, 1,
                                       "igemm_8warp_256x256 (256x256)");
    }
    double tribuf_tops = 0.0;
    if (have_tribuf)
        tribuf_tops = run_bench(tribuf_func, grid_8warp_x, grid_8warp_y, 256, 1,
                                "igemm_tribuf (128x128 3-buf)");
    double persistent_tops = 0.0;
    if (have_persistent) {
        // Persistent kernel needs different args and tile_counter reset per launch
        void *ps_args[] = { &dev_a_int8, &dev_b_int8, &dev_c, &M, &N, &K,
                            &scale_a, &scale_b, &dev_tile_counter };
        for (int i = 0; i < warmup_iters; i++) {
            CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
            CHECK_CU(cuMemsetD32(dev_tile_counter, 0, 1));
            CHECK_CU(cuLaunchKernel(persistent_func,
                persistent_num_sms, 1, 1, 256, 1, 1, 0, NULL, ps_args, NULL));
        }
        CHECK_CU(cuCtxSynchronize());
        float ms;
        {
            BenchTimer timer;
            timer.start();
            for (int i = 0; i < bench_iters; i++) {
                CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
                CHECK_CU(cuMemsetD32(dev_tile_counter, 0, 1));
                CHECK_CU(cuLaunchKernel(persistent_func,
                    persistent_num_sms, 1, 1, 256, 1, 1, 0, NULL, ps_args, NULL));
            }
            ms = timer.stop_ms() / bench_iters;
        }
        persistent_tops = compute_gflops_gemm(M, N, K, ms);
        printf("  %-35s %7.3f ms   %8.2f TOPS\n", "igemm_persistent (128x256 L2)", ms, persistent_tops);
    }
    double onlinequant_gflops = 0.0;
    if (have_onlinequant) {
        int grid_oq_x = (N + 127) / 128;
        int grid_oq_y = (M + 127) / 128;
        void *oq_args[] = { &dev_a_fp16, &dev_b_fp16, &dev_c, &M, &N, &K };
        for (int i = 0; i < warmup_iters; i++) {
            CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
            CHECK_CU(cuLaunchKernel(onlinequant_func,
                grid_oq_x, grid_oq_y, 1, 256, 1, 1, 0, NULL, oq_args, NULL));
        }
        CHECK_CU(cuCtxSynchronize());
        float ms;
        {
            BenchTimer timer;
            timer.start();
            for (int i = 0; i < bench_iters; i++) {
                CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
                CHECK_CU(cuLaunchKernel(onlinequant_func,
                    grid_oq_x, grid_oq_y, 1, 256, 1, 1, 0, NULL, oq_args, NULL));
            }
            ms = timer.stop_ms() / bench_iters;
        }
        onlinequant_gflops = compute_gflops_gemm(M, N, K, ms);
        printf("  %-35s %7.3f ms   %8.2f GFLOPS  (FP16 in)\n",
               "igemm_online_quant (FP16→INT8)", ms, onlinequant_gflops);
    }
    double inplace_gflops = 0.0;
    if (have_inplace) {
        int grid_ip_x = (N + 127) / 128;
        int grid_ip_y = (M + 127) / 128;
        void *ip_args[] = { &dev_a_fp16, &dev_b_fp16, &dev_c, &M, &N, &K };
        for (int i = 0; i < warmup_iters; i++) {
            CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
            CHECK_CU(cuLaunchKernel(inplace_func,
                grid_ip_x, grid_ip_y, 1, 256, 1, 1, 0, NULL, ip_args, NULL));
        }
        CHECK_CU(cuCtxSynchronize());
        float ms;
        {
            BenchTimer timer;
            timer.start();
            for (int i = 0; i < bench_iters; i++) {
                CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
                CHECK_CU(cuLaunchKernel(inplace_func,
                    grid_ip_x, grid_ip_y, 1, 256, 1, 1, 0, NULL, ip_args, NULL));
            }
            ms = timer.stop_ms() / bench_iters;
        }
        inplace_gflops = compute_gflops_gemm(M, N, K, ms);
        printf("  %-35s %7.3f ms   %8.2f GFLOPS  (FP16 in, in-place)\n",
               "igemm_inplace (FP16→INT8)", ms, inplace_gflops);
    }
    double bankfree_gflops = 0.0;
    if (have_bankfree) {
        int grid_bf_x = (N + 127) / 128;
        int grid_bf_y = (M + 127) / 128;
        void *bf_args[] = { &dev_a_fp16, &dev_b_fp16, &dev_c, &M, &N, &K };
        for (int i = 0; i < warmup_iters; i++) {
            CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
            CHECK_CU(cuLaunchKernel(bankfree_func,
                grid_bf_x, grid_bf_y, 1, 256, 1, 1, 0, NULL, bf_args, NULL));
        }
        CHECK_CU(cuCtxSynchronize());
        float ms;
        {
            BenchTimer timer;
            timer.start();
            for (int i = 0; i < bench_iters; i++) {
                CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
                CHECK_CU(cuLaunchKernel(bankfree_func,
                    grid_bf_x, grid_bf_y, 1, 256, 1, 1, 0, NULL, bf_args, NULL));
            }
            ms = timer.stop_ms() / bench_iters;
        }
        bankfree_gflops = compute_gflops_gemm(M, N, K, ms);
        printf("  %-35s %7.3f ms   %8.2f GFLOPS  (FP16 in, bank-free)\n",
               "igemm_bankfree (FP16→INT8)", ms, bankfree_gflops);
    }
    double warpspec_gflops = 0.0;
    if (have_warpspec) {
        int grid_ws_x = (N + 127) / 128;
        int grid_ws_y = (M + 127) / 128;
        void *ws_args[] = { &dev_a_fp16, &dev_b_fp16, &dev_c, &M, &N, &K };
        for (int i = 0; i < warmup_iters; i++) {
            CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
            CHECK_CU(cuLaunchKernel(warpspec_func,
                grid_ws_x, grid_ws_y, 1, 256, 1, 1, 0, NULL, ws_args, NULL));
        }
        CHECK_CU(cuCtxSynchronize());
        float ms;
        {
            BenchTimer timer;
            timer.start();
            for (int i = 0; i < bench_iters; i++) {
                CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
                CHECK_CU(cuLaunchKernel(warpspec_func,
                    grid_ws_x, grid_ws_y, 1, 256, 1, 1, 0, NULL, ws_args, NULL));
            }
            ms = timer.stop_ms() / bench_iters;
        }
        warpspec_gflops = compute_gflops_gemm(M, N, K, ms);
        printf("  %-35s %7.3f ms   %8.2f GFLOPS  (FP16 in, warp-spec)\n",
               "igemm_warpspec (FP16→INT8)", ms, warpspec_gflops);
    }
    double hgemm_gflops = 0.0;
    if (have_hgemm) {
        int grid_hg_x = (N + 31) / 32;
        int grid_hg_y = (M + 31) / 32;
        void *hg_args[] = { &dev_a_fp16, &dev_b_fp16, &dev_c, &M, &N, &K };
        for (int i = 0; i < warmup_iters; i++) {
            CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
            CHECK_CU(cuLaunchKernel(hgemm_func,
                grid_hg_x, grid_hg_y, 1, 64, 2, 1, 0, NULL, hg_args, NULL));
        }
        CHECK_CU(cuCtxSynchronize());
        float ms;
        {
            BenchTimer timer;
            timer.start();
            for (int i = 0; i < bench_iters; i++) {
                CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
                CHECK_CU(cuLaunchKernel(hgemm_func,
                    grid_hg_x, grid_hg_y, 1, 64, 2, 1, 0, NULL, hg_args, NULL));
            }
            ms = timer.stop_ms() / bench_iters;
        }
        hgemm_gflops = compute_gflops_gemm(M, N, K, ms);
        printf("  %-35s %7.3f ms   %8.2f GFLOPS  (FP16 baseline)\n",
               "hgemm_wmma (FP16)", ms, hgemm_gflops);
    }
    double cpasync_bk64_tops = 0.0;
    if (have_cpasync_bk64)
        cpasync_bk64_tops = run_bench(cpasync_bk64_func, grid_tiled_x, grid_tiled_y, 64, 2,
                                      "igemm_cpasync_bk64 (LDGSTS BK=64)");
    double perchannel_tops = 0.0;
    if (have_perchannel) {
        // Per-channel kernel has different args — use separate lambda
        void *pc_args[] = { &dev_a_int8, &dev_b_int8, &dev_c, &M, &N, &K,
                            &dev_pc_scale_bench, &dev_pc_zp_bench };
        for (int i = 0; i < warmup_iters; i++) {
            CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
            CHECK_CU(cuLaunchKernel(perchannel_func,
                grid_tiled_x, grid_tiled_y, 1, 64, 2, 1, 0, NULL, pc_args, NULL));
        }
        CHECK_CU(cuCtxSynchronize());
        float ms;
        {
            BenchTimer timer;
            timer.start();
            for (int i = 0; i < bench_iters; i++) {
                CHECK_CU(cuMemsetD32(dev_c, 0, c_elems));
                CHECK_CU(cuLaunchKernel(perchannel_func,
                    grid_tiled_x, grid_tiled_y, 1, 64, 2, 1, 0, NULL, pc_args, NULL));
            }
            ms = timer.stop_ms() / bench_iters;
        }
        perchannel_tops = compute_gflops_gemm(M, N, K, ms);
        printf("  %-35s %7.3f ms   %8.2f TOPS\n", "igemm_perchannel (asym dequant)", ms, perchannel_tops);
    }

    printf("  %-35s %7s      %8.0f TOPS  (theoretical)\n", "INT8 Tensor Core peak", "--", int8_peak_tops);
    printf("  %-35s %7s      %8.0f GFLOPS  (theoretical)\n", "FP16 Tensor Core peak", "--", fp16_peak_gflops);
    printf("\n");
    double best_tops = fmax(naive_tops, fmax(tiled_tops, fmax(regblk_tops,
                        fmax(handtuned_tops, fmax(aggressive_tops,
                        fmax(pipelined_tops, fmax(cpasync_tops,
                        fmax(cpasync_bk64_tops, fmax(perchannel_tops,
                        fmax(warp8_tops, fmax(warp8_256_tops,
                        fmax(tribuf_tops, persistent_tops))))))))))));
    printf("  Best: %.1f%% of INT8 peak, %.1f× vs FP16 peak\n",
           100.0 * best_tops / int8_peak_tops, best_tops / fp16_peak_gflops);
    if (tiled_tops > 0)
        printf("  Tiled vs naive: %.2f×\n", tiled_tops / naive_tops);
    if (regblk_tops > 0)
        printf("  RegBlk vs naive: %.2f×   RegBlk vs tiled: %.2f×\n",
               regblk_tops / naive_tops,
               tiled_tops > 0 ? regblk_tops / tiled_tops : 0.0);
    if (handtuned_tops > 0 && tiled_tops > 0)
        printf("  Handtuned vs tiled: %.4f×  (%+.2f%%)\n",
               handtuned_tops / tiled_tops,
               100.0 * (handtuned_tops - tiled_tops) / tiled_tops);
    if (aggressive_tops > 0 && tiled_tops > 0)
        printf("  Aggressive vs tiled: %.4f×  (%+.2f%%)\n",
               aggressive_tops / tiled_tops,
               100.0 * (aggressive_tops - tiled_tops) / tiled_tops);
    if (pipelined_tops > 0 && handtuned_tops > 0)
        printf("  Pipelined vs handtuned: %.4f×  (%+.2f%%)\n",
               pipelined_tops / handtuned_tops,
               100.0 * (pipelined_tops - handtuned_tops) / handtuned_tops);
    if (cpasync_tops > 0 && handtuned_tops > 0)
        printf("  cp.async vs handtuned: %.4f×  (%+.2f%%)\n",
               cpasync_tops / handtuned_tops,
               100.0 * (cpasync_tops - handtuned_tops) / handtuned_tops);
    if (pipelined_tops > 0 && cpasync_tops > 0)
        printf("  Pipelined vs cp.async: %.4f×  (%+.2f%%)\n",
               pipelined_tops / cpasync_tops,
               100.0 * (pipelined_tops - cpasync_tops) / cpasync_tops);
    if (cpasync_bk64_tops > 0 && cpasync_tops > 0)
        printf("  BK=64 vs BK=32 cp.async: %.4f×  (%+.2f%%)\n",
               cpasync_bk64_tops / cpasync_tops,
               100.0 * (cpasync_bk64_tops - cpasync_tops) / cpasync_tops);
    if (perchannel_tops > 0 && cpasync_tops > 0)
        printf("  Per-channel vs symmetric: %.4f×  (%+.2f%%)\n",
               perchannel_tops / cpasync_tops,
               100.0 * (perchannel_tops - cpasync_tops) / cpasync_tops);
    if (warp8_tops > 0 && cpasync_tops > 0)
        printf("  8-warp 128x128 vs 4-warp 64x64: %.4f×  (%+.2f%%)\n",
               warp8_tops / cpasync_tops,
               100.0 * (warp8_tops - cpasync_tops) / cpasync_tops);
    if (warp8_256_tops > 0 && warp8_tops > 0)
        printf("  128x256 vs 128x128: %.4f×  (%+.2f%%)\n",
               warp8_256_tops / warp8_tops,
               100.0 * (warp8_256_tops - warp8_tops) / warp8_tops);
    if (warp8_256x256_tops > 0 && warp8_256_tops > 0)
        printf("  256x256 vs 128x256: %.4f×  (%+.2f%%)\n",
               warp8_256x256_tops / warp8_256_tops,
               100.0 * (warp8_256x256_tops - warp8_256_tops) / warp8_256_tops);
    if (tribuf_tops > 0 && warp8_tops > 0)
        printf("  Triple-buf vs double-buf (128x128): %.4f×  (%+.2f%%)\n",
               tribuf_tops / warp8_tops,
               100.0 * (tribuf_tops - warp8_tops) / warp8_tops);
    if (persistent_tops > 0 && warp8_256_tops > 0)
        printf("  Persistent vs grid-launch (128x256): %.4f×  (%+.2f%%)\n",
               persistent_tops / warp8_256_tops,
               100.0 * (persistent_tops - warp8_256_tops) / warp8_256_tops);
    if (onlinequant_gflops > 0 && hgemm_gflops > 0)
        printf("  Online-quant IMMA vs HGEMM (FP16→FP16): %.4f×  (%+.2f%%)\n",
               onlinequant_gflops / hgemm_gflops,
               100.0 * (onlinequant_gflops - hgemm_gflops) / hgemm_gflops);
    if (bankfree_gflops > 0 && inplace_gflops > 0)
        printf("  Bank-free vs in-place: %.4f×  (%+.2f%%)\n",
               bankfree_gflops / inplace_gflops,
               100.0 * (bankfree_gflops - inplace_gflops) / inplace_gflops);
    if (bankfree_gflops > 0 && hgemm_gflops > 0)
        printf("  Bank-free IMMA vs HGEMM (FP16→FP16): %.4f×  (%+.2f%%)\n",
               bankfree_gflops / hgemm_gflops,
               100.0 * (bankfree_gflops - hgemm_gflops) / hgemm_gflops);
    if (warpspec_gflops > 0 && inplace_gflops > 0)
        printf("  Warp-spec vs in-place: %.4f×  (%+.2f%%)\n",
               warpspec_gflops / inplace_gflops,
               100.0 * (warpspec_gflops - inplace_gflops) / inplace_gflops);
    if (warpspec_gflops > 0 && hgemm_gflops > 0)
        printf("  Warp-spec IMMA vs HGEMM (FP16→FP16): %.4f×  (%+.2f%%)\n",
               warpspec_gflops / hgemm_gflops,
               100.0 * (warpspec_gflops - hgemm_gflops) / hgemm_gflops);

    printf("\nSASS inspection:\n");
    printf("  cuobjdump -sass igemm.sm_86.cubin | grep IMMA                      # naive\n");
    printf("  cuobjdump -sass igemm_tiled.sm_86.cubin | grep IMMA                # tiled\n");
    printf("  cuobjdump -sass igemm_register_blocked.sm_86.cubin | grep IMMA     # regblk\n");

    // --- Cleanup ---
    cuMemFree(dev_a_int8);
    cuMemFree(dev_b_int8);
    cuMemFree(dev_c);
    cuMemFree(dev_pc_scale_bench);
    cuMemFree(dev_pc_zp_bench);
    if (dev_tile_counter) cuMemFree(dev_tile_counter);
    if (dev_a_fp16) cuMemFree(dev_a_fp16);
    if (dev_b_fp16) cuMemFree(dev_b_fp16);
    free(host_pc_scale);
    free(host_pc_zp);
    cuModuleUnload(igemm_module);
    if (tiled_module) cuModuleUnload(tiled_module);
    if (regblk_module) cuModuleUnload(regblk_module);
    if (handtuned_module) cuModuleUnload(handtuned_module);
    if (aggressive_module) cuModuleUnload(aggressive_module);
    if (pipelined_module) cuModuleUnload(pipelined_module);
    if (cpasync_module) cuModuleUnload(cpasync_module);
    if (persistent_module) cuModuleUnload(persistent_module);
    if (onlinequant_module) cuModuleUnload(onlinequant_module);
    if (bankfree_module) cuModuleUnload(bankfree_module);
    if (warpspec_module) cuModuleUnload(warpspec_module);
    if (hgemm_module) cuModuleUnload(hgemm_module);
    cuCtxDestroy(cu_context);
    free(host_a_fp32); free(host_b_fp32); free(host_c_fp32);
    free(host_ref); free(host_a_int8); free(host_b_int8);

    return 0;
}
