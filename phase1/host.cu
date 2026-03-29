/*
 * host.cu — Phase 1 host driver using the CUDA Driver API
 *
 * We use the Driver API (not the Runtime API) because it lets us load
 * a .cubin file directly with cuModuleLoad(). This is how we run hand-
 * modified SASS without going through the nvcc link step.
 *
 * Driver API vs Runtime API:
 *   Runtime API: #include <cuda_runtime.h>, cudaMalloc, cudaMemcpy, kernel<<<>>>
 *   Driver API:  #include <cuda.h>, cuMemAlloc, cuMemcpy, cuLaunchKernel
 *
 * The Driver API is more verbose but gives direct control over cubin loading.
 *
 * Build:
 *   nvcc -o host host.cu -lcuda -arch=sm_86
 *
 * Usage (after building phase1/vector_add.cubin):
 *   host.exe vector_add.sm_86.cubin          <- run original
 *   host.exe vector_add.sm_86.modified.cubin <- run hand-modified SASS
 *
 * Expected output with FADD (addition):
 *   a[0]=1.0  b[0]=10.0  c[0]=11.0  expected=11.0  OK
 *   a[1]=2.0  b[1]=20.0  c[1]=22.0  expected=22.0  OK
 *   ...
 *
 * After changing FADD -> FMUL in the .cuasm:
 *   a[0]=1.0  b[0]=10.0  c[0]=10.0  expected=10.0  OK  (1*10=10)
 *   a[1]=2.0  b[1]=20.0  c[1]=40.0  expected=40.0  OK  (2*20=40)
 *   ...
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda.h>

#define NUM_ELEMENTS 32
#define BLOCK_SIZE   32

// Check a CUDA Driver API call and exit on failure
#define CHECK_CU(call)                                                          \
    do {                                                                        \
        CUresult cu_result = (call);                                            \
        if (cu_result != CUDA_SUCCESS) {                                        \
            const char *error_string = nullptr;                                 \
            cuGetErrorString(cu_result, &error_string);                         \
            fprintf(stderr, "CUDA Driver API error at %s:%d — %s\n",           \
                    __FILE__, __LINE__,                                          \
                    error_string ? error_string : "unknown error");             \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)


int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <path_to_cubin> [multiply|add]\n", argv[0]);
        fprintf(stderr, "  add      — expect c[i] = a[i] + b[i]  (default)\n");
        fprintf(stderr, "  multiply — expect c[i] = a[i] * b[i]  (after FADD->FMUL modification)\n");
        return EXIT_FAILURE;
    }

    const char *cubin_path = argv[1];
    bool expect_multiply = (argc >= 3 && strcmp(argv[2], "multiply") == 0);

    printf("=== bare-metal Phase 1: vector_add ===\n");
    printf("Cubin: %s\n", cubin_path);
    printf("Mode:  %s\n\n", expect_multiply ? "multiply (FMUL)" : "add (FADD)");

    // --- Initialize CUDA Driver ---
    CHECK_CU(cuInit(0));

    CUdevice cu_device;
    CHECK_CU(cuDeviceGet(&cu_device, 0));

    char device_name[256];
    CHECK_CU(cuDeviceGetName(device_name, sizeof(device_name), cu_device));
    printf("Device: %s\n", device_name);

    int compute_major, compute_minor;
    CHECK_CU(cuDeviceGetAttribute(&compute_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cu_device));
    CHECK_CU(cuDeviceGetAttribute(&compute_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cu_device));
    printf("Compute: sm_%d%d\n\n", compute_major, compute_minor);

    CUcontext cu_context;
    CHECK_CU(cuCtxCreate(&cu_context, 0, cu_device));

    // --- Load the cubin directly ---
    CUmodule cu_module;
    CUresult load_result = cuModuleLoad(&cu_module, cubin_path);
    if (load_result != CUDA_SUCCESS) {
        const char *error_string = nullptr;
        cuGetErrorString(load_result, &error_string);
        fprintf(stderr, "Failed to load cubin '%s': %s\n", cubin_path,
                error_string ? error_string : "unknown");
        fprintf(stderr, "Make sure the cubin was compiled for sm_%d%d\n",
                compute_major, compute_minor);
        return EXIT_FAILURE;
    }

    CUfunction kernel_func;
    CHECK_CU(cuModuleGetFunction(&kernel_func, cu_module, "vector_add"));
    printf("Kernel 'vector_add' loaded from cubin.\n\n");

    // --- Allocate and initialize host memory ---
    float host_a[NUM_ELEMENTS];
    float host_b[NUM_ELEMENTS];
    float host_c[NUM_ELEMENTS];

    for (int element_idx = 0; element_idx < NUM_ELEMENTS; element_idx++) {
        host_a[element_idx] = (float)(element_idx + 1);        // 1, 2, 3, ..., 32
        host_b[element_idx] = (float)(element_idx + 1) * 10.0f; // 10, 20, 30, ..., 320
        host_c[element_idx] = 0.0f;
    }

    // --- Allocate device memory ---
    CUdeviceptr device_a, device_b, device_c;
    size_t buffer_size = NUM_ELEMENTS * sizeof(float);

    CHECK_CU(cuMemAlloc(&device_a, buffer_size));
    CHECK_CU(cuMemAlloc(&device_b, buffer_size));
    CHECK_CU(cuMemAlloc(&device_c, buffer_size));

    // --- Copy input data to device ---
    CHECK_CU(cuMemcpyHtoD(device_a, host_a, buffer_size));
    CHECK_CU(cuMemcpyHtoD(device_b, host_b, buffer_size));

    // --- Launch the kernel via Driver API ---
    int num_elements = NUM_ELEMENTS;
    void *kernel_args[] = { &device_a, &device_b, &device_c, &num_elements };

    CHECK_CU(cuLaunchKernel(
        kernel_func,
        1, 1, 1,          // grid:  1 block
        BLOCK_SIZE, 1, 1, // block: 32 threads (one warp)
        0,                // shared memory bytes
        NULL,             // stream (default)
        kernel_args,
        NULL
    ));

    // Wait for kernel to complete
    CHECK_CU(cuCtxSynchronize());

    // --- Copy results back ---
    CHECK_CU(cuMemcpyDtoH(host_c, device_c, buffer_size));

    // --- Verify results ---
    printf("Results (showing first 8 elements):\n");
    printf("  %-8s %-8s %-12s %-12s %s\n", "a[i]", "b[i]", "c[i] (GPU)", "expected", "status");
    printf("  %s\n", "---------------------------------------------------");

    int num_errors = 0;
    for (int element_idx = 0; element_idx < NUM_ELEMENTS; element_idx++) {
        float expected;
        if (expect_multiply) {
            expected = host_a[element_idx] * host_b[element_idx];
        } else {
            expected = host_a[element_idx] + host_b[element_idx];
        }

        bool is_correct = (fabsf(host_c[element_idx] - expected) < 1e-3f);
        if (!is_correct) {
            num_errors++;
        }

        // Print first 8 elements + any errors
        if (element_idx < 8 || !is_correct) {
            printf("  %-8.1f %-8.1f %-12.1f %-12.1f %s\n",
                   host_a[element_idx],
                   host_b[element_idx],
                   host_c[element_idx],
                   expected,
                   is_correct ? "OK" : "MISMATCH");
        }
    }
    if (NUM_ELEMENTS > 8) {
        printf("  ... (%d more elements)\n", NUM_ELEMENTS - 8);
    }

    printf("\n");
    if (num_errors == 0) {
        printf("PASS: All %d elements correct.\n", NUM_ELEMENTS);
        if (expect_multiply) {
            printf("      FMUL modification confirmed — GPU is multiplying, not adding!\n");
        } else {
            printf("      Original FADD kernel working correctly.\n");
        }
    } else {
        printf("FAIL: %d/%d elements incorrect.\n", num_errors, NUM_ELEMENTS);
    }

    // --- Cleanup ---
    cuMemFree(device_a);
    cuMemFree(device_b);
    cuMemFree(device_c);
    cuModuleUnload(cu_module);
    cuCtxDestroy(cu_context);

    return (num_errors == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
