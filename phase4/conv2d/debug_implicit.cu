#include <cstdio>
#include <cuda.h>
#include <cuda_fp16.h>
#define CHECK_CU(x) do { CUresult r = x; if (r != CUDA_SUCCESS) { printf("CUDA error %d at line %d\n", r, __LINE__); return 1; } } while(0)
int main() {
    printf("Init...\n"); CHECK_CU(cuInit(0));
    CUdevice dev; CHECK_CU(cuDeviceGet(&dev, 0));
    CUcontext ctx; CHECK_CU(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK_CU(cuCtxSetCurrent(ctx));
    printf("Loading modules...\n");
    CUmodule mod; CHECK_CU(cuModuleLoad(&mod, "conv2d_implicit_gemm.sm_86.cubin"));
    printf("Getting function...\n");
    CUfunction fn; CHECK_CU(cuModuleGetFunction(&fn, mod, "implicit_gemm_conv"));
    printf("Allocating...\n");
    CUdeviceptr dX, dB, dY;
    CHECK_CU(cuMemAlloc(&dX, 64*64*64*sizeof(float)));
    CHECK_CU(cuMemAlloc(&dB, 64*9*64*sizeof(__half)));
    CHECK_CU(cuMemAlloc(&dY, 64*64*64*sizeof(float)));
    printf("Launching...\n");
    int nb=1, h=64, w=64, cin=64, kh=3, kw=3, pad=1, outh=64, outw=64, M=4096, Kdim=576, Cout=64;
    void* pargs[] = {&dX, &dB, &dY, &nb, &h, &w, &cin, &kh, &kw, &pad, &outh, &outw, &M, &Kdim, &Cout};
    CHECK_CU(cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 8192));
    CHECK_CU(cuLaunchKernel(fn, 1,1,1, 128,1,1, 8192, 0, pargs, 0));
    printf("Syncing...\n");
    CHECK_CU(cuCtxSynchronize());
    printf("Done!\n");
    return 0;
}
