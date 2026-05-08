#include <cstdio>
#include <cuda.h>
#include <cuda_fp16.h>
int main() {
    CUresult r;
    r = cuInit(0); if (r) { printf("cuInit %d\n", r); return 1; }
    CUdevice dev;
    r = cuDeviceGet(&dev, 0); if (r) { printf("cuDeviceGet %d\n", r); return 1; }
    CUcontext ctx;
    r = cuDevicePrimaryCtxRetain(&ctx, dev); if (r) { printf("cuDevicePrimaryCtxRetain %d\n", r); return 1; }
    r = cuCtxSetCurrent(ctx); if (r) { printf("cuCtxSetCurrent %d\n", r); return 1; }
    CUmodule mod;
    r = cuModuleLoad(&mod, "conv2d_implicit_gemm.sm_86.cubin");
    if (r) { printf("cuModuleLoad %d (file not found?)\n", r); return 1; }
    printf("Loaded OK\n");

    CUfunction fn;
    r = cuModuleGetFunction(&fn, mod, "implicit_gemm_conv");
    if (r) { printf("cuModuleGetFunction %d\n", r); return 1; }
    printf("Function OK\n");

    // Small grid test (1 block) — already worked
    int N=1, H=64, W=64, Cin=64, kH=3, kW=3, pad=1, oH=64, oW=64, M=64, K=576, Cout=64;
    size_t Xs=N*H*W*Cin, Bs=K*Cout, Ys=M*Cout;
    CUdeviceptr dX, dB, dY;
    cuMemAlloc(&dX, Xs*sizeof(float));
    cuMemAlloc(&dB, Bs*sizeof(__half));
    cuMemAlloc(&dY, Ys*sizeof(float));
    void* args[] = {&dX,&dB,&dY,&N,&H,&W,&Cin,&kH,&kW,&pad,&oH,&oW,&M,&K,&Cout};
    cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 8192);
    printf("Launch 1 block...\n");
    cuLaunchKernel(fn, 1,1,1, 128,1,1, 8192,0, args,0);
    cuCtxSynchronize(); printf("1 block OK\n");

    // Larger grid test (64x1)
    M=4096;
    cuMemFree(dY); cuMemAlloc(&dY, M*Cout*sizeof(float));
    void* args2[] = {&dX,&dB,&dY,&N,&H,&W,&Cin,&kH,&kW,&pad,&oH,&oW,&M,&K,&Cout};
    printf("Launch 64x1...\n");
    cuLaunchKernel(fn, 64,1,1, 128,1,1, 8192,0, args2,0);
    cuCtxSynchronize(); printf("64x1 OK\n");
    return 0;
}
