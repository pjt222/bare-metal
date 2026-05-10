#include <cstdio>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cstring>
#define CHECK_CU(x) do { CUresult r = x; if (r != CUDA_SUCCESS) { printf("CUDA err %d at %d\n", r, __LINE__); return 1; } } while(0)
int main() {
    CHECK_CU(cuInit(0)); CUdevice dev; CHECK_CU(cuDeviceGet(&dev, 0));
    CUcontext ctx; CHECK_CU(cuDevicePrimaryCtxRetain(&ctx, dev)); CHECK_CU(cuCtxSetCurrent(ctx));
    CUmodule mod; CHECK_CU(cuModuleLoad(&mod, "conv2d_implicit_gemm.sm_86.cubin"));
    CUfunction fn; CHECK_CU(cuModuleGetFunction(&fn, mod, "implicit_gemm_conv"));

    int N=1, H=64, W=64, Cin=64, kH=3, kW=3, pad=1;
    int out_H=64, out_W=64, M=4096, Kdim=576, Cout=64;
    size_t Xe = (size_t)N*H*W*Cin;
    size_t Ye = (size_t)M*Cout;
    size_t Be = (size_t)Kdim*Cout;
    CUdeviceptr dX, dB, dY;
    CHECK_CU(cuMemAlloc(&dX, Xe*sizeof(float)));
    CHECK_CU(cuMemAlloc(&dB, Be*sizeof(__half)));
    CHECK_CU(cuMemAlloc(&dY, Ye*sizeof(float)));
    CHECK_CU(cuMemsetD32(dX, 0, Xe));
    CHECK_CU(cuMemsetD32(dB, 0, Be));

    void* args[] = {&dX, &dB, &dY, &N, &H, &W, &Cin, &kH, &kW, &pad, &out_H, &out_W, &M, &Kdim, &Cout};
    int grid_m = (M + 64 - 1) / 64;
    int grid_n = (Cout + 64 - 1) / 64;
    printf("Grid: %d x %d\n", grid_m, grid_n);
    printf("Smem: %d bytes\n", 8192);

    CHECK_CU(cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 8192));
    printf("Launching implicit...\n");
    CHECK_CU(cuLaunchKernel(fn, grid_m, grid_n, 1, 128, 1, 1, 8192, 0, args, 0));
    printf("Syncing...\n");
    CHECK_CU(cuCtxSynchronize());
    printf("OK!\n");
    return 0;
}
