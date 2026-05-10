#include <cstdio>
#include <cuda.h>
#define CHK(x) do { CUresult r=x; if(r!=CUDA_SUCCESS){printf("ERR %d at %d\n",r,__LINE__); return 1;} } while(0)
int main() {
    CHK(cuInit(0)); CUdevice dev; CHK(cuDeviceGet(&dev,0));
    CUcontext ctx; CHK(cuDevicePrimaryCtxRetain(&ctx,dev)); CHK(cuCtxSetCurrent(ctx));
    CUdeviceptr d_ctr; CHK(cuMemAlloc(&d_ctr, 4));
    CHK(cuMemsetD32(d_ctr, 0, 1));
    unsigned int v = 0xDEADBEEF;
    CHK(cuMemcpyDtoH(&v, d_ctr, 4)); printf("memset: %u\n", v);

    const char* ptx = ".version 7.0\n.target sm_86\n.address_size 64\n"
        ".entry test_kernel(.param .u64 p) {\n"
        "  ld.param.u64 %rd0, [p];\n"
        "  mov.u32 %r0, 42;\n"
        "  st.global.u32 [%rd0], %r0;\n"
        "  ret;\n} ";
    CUmodule mod; CHK(cuModuleLoadData(&mod, ptx));
    CUfunction fn; CHK(cuModuleGetFunction(&fn, mod, "test_kernel"));
    void* a[] = {&d_ctr};
    CHK(cuLaunchKernel(fn, 1,1,1, 1,1,1, 0,0,a,0));
    CHK(cuCtxSynchronize());
    CHK(cuMemcpyDtoH(&v, d_ctr, 4)); printf("kernel: %u\n", v);
    return 0;
}
