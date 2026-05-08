#include <cstdio>
#include <cuda.h>
__global__ void test_counter(unsigned int* c) { atomicAdd(c, 1); }
int main() {
  cuInit(0); CUdevice d; cuDeviceGet(&d,0); CUcontext ctx; cuDevicePrimaryCtxRetain(&ctx,d); cuCtxSetCurrent(ctx);
  unsigned int* d_c; cuMemAlloc((CUdeviceptr*)&d_c, 4); cuMemsetD32((CUdeviceptr)d_c, 0, 1);
  void* a[] = { &d_c };
  cuLaunchKernel((CUfunction)test_counter, 96,1,1, 512,1,1, 0,0,a,0);
  cuCtxSynchronize();
  unsigned int v = 0xFF; cuMemcpyDtoH(&v, (CUdeviceptr)d_c, 4); printf("counter = %u\n", v);
  return 0;
}
