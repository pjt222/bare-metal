#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include "../common/bench_driver.h"
#include "sparse_meta_int8.h"
int main(int argc, char **argv) {
    int M = (argc>1)?atoi(argv[1]):512;
    int N = (argc>2)?atoi(argv[2]):M;
    int K = (argc>3)?atoi(argv[3]):M;
    if(M%16||N%16||K%32){printf("bad align\n");return 1;}
    BenchDriver driver; driver.init_context();
    size_t elems_comp=(size_t)M*(K/2), elems_b=(size_t)K*N, elems_c=(size_t)M*N;
    size_t meta_count=sparse_meta_count_int8(M,K);
    float scale_a=1.0f/127, scale_b=1.0f/127;
    auto h_A=driver.host_alloc<int8_t>((size_t)M*K);
    auto h_Ac=driver.host_alloc<int8_t>(elems_comp);
    auto h_B=driver.host_alloc<int8_t>(elems_b);
    auto h_meta=driver.host_alloc<uint32_t>(meta_count);
    auto h_ref_i32=driver.host_alloc<int32_t>(elems_c);
    auto h_ref=driver.host_alloc<float>(elems_c);
    auto h_out=driver.host_alloc<float>(elems_c);
    gen_random_sparse_2_4_int8(h_A.get(),M,K,42); compress_2_4_int8(h_A.get(),M,K,h_Ac.get(),h_meta.get());
    srand(137); for(size_t i=0;i<elems_b;i++){int8_t v;do{v=(int8_t)((rand()%255)-127);}while(v==0); h_B[i]=v;}
    cpu_sparse_gemm_int8(h_A.get(),h_B.get(),h_ref_i32.get(),M,N,K);
    float ds=scale_a*scale_b; for(size_t i=0;i<elems_c;i++)h_ref[i]=(float)h_ref_i32[i]*ds;
    auto d_Ac=driver.device_alloc<int8_t>(elems_comp);
    auto d_B=driver.device_alloc<int8_t>(elems_b);
    auto d_C=driver.device_alloc<float>(elems_c);
    auto d_meta=driver.device_alloc<uint32_t>(meta_count);
    auto d_ctr=driver.device_alloc<unsigned int>(1);
    driver.copy_h2d(d_Ac,h_Ac,elems_comp*sizeof(int8_t)); driver.copy_h2d(d_B,h_B,elems_b*sizeof(int8_t)); driver.copy_h2d(d_meta,h_meta,meta_count*sizeof(uint32_t));
    CUdeviceptr dctr=(CUdeviceptr)d_ctr.ptr;
    CHECK_CU(cuMemsetD32(dctr,0,1));
    unsigned int chk=0xFF;
    CHECK_CU(cuMemcpyDtoH(&chk,dctr,sizeof(chk)));
    printf("Pre-launch counter: %u\n",chk);

    CUfunction fn=driver.load_kernel("igemm_sparse_tiled_persistent.sm_86.cubin","igemm_sparse_tiled_persistent");
    dim3 grid(96,1,1), block(512,1,1);
    void* args[]={&d_Ac.ptr,&d_B.ptr,&d_C.ptr,&d_meta.ptr,&M,&N,&K,&scale_a,&scale_b,&dctr};
    CHECK_CU(cuMemsetD32((CUdeviceptr)d_C.ptr,0,elems_c));
    CHECK_CU(cuLaunchKernel(fn,grid.x,grid.y,grid.z,block.x,block.y,block.z,0,nullptr,args,nullptr));
    CHECK_CU(cuCtxSynchronize());
    CHECK_CU(cuMemcpyDtoH(&chk,dctr,sizeof(chk)));
    printf("Post-launch counter: %u\n",chk);
    driver.copy_d2h(h_out,d_C,elems_c*sizeof(float));
    driver.check(h_out.get(),h_ref.get(),(int)elems_c,0.5f,0.1f,"persistent");
    return 0;
}
