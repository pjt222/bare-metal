// gather_nvcc.cu — kernel-only build of bench_cymatic.cu's gather_sum,
// for fair PTX/SASS comparison vs cuda-oxide port (gather_oxide.ptx).
//
// Source-identical to phase4/cymatic/bench_cymatic.cu lines 84-94.

extern "C" __global__ void gather_sum(const float * __restrict__ data,
                                       const int   * __restrict__ idx,
                                       float       * __restrict__ out,
                                       int n, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float s = 0.0f;
    for (int it = 0; it < iters; ++it) {
        for (int k = tid; k < n; k += stride) {
            s += data[idx[k]];
        }
    }
    if (tid == 0) atomicAdd(out, s);
}
