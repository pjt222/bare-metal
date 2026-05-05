/*
 * verify_wmma_ab_layout.cu — Determine WMMA m16n16k16 matrix_a and matrix_b fragment layouts
 *
 * For matrix_a/matrix_b fragments, store_matrix_sync does NOT exist (only for
 * accumulators). So we use a different approach: load a matrix filled with
 * unique values via WMMA, then read back frag.x[] elements directly. Since
 * each element is a known __half value, we can decode which (row, col) in the
 * source matrix it came from.
 *
 * Values: M[r][c] = __float2half(r * 16 + c + 1)  (1..256, all exact in FP16)
 * Decode: row = ((int)val - 1) / 16, col = ((int)val - 1) % 16
 *
 * Build:
 *   nvcc --cubin -arch=sm_86 -O2 -o verify_wmma_ab_layout.sm_86.cubin verify_wmma_ab_layout.cu
 *   nvcc -arch=sm_86 -O2 -o verify_wmma_ab_layout verify_wmma_ab_layout.cu -lcuda
 */

#include <cstdio>
#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

struct FragEntry {
    int lane;
    int elem_idx;
    int row;
    int col;
    uint32_t raw_reg;  // uint32_t register containing this element (elem_idx / 2)
};

// ---- Probe kernel for matrix_a (row_major) ----
extern "C" __global__ void probe_frag_a(FragEntry *out) {
    __shared__ __half smem[16 * 16];
    int lane = threadIdx.x;

    // Fill with unique values: smem[r][c] = r*16 + c + 1
    for (int i = lane; i < 256; i += 32) {
        int r = i / 16, c = i % 16;
        smem[i] = __float2half((float)(r * 16 + c + 1));
    }
    __syncwarp();

    // Load via WMMA
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> frag;
    wmma::load_matrix_sync(frag, smem, 16);

    // Read back raw uint32_t registers
    uint32_t *regs = reinterpret_cast<uint32_t*>(&frag.x[0]);

    // Extract per-element mapping
    for (int i = 0; i < frag.num_elements; i++) {
        float val = __half2float(frag.x[i]);
        int ival = (int)(val + 0.5f);  // round to nearest int
        int row = (ival - 1) / 16;
        int col = (ival - 1) % 16;

        int idx = lane * frag.num_elements + i;
        out[idx].lane = lane;
        out[idx].elem_idx = i;
        out[idx].row = row;
        out[idx].col = col;
        out[idx].raw_reg = regs[i / 2];
    }
}

// ---- Probe kernel for matrix_b (row_major) ----
extern "C" __global__ void probe_frag_b_row(FragEntry *out) {
    __shared__ __half smem[16 * 16];
    int lane = threadIdx.x;

    for (int i = lane; i < 256; i += 32) {
        int r = i / 16, c = i % 16;
        smem[i] = __float2half((float)(r * 16 + c + 1));
    }
    __syncwarp();

    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> frag;
    wmma::load_matrix_sync(frag, smem, 16);

    uint32_t *regs = reinterpret_cast<uint32_t*>(&frag.x[0]);

    for (int i = 0; i < frag.num_elements; i++) {
        float val = __half2float(frag.x[i]);
        int ival = (int)(val + 0.5f);
        int row = (ival - 1) / 16;
        int col = (ival - 1) % 16;

        int idx = lane * frag.num_elements + i;
        out[idx].lane = lane;
        out[idx].elem_idx = i;
        out[idx].row = row;
        out[idx].col = col;
        out[idx].raw_reg = regs[i / 2];
    }
}

// ---- Probe kernel for matrix_b (col_major) ----
// PTX mma.sp uses .row.col (A row-major, B col-major), so probe this variant too
extern "C" __global__ void probe_frag_b_col(FragEntry *out) {
    __shared__ __half smem[16 * 16];
    int lane = threadIdx.x;

    // Store in COLUMN-major: smem[col * 16 + row] = value for (row, col)
    // Value encoding: B[r][c] = r*16 + c + 1 (same logical matrix)
    for (int i = lane; i < 256; i += 32) {
        int col = i / 16, row = i % 16;  // col-major indexing
        smem[i] = __float2half((float)(row * 16 + col + 1));
    }
    __syncwarp();

    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> frag;
    wmma::load_matrix_sync(frag, smem, 16);  // leading dim = 16 (col stride)

    uint32_t *regs = reinterpret_cast<uint32_t*>(&frag.x[0]);

    for (int i = 0; i < frag.num_elements; i++) {
        float val = __half2float(frag.x[i]);
        int ival = (int)(val + 0.5f);
        int row = (ival - 1) / 16;
        int col = (ival - 1) % 16;

        int idx = lane * frag.num_elements + i;
        out[idx].lane = lane;
        out[idx].elem_idx = i;
        out[idx].row = row;
        out[idx].col = col;
        out[idx].raw_reg = regs[i / 2];
    }
}

// ---- Host ----

void print_mapping(const char *title, FragEntry *entries, int num_entries, int num_elements) {
    printf("\n=== %s ===\n\n", title);

    // Detailed table
    printf("Lane  gID  tid  Elem  Row  Col  Reg[elem/2]\n");
    printf("----  ---  ---  ----  ---  ---  -----------\n");
    for (int i = 0; i < num_entries; i++) {
        FragEntry &e = entries[i];
        printf("%4d  %3d  %3d  %4d  %3d  %3d  0x%08x\n",
               e.lane, e.lane >> 2, e.lane & 3,
               e.elem_idx, e.row, e.col, e.raw_reg);
    }

    // Summary: group by groupID, show row/col pattern per element
    printf("\n--- Summary by groupID (lanes 0-3 shown as representative) ---\n");
    printf("gID  ");
    for (int e = 0; e < num_elements; e++) printf("x[%2d]     ", e);
    printf("\n");

    for (int gid = 0; gid < 8; gid++) {
        int representative_lane = gid * 4;  // first lane in group
        printf("%3d  ", gid);
        for (int e = 0; e < num_elements; e++) {
            int idx = representative_lane * num_elements + e;
            printf("(%2d,%2d)   ", entries[idx].row, entries[idx].col);
        }
        printf("\n");
    }

    // Register-level summary: which (row,col) pairs per uint32_t register
    int num_regs = num_elements / 2;
    printf("\n--- Register-level summary (uint32_t regs, lane 0) ---\n");
    printf("Reg#  x[lo]       x[hi]       Raw\n");
    for (int r = 0; r < num_regs; r++) {
        int lo = r * 2;
        int hi = r * 2 + 1;
        printf("%4d  (%2d,%2d)     (%2d,%2d)     0x%08x\n",
               r,
               entries[lo].row, entries[lo].col,
               entries[hi].row, entries[hi].col,
               entries[lo].raw_reg);
    }
}

int main() {
    printf("=== WMMA m16n16k16 Fragment Layout Probe (sm_86) ===\n");

    CUresult err = cuInit(0);
    CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuCtxCreate(&ctx, 0, dev);

    CUmodule mod;
    err = cuModuleLoad(&mod, "verify_wmma_ab_layout.sm_86.cubin");
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "Load cubin failed (err=%d). Build first:\n", err);
        fprintf(stderr, "  nvcc --cubin -arch=sm_86 -O2 -o verify_wmma_ab_layout.sm_86.cubin verify_wmma_ab_layout.cu\n");
        return 1;
    }

    // We expect 16 elements per fragment (matrix_a and matrix_b for m16n16k16 __half)
    int num_elements = 16;
    int num_entries = 32 * num_elements;  // 32 lanes × 16 elements = 512

    CUdeviceptr d_out;
    cuMemAlloc(&d_out, num_entries * sizeof(FragEntry));
    FragEntry *h_out = new FragEntry[num_entries];

    // ---- Probe matrix_a (row_major) ----
    {
        CUfunction fn;
        cuModuleGetFunction(&fn, mod, "probe_frag_a");
        cuMemsetD8(d_out, 0, num_entries * sizeof(FragEntry));

        void *args[] = { &d_out };
        cuLaunchKernel(fn, 1,1,1, 32,1,1, 16*16*sizeof(__half), NULL, args, NULL);
        cuCtxSynchronize();

        cuMemcpyDtoH(h_out, d_out, num_entries * sizeof(FragEntry));
        print_mapping("matrix_a (row_major)", h_out, num_entries, num_elements);
    }

    // ---- Probe matrix_b (row_major) ----
    {
        CUfunction fn;
        cuModuleGetFunction(&fn, mod, "probe_frag_b_row");
        cuMemsetD8(d_out, 0, num_entries * sizeof(FragEntry));

        void *args[] = { &d_out };
        cuLaunchKernel(fn, 1,1,1, 32,1,1, 16*16*sizeof(__half), NULL, args, NULL);
        cuCtxSynchronize();

        cuMemcpyDtoH(h_out, d_out, num_entries * sizeof(FragEntry));
        print_mapping("matrix_b (row_major)", h_out, num_entries, num_elements);
    }

    // ---- Probe matrix_b (col_major) ----
    {
        CUfunction fn;
        cuModuleGetFunction(&fn, mod, "probe_frag_b_col");
        cuMemsetD8(d_out, 0, num_entries * sizeof(FragEntry));

        void *args[] = { &d_out };
        cuLaunchKernel(fn, 1,1,1, 32,1,1, 16*16*sizeof(__half), NULL, args, NULL);
        cuCtxSynchronize();

        cuMemcpyDtoH(h_out, d_out, num_entries * sizeof(FragEntry));
        print_mapping("matrix_b (col_major)", h_out, num_entries, num_elements);
    }

    delete[] h_out;
    cuMemFree(d_out);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);
    return 0;
}
