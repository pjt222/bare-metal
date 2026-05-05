/*
 * verify_wmma_layout.cu — Determine WMMA m16n16k16 accumulator fragment layout
 *
 * For each fragment element x[i] (i=0..7), determine which (row, col) of the
 * 16×16 output tile it corresponds to. This is done by:
 *   1. Creating an accumulator fragment initialized to 0
 *   2. Setting one element x[i] = 1.0
 *   3. Storing to smem via store_matrix_sync
 *   4. Scanning smem to find which (row, col) has the 1.0
 *
 * The result is written to global memory as a mapping table.
 *
 * Build:
 *   nvcc -arch=sm_86 -O2 -o verify_wmma_layout verify_wmma_layout.cu -lcuda
 */

#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Output per thread: 8 entries of (element_idx, row, col)
struct LayoutEntry {
    int lane;
    int elem_idx;
    int row;
    int col;
};

extern "C" __global__ void probe_layout(LayoutEntry *out) {
    __shared__ float smem[WMMA_M * WMMA_N];  // 16×16 = 256 floats

    int lane = threadIdx.x;  // 0..31 (one warp)
    if (lane >= 32) return;

    for (int elem = 0; elem < 8; elem++) {
        // Create fragment with only x[elem] = 1.0
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag;
        wmma::fill_fragment(frag, 0.0f);
        frag.x[elem] = 1.0f;

        // Store to shared memory
        wmma::store_matrix_sync(smem, frag, WMMA_N, wmma::mem_row_major);
        __syncwarp();

        // Scan smem to find the 1.0
        int found_row = -1, found_col = -1;
        for (int r = 0; r < WMMA_M; r++) {
            for (int c = 0; c < WMMA_N; c++) {
                if (smem[r * WMMA_N + c] == 1.0f) {
                    // Multiple threads set x[elem]=1.0, so multiple positions have 1.0.
                    // We need to find OUR position. Since all threads in the warp
                    // execute store_matrix_sync together, the smem has the SUM of all
                    // threads' contributions. So smem[r][c] = count of threads whose
                    // x[elem] maps to (r,c).
                    //
                    // Instead, let's use a different approach: set only THIS thread's
                    // fragment element to 1.0 by using lane as a disambiguator.
                }
            }
        }
    }

    // Better approach: use unique values per thread
    for (int elem = 0; elem < 8; elem++) {
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag;
        wmma::fill_fragment(frag, 0.0f);
        float marker = (float)(lane * 8 + elem + 1);  // unique nonzero value per (lane, elem)
        frag.x[elem] = marker;

        wmma::store_matrix_sync(smem, frag, WMMA_N, wmma::mem_row_major);
        __syncwarp();

        // Find our marker in smem
        int found_row = -1, found_col = -1;
        for (int r = 0; r < WMMA_M; r++) {
            for (int c = 0; c < WMMA_N; c++) {
                if (smem[r * WMMA_N + c] == marker) {
                    found_row = r;
                    found_col = c;
                }
            }
        }

        int out_idx = lane * 8 + elem;
        out[out_idx].lane = lane;
        out[out_idx].elem_idx = elem;
        out[out_idx].row = found_row;
        out[out_idx].col = found_col;

        __syncwarp();
    }
}

int main() {
    printf("=== WMMA m16n16k16 Accumulator Fragment Layout (sm_86) ===\n\n");

    CUresult err;
    err = cuInit(0);
    CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuCtxCreate(&ctx, 0, dev);

    CUmodule mod;
    // Compile inline
    err = cuModuleLoad(&mod, "verify_wmma_layout.sm_86.cubin");
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "Load cubin failed. Build first:\n");
        fprintf(stderr, "  nvcc --cubin -arch=sm_86 -O2 -o verify_wmma_layout.sm_86.cubin verify_wmma_layout.cu\n");
        return 1;
    }

    CUfunction fn;
    cuModuleGetFunction(&fn, mod, "probe_layout");

    int num_entries = 32 * 8;  // 32 lanes × 8 elements
    CUdeviceptr d_out;
    cuMemAlloc(&d_out, num_entries * sizeof(LayoutEntry));

    void *args[] = { &d_out };
    cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 256 * sizeof(float), NULL, args, NULL);
    cuCtxSynchronize();

    LayoutEntry *h_out = (LayoutEntry*)malloc(num_entries * sizeof(LayoutEntry));
    cuMemcpyDtoH(h_out, d_out, num_entries * sizeof(LayoutEntry));

    // Print layout table
    printf("Lane  Elem  Row  Col\n");
    printf("----  ----  ---  ---\n");
    for (int i = 0; i < num_entries; i++) {
        LayoutEntry &e = h_out[i];
        printf("%4d  %4d  %3d  %3d\n", e.lane, e.elem_idx, e.row, e.col);
    }

    // Print summary: for each element index, which row does it map to?
    printf("\n=== Row mapping per element (grouped by lane) ===\n");
    printf("Lane  groupID  x[0].row  x[1].row  x[2].row  x[3].row  x[4].row  x[5].row  x[6].row  x[7].row\n");
    for (int lane = 0; lane < 32; lane++) {
        int base = lane * 8;
        printf("%4d  %7d", lane, lane >> 2);
        for (int e = 0; e < 8; e++) {
            printf("  %8d", h_out[base + e].row);
        }
        printf("\n");
    }

    cuMemFree(d_out);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);
    free(h_out);
    return 0;
}
