# phase4/conv2d — 2D convolution: direct, im2col, implicit GEMM

Three implementations of the same operator, each progressively closer to
DRAM-once and Tensor-Core-saturating execution. The progression
illustrates Law 2 ("read each byte of DRAM exactly once") — direct conv
re-reads input 9× per output, im2col explicitly materializes the 9× col
buffer, implicit GEMM eliminates it entirely.

## Files

### Kernels

| File | Approach | Status |
|---|---|---|
| `conv2d.cu`                       | Direct convolution, scalar FFMA | reference |
| `conv2d_im2col.cu`                | Im2col + WMMA HGEMM                | works, dominated by col-buffer DRAM |
| `conv2d_implicit_gemm.cu`         | Implicit GEMM v1                   | superseded by v2 |
| `conv2d_implicit_gemm_v2.cu`      | Implicit GEMM v2 (canonical)       | **2.18× ResBlock outlier**, Obs GG |
| `conv2d_direct.cuasm`             | hand-tuned SASS over `conv2d.cu`   | reference for hand-edit research |
| `wmma_gemm.cuasm`                 | WMMA primitive shared with im2col path | building block |

### Bench

| File | Compares |
|---|---|
| `bench.cu`                       | direct vs im2col baseline |
| `bench_im2col.cu`                | im2col path in isolation |
| `bench_implicit_gemm.cu`         | implicit GEMM v1 alone |
| `bench_implicit_v2.cu`           | implicit GEMM v2 alone |

### Debug harnesses (kept; gitignored binaries)

`debug_implicit.cu`, `debug_implicit2.cu`, `debug3.cu` — minimal repros
for tile-shape / smem-layout investigations during the v1→v2 transition.

## Headline (RTX 3070 Ti Laptop, sm_86)

64×64 image, 320 channels (Stable Diffusion middle-block):

| Variant       | TFLOPS | % peak | Notes |
|---------------|-------:|-------:|---|
| Direct (FFMA) |   ~0.4 |   2%   | bandwidth-starved (9× DRAM reads) |
| Im2col + WMMA |   ~3.0 |  17%   | 1× DRAM, but col buffer in L2 |
| Implicit v2   | **6.7**|  38%   | no col buffer, on-the-fly index gen |

## Build

```bash
nvcc -arch=sm_86 -O2 --cubin conv2d_implicit_gemm_v2.cu -o conv2d_implicit_gemm_v2.sm_86.cubin
nvcc -arch=sm_86 -O2 -o bench_implicit_v2 bench_implicit_v2.cu -lcuda -I../../kernels/_common
./bench_implicit_v2
```

## Cross-references

- [Obs 2](../../docs/gpu_reflections.md) — "Your Conv2d Reads X Nine Times From VRAM" (the original observation)
- [Obs GG](../../docs/gpu_reflections.md) — implicit GEMM v2: 2.18× on ResBlock outlier
- [docs/tutorial/04-software-pipelining.md](../../docs/tutorial/04-software-pipelining.md) — cp.async regime analysis
