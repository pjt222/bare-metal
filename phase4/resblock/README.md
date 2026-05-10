# phase4/resblock — fused ResNet block (groupnorm → SiLU → conv → groupnorm → SiLU → conv → add)

Stable Diffusion's UNet ResBlock as a single fused kernel. End-to-end
showcase: chains three already-built primitives (conv2d, groupnorm,
activation) into one launch with intermediate values held in registers
or shared memory. The conv2d kernel choice **dominates ResBlock
runtime** — see Obs R for the 7× improvement from picking the right
implementation.

## Files

| File | Variant | Notes |
|---|---|---|
| `resblock_fused.cu`             | Direct-conv fused (baseline) | reference |
| `bench.cu`                      | resblock vs separated launches |
| `bench_implicit.cu`             | resblock with implicit_gemm v1 conv |
| `bench_implicit_v2.cu`          | resblock with implicit_gemm v2 conv | **2.18× over v1** (Obs GG) |
| `resblock.sm_86.cubin`          | compiled cubin (direct variant) |

## Build

```bash
nvcc -arch=sm_86 -O2 -o bench_implicit_v2 bench_implicit_v2.cu -lcuda -I../../kernels/_common
./bench_implicit_v2
```

## Headline (RTX 3070 Ti Laptop, sm_86)

ResBlock 320-channel @ 64×64 (SD middle block):

| Conv backend                  | ms  | TFLOPS | vs baseline |
|-------------------------------|----:|-------:|------------:|
| Direct conv2d (`resblock_fused.cu`) | 19.0 |   ~0.4 | 1.00× |
| im2col HGEMM                  |  ~5 |    ~3  | ~3.8× |
| implicit GEMM v1              | 2.92|   ~9.7 | 6.5× |
| implicit GEMM v2              | 1.34|  21.0  | **7.0× wall, 2.18× v1→v2** |

## Cross-references

- [Obs R](../../docs/gpu_reflections.md) — "ResBlock conv2d swap: 7× speedup from picking the right kernel"
- [Obs GG](../../docs/gpu_reflections.md) — implicit GEMM v2 specifically, with ResBlock as outlier-of-interest
- [phase4/conv2d/README.md](../conv2d/README.md) — the underlying conv kernels
- [kernels/reductions/groupnorm/README.md](../groupnorm/README.md) — the normalization step
