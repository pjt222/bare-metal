# phase4/groupnorm — group normalization

Per-group mean/variance normalization used in Stable Diffusion's UNet
ResBlocks (32 groups by convention). Two reductions per group (sum,
sum-of-squares), then normalize+affine. The inner reduction is the
showcase: warp-level `SHFL.BFLY` butterfly trees plus `MUFU.RSQ` for
the inverse square root.

## Files

| File | Purpose |
|---|---|
| `groupnorm.cu`              | Kernel: per-block group reduce, normalize, scale+shift |
| `bench.cu`                  | bandwidth + correctness vs CPU reference |
| `groupnorm.sm_86.cubin`     | compiled cubin |

## Build

```bash
nvcc -arch=sm_86 -O2 --cubin groupnorm.cu -o groupnorm.sm_86.cubin
nvcc -arch=sm_86 -O2 -o bench bench.cu -lcuda -I../../phase2/common
./bench
```

## Key SASS

```sass
SHFL.BFLY  R12, R12, 0x10, ...   ; warp butterfly reduce, 5 instructions for 32-lane sum
MUFU.RSQ   R8, R8                ; hardware 1/sqrt, ~16 cycle latency
MUFU.RCP   R9, R9                ; reciprocal for the 1/N divide
FMUL       R10, R0, R9           ; (x - mean) * rsqrt_var
```

## Headline (RTX 3070 Ti Laptop, sm_86)

| Shape (SD 320ch) | Achieved | DRAM peak | % peak |
|---|---:|---:|---:|
| 4×64×64×320 fp32 | ~50 GB/s | 608 GB/s | 8.2% |

DRAM-bandwidth-bound; the group reduction is small relative to the
elementwise pass. SHFL is free here — the reduction never stalls
behind it.

## Cross-references

- [docs/fragment_shfl_reductions.md](../../docs/fragment_shfl_reductions.md) — butterfly-reduce SASS pattern catalog
- [Obs R](../../docs/gpu_reflections.md) — ResBlock end-to-end where this kernel is one of three components
