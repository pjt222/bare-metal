# kernels/elementwise/timestep_emb — sinusoidal timestep embedding

Maps a scalar diffusion timestep `t ∈ [0, 1000)` to a sinusoidal
embedding vector of dimension `d_model`, using the Transformer
positional-encoding formula. Showcase kernel for `MUFU.SIN`,
`MUFU.COS`, and `MUFU.EX2` — Ampere's hardware special-function units.

## Files

| File | Purpose |
|---|---|
| `timestep_emb.cu`               | Kernel + host wrapper |
| `bench.cu`                      | runtime + correctness vs CPU reference |
| `timestep_emb.sm_86.cubin`      | compiled cubin |

## Build

```bash
# IMPORTANT: --use_fast_math is required for MUFU.SIN/COS to appear in SASS.
# Without it, nvcc emits a multi-instruction polynomial approximation.
nvcc -arch=sm_86 -O2 --use_fast_math --cubin timestep_emb.cu \
     -o timestep_emb.sm_86.cubin
nvcc -arch=sm_86 -O2 --use_fast_math -o bench bench.cu \
     -lcuda -I../../kernels/_common
./bench
```

## Key SASS

```sass
MUFU.SIN  R8, R0      ; hardware sine, ~16 cycle latency
MUFU.COS  R9, R0      ; hardware cosine
MUFU.EX2  R5, R5      ; 2^x — used for exp(-log(10000)*i/d) via x*log2(e)
```

The frequency calc `exp(-log(10000) * i / (d_model/2))` is rewritten as
`exp2f(-log2(10000) * i / (d_model/2))` so it routes through `MUFU.EX2`
instead of the software `expf` polynomial.

## Tolerance

`abs=5e-4, rel=5e-4` (fast-math sin/cos drops a couple of mantissa bits).

## Cross-references

- [docs/tutorial/06-the-four-laws.md](../../docs/tutorial/06-the-four-laws.md) — Law 3 ("fill the warp schedulers") on MUFU latency hiding
- [kernels/attention/cross_attention/README.md](../cross_attention/README.md) — uses the timestep embedding as conditioning input
