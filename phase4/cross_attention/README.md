# phase4/cross_attention — image-conditioned attention (Q from latent, KV from text)

Cross-attention block from Stable Diffusion's UNet: Q comes from the
spatial latent feature map (HWxD_q), K and V from a text-encoder context
sequence (LxD_kv). Same softmax(QKᵀ/√d)V structure as self-attention but
asymmetric shapes — Q is large, KV is short (~77 tokens for SD's CLIP).
That shape asymmetry changes what tiling wins.

## Files

| File | Variant | Status |
|---|---|---|
| `cross_attn.cu`              | baseline (mirrors phase3 flash-attn structure)        | reference |
| `cross_attn_v2.cu`           | reshaped tile to fit short KV context                 | working |
| `cross_attn_v2_pad.cu`       | + smem padding to remove bank conflicts               | working |
| `cross_attn_pipelined.cu`    | + cp.async on K/V load, double-buffered               | working |

| Bench | Pairs |
|---|---|
| `bench.cu`              | baseline vs CPU reference |
| `bench_v2.cu`           | v2 + v2_pad |
| `bench_pipelined.cu`    | pipelined vs v2_pad |

## Build

```bash
nvcc -arch=sm_86 -O2 --cubin cross_attn_pipelined.cu -o cross_attn_pipelined.sm_86.cubin
nvcc -arch=sm_86 -O2 -o bench_pipelined bench_pipelined.cu -lcuda -I../../phase2/common
./bench_pipelined
```

## Key SASS

`HMMA.16816.F32`, `MUFU.EX2` (online softmax), `SHFL.BFLY`
(warp-reduce row max + denom), `LDGSTS` (cp.async in pipelined variant).

## Cross-references

- Phase 3 [flash_attention/README.md](../../phase3/flash_attention/README.md) for the symmetric self-attention case
- [docs/tutorial/05-flash-attention.md](../../docs/tutorial/05-flash-attention.md) — softmax + tiling derivation
- [docs/kernels.md](../../docs/kernels.md) — measured numbers
