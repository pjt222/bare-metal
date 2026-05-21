# kernels/attention/cross_attention — image-conditioned attention (Q from latent, KV from text)

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
| `dispatch.h`                 | regime dispatch helper (header-only)                  | working |

| Bench | Pairs |
|---|---|
| `bench.cu`              | baseline vs CPU reference |
| `bench_v2.cu`           | v2 + v2_pad |
| `bench_pipelined.cu`    | pipelined vs v2_pad |
| `bench_dispatch.cu`     | dispatch correctness across CLIP-77, threshold, typical SD |

## Build

```bash
nvcc -arch=sm_86 -O2 --cubin cross_attn_pipelined.cu -o cross_attn_pipelined.sm_86.cubin
nvcc -arch=sm_86 -O2 -o bench_pipelined bench_pipelined.cu -lcuda -I../../kernels/_common
./bench_pipelined
```

## Key SASS

`HMMA.16816.F32`, `MUFU.EX2` (online softmax), `SHFL.BFLY`
(warp-reduce row max + denom), `LDGSTS` (cp.async in pipelined variant).

## Regime dispatch

`dispatch.h` provides `cross_attn_pick(seq_q, seq_kv)` which returns the
`CrossAttnVariant` (cubin path, symbol, smem bytes) to use for a given
problem size.

### Boundary

```
(size_t)seq_q * seq_kv >= 200 000  →  cross_attn_v2_pad  (27 KB smem)
(size_t)seq_q * seq_kv <  200 000  →  cross_attn_br16    (48 KB smem)
```

The threshold 200 000 is derived from the cross-attention regime-split
measurements documented in **Obs X** (`docs/gpu_reflections.md`, the
"Cross-attention regime split" table). Obs X measures the padded variant
against the unpadded `cross_attn_v2` (column `pad / v2`):

| Configuration | seq_q × seq_kv | v2_pad / v2 |
|---|---:|---:|
| CLIP-77 (256 × 77) | 19 712 | 0.68× — padding *loses* |
| 512 × 256 | 131 072 | 1.52× |
| typical SD (1024 × 256) | 262 144 | 1.91× — padding wins |

Padding loses below the threshold and wins decisively above it. The
200 000 boundary is taken directly from the "Production guidance" block
in Obs X. That block specifies a three-tier dispatch (baseline / v2 /
v2_pad, with a second cut at 50 000); issue #103 scopes this helper to
the two-way baseline / v2_pad split, so the intermediate v2 tier is
collapsed here. Revisit if a v2-only regime is needed. See also **Obs P**
(`docs/gpu_reflections.md`, "Flash Attention smem_work elimination") for
the smem-work elimination background and **Obs JJ** ("Cymatic on Flash
Attention K/V") for the cp.async structural-limit context.

### Usage example

```cpp
#include "dispatch.h"

CrossAttnVariant v = cross_attn_pick(seq_q, seq_kv);
CUfunction fn = load_kernel(v.cubin_path, v.symbol);
cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                   (int)v.smem_bytes);
void *args[] = { &dQ, &dK, &dV, &dO, &seq_q, &seq_kv, &num_heads, &scale };
cuLaunchKernel(fn, cross_attn_grid_x(seq_q), num_heads, batch,
               CROSS_ATTN_BLOCK_THREADS, 1, 1,
               (unsigned)v.smem_bytes, nullptr, args, nullptr);
```

## Cross-references

- Phase 3 [flash_attention/README.md](../flash_attention/README.md) for the symmetric self-attention case
- [docs/tutorial/05-flash-attention.md](../../../docs/tutorial/05-flash-attention.md) — softmax + tiling derivation
- [docs/inventory.md](../../../docs/inventory.md) — measured numbers
- **Obs P** (`docs/gpu_reflections.md` line ~1062) — smem_work elimination
- **Obs X** (`docs/gpu_reflections.md` line ~1835) — +8 padding + cross-attn regime split
- **Obs JJ** (`docs/gpu_reflections.md` line ~2990) — cp.async structural limit for FA K/V
