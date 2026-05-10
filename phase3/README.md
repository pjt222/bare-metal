# Phase 3 — Flash Attention

Single subdir, single algorithm, longest optimization arc in the project:
**21 kernel variants** from scalar O(N²) reference to Br=16 HMMA tiling
with persistent grids and cp.async pipelining. End-to-end speedup over
the naive reference: ~19×.

## Subdirectory

| Dir | Contents |
|---|---|
| [`flash_attention/`](flash_attention/) | All 21 kernel variants + 14 bench harnesses, full lineage |

## Headline arc

```
flash_attn.cu              (scalar O(N²), reference)
  ↓  +SHFL.BFLY warp reduce, online softmax
flash_attn_wmma.cu         (WMMA fragments, no tiling)
  ↓  +Br=16 inner tiling
flash_attn_br16.cu
  ↓  +Bc=128, register-resident V (no SMEM round-trip)
flash_attn_br16_regpv.cu   (~6.1 TFLOPS, current "fast" baseline)
  ↓  +cp.async double-buffer
flash_attn_br16_pipeline.cu, flash_attn_br16_v2_pipeline_pad.cu
  ↓  +persistent grid (1 block per SM)
flash_attn_persistent.cu, flash_attn_v2_persistent_pad.cu
  ↓  +split-Q (work in flight)
flash_attn_split_q.cu      (current frontier)
```

## Cross-references

- [docs/tutorial/05-flash-attention.md](../docs/tutorial/05-flash-attention.md) — full prose walkthrough including the 3 instructive failures
- [docs/gpu_reflections.md](../docs/gpu_reflections.md) — Obs C, D, E, F, J, JJ on attention-specific findings
- [docs/kernels.md](../docs/kernels.md) — measured numbers per variant
- [docs/comparison_to_sota.md](../docs/comparison_to_sota.md) — gap to FlashAttention-2 / cuDNN
