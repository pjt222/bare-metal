# Phase 5 — Multi-head attention layer composition

End-to-end **attention layer** assembled from phase 2/3/4 primitives:
QKV projection (HGEMM) → reshape → Flash Attention → output projection
(HGEMM) → residual add. Tests whether the optimizations from earlier
phases compose without on-chip-memory contention.

## Subdirectory

| Dir | Contents |
|---|---|
| [`attention_layer/`](attention_layer/) | Composition kernel + bench harness |

## Status

Working composition; runtime dominated by the two GEMM projections.
Detailed per-stage timing in `attention_layer/`.

## Cross-references

- [kernels/gemm/hgemm/README.md](../kernels/gemm/hgemm/README.md) — projection backbone
- [kernels/attention/flash_attention/README.md](../kernels/attention/flash_attention/README.md) — attention core
- [docs/kernels.md](../docs/kernels.md) — measured layer-wide numbers
