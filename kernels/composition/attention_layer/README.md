# kernels/composition/attention_layer — multi-head attention layer end-to-end

Composes QKV projection (HGEMM) → Flash Attention (phase 3 Br=16) →
output projection (HGEMM) → residual add into a single layer-level
benchmark. Validates that the per-kernel optimizations from earlier
phases stack when run back-to-back on the same on-chip resources.

## Files

| File | Purpose |
|---|---|
| `utils.cu`                | Layer composition: launches the four kernels in sequence |
| `utils.sm_86.cubin`       | compiled cubin |
| `bench.cu`                | layer-wide timing, per-stage breakdown, vs theoretical roofline |

## Build

```bash
nvcc -arch=sm_86 -O2 --cubin utils.cu -o utils.sm_86.cubin
nvcc -arch=sm_86 -O2 -o bench bench.cu -lcuda -I../../kernels/_common
./bench
```

## What this measures

For typical shapes (seq_len=1024, d_model=512, heads=8) the layer-wide
runtime breaks down approximately:

| Stage             | Share of runtime |
|-------------------|------------------|
| QKV projection    | ~35–40% (2 HGEMMs)  |
| Flash Attention   | ~25%             |
| Output projection | ~25%             |
| Residual + norm   | <5%              |

Conclusion: **phase-2 HGEMM dominates**, so layer-level wins come from
projection-throughput improvements, not from further attention-kernel
tuning. This motivates the phase-2 register-tile / SMEM-layout
investigations even though phase 3 holds the headline TFLOPS number.

## Cross-references

- [kernels/gemm/hgemm/README.md](../../gemm/hgemm/README.md) — projection backbone
- [kernels/attention/flash_attention/README.md](../../attention/flash_attention/README.md) — attention core
- [docs/comparison_to_sota.md](../../../docs/comparison_to_sota.md) — vs cuDNN MHA layer
