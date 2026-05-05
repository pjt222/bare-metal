---
title: "Create unified Flash Attention benchmark harness"
labels: ["enhancement", "flash-attention"]
---

## Problem
Phase 3 has 10+ Flash Attention variants (scalar, 4-warp, br16, bc128, pipeline, fused, persistent, split-q, wmma, regpv). Each has its own `bench_*.cu`. There is no single command that runs all variants and produces a comparison table.

Determining "current best" requires reading `docs/gpu_reflections.md` or manually compiling and running 10 separate binaries.

## Variants Inventory
| File | Kernel | Key Feature |
|------|--------|-------------|
| `flash_attn.cu` | `flash_attn_1warp` | Scalar baseline |
| `flash_attn_br16.cu` | `flash_attn_br16` | HMMA QK^T + PV |
| `flash_attn_br16_bc128.cu` | `flash_attn_br16_bc128` | Bc=128 (slower: occupancy cliff) |
| `flash_attn_br16_pipeline.cu` | `flash_attn_br16_pipeline` | cp.async (slower: overhead) |
| `flash_attn_br16_regpv.cu` | `flash_attn_br16_regpv` | Register-resident PV (+39%) |
| `flash_attn_fused.cu` | `flash_attn_fused` | Fused variant |
| `flash_attn_persistent.cu` | `flash_attn_persistent` | Persistent grid |
| `flash_attn_split_q.cu` | `flash_attn_split_q` | Split-Q L2 reuse |
| `flash_attn_wmma.cu` | `flash_attn_wmma` | WMMA path |

Plus 10 corresponding `bench_*.cu` files.

## Proposed Solution
Create `phase3/flash_attention/bench_all.cu`:
```cpp
// Single binary: ./bench_all <seq_len> <batch> <heads>
// Compiles ALL kernels, runs each, prints comparison table

int main(...) {
    for (auto &variant : flash_variants) {
        float ms = variant.run(params);
        double gflops = compute_flash_gflops(params, ms);
        results.push_back({variant.name, ms, gflops});
    }
    print_comparison_table(results);
}
```

Or: Python script `scripts/bench_flash.py` that discovers and runs compiled `bench_*` executables.

## Acceptance Criteria
- [ ] Single command runs all Flash Attention variants
- [ ] Outputs markdown-formatted comparison table (copy-paste to README)
- [ ] Skips variants that failed compilation
- [ ] Records best variant per config for quick reference

## Effort
Medium — 1 day for harness, 1 day for integration/testing.
