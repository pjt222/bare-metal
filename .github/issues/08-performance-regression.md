---
title: "Create automated performance regression script"
labels: ["enhancement", "infrastructure", "testing"]
---

## Problem
Performance data in READMEs and `gpu_reflections.md` is static. As kernels evolve, documented numbers drift from actual code. There is no automated way to detect performance regressions.

## Proposed Solution

### Step 1: Record baselines
Create `docs/baselines.json`:
```json
{
  " recorded_date": "2026-05-05",
  "platform": "RTX 3070 Ti Laptop (GA104, sm_86)",
  "kernels": {
    "kernels/gemm/hgemm/hgemm_16warp.cu": {
      "2048_2048_2048": {"ms": 0.527, "gflops": 31910},
      "4096_4096_4096": {"ms": 4.22, "gflops": 31910}
    },
    "kernels/gemm/igemm/igemm_pipelined_cpasync.cu": {
      "4096_4096_4096": {"ms": 6.6, "tops": 20688}
    },
    "phase3/flash_attention/flash_attn_br16_regpv.cu": {
      "1024_8_8": {"ms": 2.81, "gflops": 6112}
    }
  }
}
```

### Step 2: Create `scripts/bench_regress.py`
```python
#!/usr/bin/env python3
"""Run benchmarks and compare against recorded baselines."""
import json, subprocess, sys

def run_benchmark(exe_path, args):
    result = subprocess.run([exe_path] + args, capture_output=True, text=True)
    # Parse output for GFLOPS/TOPS/ms
    return parse_bench_output(result.stdout)

def check_regression(current, baseline, tolerance=0.10):
    """Flag if current is >10% worse than baseline."""
    ratio = baseline / current  # >1 means regression
    if ratio < (1 - tolerance):
        return f"REGRESSION: {ratio:.2%} of baseline"
    return "OK"

if __name__ == "__main__":
    baselines = json.load(open("docs/baselines.json"))
    results = {}
    for kernel, configs in baselines["kernels"].items():
        exe = kernel.replace(".cu", "").replace("/", "_")
        for config, expected in configs.items():
            current = run_benchmark(f"./{exe}", config.split("_"))
            status = check_regression(current["gflops"], expected["gflops"])
            print(f"{kernel} {config}: {status}")
```

### Step 3: CI integration (future)
When GitHub Actions supports self-hosted runners with GPU, run `scripts/bench_regress.py` on every PR.

## Scope
Start with top 5 kernels (highest GEMM/attention throughput):
1. `kernels/gemm/hgemm/hgemm_16warp.cu`
2. `kernels/gemm/hgemm_sparse/hgemm_sparse_tiled.cu`
3. `kernels/gemm/igemm/igemm_pipelined_cpasync.cu`
4. `phase3/flash_attention/flash_attn_br16_regpv.cu`
5. `phase4/conv2d/conv2d_implicit_gemm.cu`

## Acceptance Criteria
- [ ] `docs/baselines.json` exists with at least 5 kernel configs
- [ ] `scripts/bench_regress.py` runs benchmarks and reports regressions
- [ ] Script exits non-zero if any benchmark regresses >10%
- [ ] Documented in top-level README: "How to check for regressions"

## Effort
Medium — 1 day for script, 1 day for baseline recording.
