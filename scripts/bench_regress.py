#!/usr/bin/env python3
"""
scripts/bench_regress.py — Automated performance regression checker.

Runs benchmark executables and compares against recorded baselines in
 docs/baselines.json. Exits non-zero if any kernel regresses beyond tolerance.

Usage:
    python3 scripts/bench_regress.py              # check all baselines
    python3 scripts/bench_regress.py --kernel phase2/hgemm/hgemm_16warp.cu
    python3 scripts/bench_regress.py --tolerance 0.15
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
BASELINES_PATH = REPO_ROOT / "docs" / "baselines.json"
DEFAULT_TOLERANCE = 0.10  # flag regression if >10% worse than baseline


def find_executable(kernel_path: str) -> Path | None:
    """Map kernel .cu path to likely bench executable path."""
    # Try exact basename without .cu
    base = Path(kernel_path).stem  # e.g. "hgemm_16warp"
    parent = Path(kernel_path).parent  # e.g. "phase2/hgemm"

    candidates = [
        parent / "bench",
        parent / f"bench_{base}",
        parent / base,
    ]
    # Also check if bench executable already exists from Makefile
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def run_benchmark(exe_path: Path, args: list[str]) -> dict:
    """Run benchmark executable and parse output for timing + throughput."""
    result = subprocess.run(
        [str(exe_path)] + args,
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = result.stdout + result.stderr

    metrics = {}
    # Parse common output patterns:
    #  "  0.527 ms   31910.00 GFLOPS"
    #  "  6.600 ms   20688.00 TOPS"
    m = re.search(r"([\d.]+)\s*ms.*?(\d[\d,.]*)\s*(GFLOPS|TOPS)", output, re.IGNORECASE)
    if m:
        metrics["ms"] = float(m.group(1))
        metrics["throughput"] = float(m.group(2).replace(",", ""))
        metrics["unit"] = m.group(3).upper()
    else:
        # Fallback: look for any ms and GFLOPS/TOPS separately
        m_ms = re.search(r"([\d.]+)\s*ms", output)
        m_tp = re.search(r"(\d[\d,.]*)\s*(GFLOPS|TOPS)", output, re.IGNORECASE)
        if m_ms:
            metrics["ms"] = float(m_ms.group(1))
        if m_tp:
            metrics["throughput"] = float(m_tp.group(1).replace(",", ""))
            metrics["unit"] = m_tp.group(2).upper()

    metrics["raw_output"] = output
    metrics["returncode"] = result.returncode
    return metrics


def check_regression(current: dict, baseline: dict, tolerance: float) -> tuple[bool, str]:
    """Compare current run against baseline. Returns (is_regression, message)."""
    if current.get("returncode", 0) != 0:
        return True, f"CRASH (exit={current['returncode']})"

    unit = current.get("unit", "GFLOPS")
    baseline_val = baseline.get(unit.lower(), baseline.get("gflops", baseline.get("tops", 0)))
    current_val = current.get("throughput", 0)

    if baseline_val == 0 or current_val == 0:
        return True, f"NO_DATA (baseline={baseline_val}, current={current_val})"

    ratio = current_val / baseline_val
    if ratio < (1.0 - tolerance):
        return True, f"REGRESSION {ratio:.1%} of baseline ({current_val:.0f} vs {baseline_val:.0f} {unit})"
    elif ratio > (1.0 + tolerance):
        return False, f"IMPROVED {ratio:.1%} of baseline ({current_val:.0f} vs {baseline_val:.0f} {unit})"
    else:
        return False, f"OK {ratio:.1%} of baseline ({current_val:.0f} vs {baseline_val:.0f} {unit})"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Performance regression checker")
    parser.add_argument("--kernel", help="Check only this kernel path")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE, help="Regression tolerance (default 0.10)")
    parser.add_argument("--list", action="store_true", help="List recorded baselines and exit")
    args = parser.parse_args()

    if not BASELINES_PATH.exists():
        print(f"ERROR: Baselines file not found: {BASELINES_PATH}")
        print("Run benchmarks manually and record results to docs/baselines.json")
        sys.exit(1)

    with open(BASELINES_PATH) as f:
        baselines = json.load(f)

    if args.list:
        print(f"Baselines recorded: {baselines.get('recorded_date', 'unknown')}")
        print(f"Platform: {baselines.get('platform', 'unknown')}")
        for kernel, configs in baselines.get("kernels", {}).items():
            print(f"\n{kernel}")
            for cfg, data in configs.items():
                unit = "GFLOPS" if "gflops" in data else "TOPS"
                print(f"  {cfg}: {data.get('ms', '?')} ms, {data.get(unit.lower(), '?')} {unit}")
        sys.exit(0)

    kernels = baselines.get("kernels", {})
    if args.kernel:
        if args.kernel not in kernels:
            print(f"ERROR: Kernel '{args.kernel}' not found in baselines")
            sys.exit(1)
        kernels = {args.kernel: kernels[args.kernel]}

    regressions = 0
    improvements = 0
    total = 0

    print("=" * 70)
    print(f"  Performance Regression Check")
    print(f"  Tolerance: {args.tolerance:.0%}")
    print(f"  Baselines: {baselines.get('recorded_date', 'unknown')}")
    print("=" * 70)

    for kernel_path, configs in kernels.items():
        exe = find_executable(kernel_path)
        if exe is None:
            print(f"\n{kernel_path}")
            print(f"  SKIP — executable not found (try: make benches)")
            continue

        for cfg, expected in configs.items():
            total += 1
            cfg_args = cfg.split("_")
            current = run_benchmark(exe, cfg_args)
            is_reg, msg = check_regression(current, expected, args.tolerance)

            print(f"\n{kernel_path} [{cfg}]")
            print(f"  {msg}")

            if is_reg:
                regressions += 1
            elif "IMPROVED" in msg:
                improvements += 1

    print("\n" + "=" * 70)
    print(f"  Total: {total} | Regressions: {regressions} | Improvements: {improvements}")
    if regressions > 0:
        print(f"  RESULT: FAILED — {regressions} regression(s) detected")
        sys.exit(1)
    else:
        print(f"  RESULT: PASSED — all benchmarks within tolerance")
        sys.exit(0)


if __name__ == "__main__":
    main()
