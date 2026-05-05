#!/usr/bin/env python3
"""
scripts/bench_flash_all.py — Unified Flash Attention benchmark harness.

Discovers and runs all Flash Attention bench executables in phase3/flash_attention/,
producing a markdown-formatted comparison table.

Usage:
    python3 scripts/bench_flash_all.py <seq_len> <batch> <heads>
    python3 scripts/bench_flash_all.py 1024 8 8
    python3 scripts/bench_flash_all.py --build 1024 8 8  # build missing benches first
"""

import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
FLASH_DIR = REPO_ROOT / "phase3" / "flash_attention"


def discover_benches():
    """Find all bench executables in flash_attention/."""
    benches = {}
    for f in FLASH_DIR.iterdir():
        if f.name.startswith("bench") and f.is_file() and f.suffix == "":
            # Map executable name to expected kernel name
            variant = f.name.replace("bench_", "").replace("bench", "flash_attn")
            benches[f.name] = str(f)
    return benches


def run_bench(exe_path, args):
    """Run a single bench executable and parse results."""
    result = subprocess.run(
        [exe_path] + args,
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = result.stdout + result.stderr

    metrics = {"raw": output, "returncode": result.returncode}

    # Parse timing and GFLOPS
    m = re.search(r"([\d.]+)\s*ms.*?(\d[\d,.]*)\s*GFLOPS", output, re.IGNORECASE)
    if m:
        metrics["ms"] = float(m.group(1))
        metrics["gflops"] = float(m.group(2).replace(",", ""))

    # Parse correctness
    if "PASS" in output:
        metrics["check"] = "PASS"
    elif "FAIL" in output:
        metrics["check"] = "FAIL"
    else:
        metrics["check"] = "?"

    return metrics


def print_table(results, seq_len, batch, heads):
    print("\n" + "=" * 70)
    print(f"  Flash Attention Comparison  (seq={seq_len}, batch={batch}, heads={heads})")
    print("=" * 70)
    print(f"  {'Variant':<28} {'Time(ms)':>10} {'GFLOPS':>12} {'Check':>8}")
    print(f"  {'-'*28} {'-'*10} {'-'*12} {'-'*8}")

    best_ms = min((r["ms"] for r in results if "ms" in r), default=0)

    for r in results:
        name = r["name"]
        if "ms" not in r:
            print(f"  {name:<28} {'—':>10} {'—':>12} {r.get('check', 'SKIP'):>8}")
            continue
        marker = "*" if abs(r["ms"] - best_ms) < 0.001 else " "
        print(f" {marker}{name:<27} {r['ms']:>10.3f} {r.get('gflops', 0):>12.0f} {r.get('check', '?'):>8}")

    print("=" * 70)
    print("  * = fastest variant")
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Flash Attention benchmark harness")
    parser.add_argument("seq_len", type=int, nargs="?", default=1024)
    parser.add_argument("batch", type=int, nargs="?", default=8)
    parser.add_argument("heads", type=int, nargs="?", default=8)
    parser.add_argument("--build", action="store_true", help="Build missing benchmarks via make")
    args = parser.parse_args()

    benches = discover_benches()
    if not benches:
        print("No bench executables found in phase3/flash_attention/")
        if args.build:
            print("Attempting to build...")
            subprocess.run(["make", "-C", str(FLASH_DIR.parent.parent), "phase3"])
            benches = discover_benches()
        if not benches:
            print("Run: make phase3")
            sys.exit(1)

    bench_args = [str(args.seq_len), str(args.batch), str(args.heads)]
    results = []

    # Define order for consistent output
    preferred_order = [
        "bench", "bench_br16", "bench_br16_regpv",
        "bench_br16_bc128", "bench_br16_pipeline",
        "bench_fused", "bench_persistent",
        "bench_split_q", "bench_wmma"
    ]

    ordered = []
    for name in preferred_order:
        if name in benches:
            ordered.append((name, benches.pop(name)))
    # Add any remaining
    ordered.extend(sorted(benches.items()))

    print(f"Running {len(ordered)} Flash Attention variants...")
    for name, exe_path in ordered:
        print(f"\n--- {name} ---")
        if not Path(exe_path).exists():
            results.append({"name": name, "check": "NOT_FOUND"})
            continue
        metrics = run_bench(exe_path, bench_args)
        metrics["name"] = name
        results.append(metrics)
        if "ms" in metrics:
            print(f"  {metrics['ms']:.3f} ms  {metrics.get('gflops', 0):.0f} GFLOPS  [{metrics['check']}]")

    print_table(results, args.seq_len, args.batch, args.heads)

    # Exit non-zero if any variant failed
    fails = sum(1 for r in results if r.get("check") == "FAIL")
    sys.exit(1 if fails > 0 else 0)


if __name__ == "__main__":
    main()
