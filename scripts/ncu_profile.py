#!/usr/bin/env python3
"""
ncu_profile.py — Wrap `ncu` for one kernel + bench, emit CSV + markdown row.

Usage:
    python scripts/ncu_profile.py \
        --kernel flash_attn_br16_v2_pipeline \
        --bench phase3/flash_attention/bench_v2_variants \
        --args "1024 8 8" \
        --label "FA v2 pipeline (seq=1024, b=8, h=8)" \
        --launch-skip 5 --launch-count 1 \
        --out results/ncu/fa_v2_pipeline.csv

Permissions: NCU needs GPU performance counters enabled. On WSL2/Windows host:
  NVIDIA Control Panel → Desktop menu → Enable Developer Settings →
  Manage GPU Performance Counters → "Allow access to all users". Reboot.

On Linux native:
  sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0
"""

import argparse
import csv
import io
import os
import subprocess
import sys
from pathlib import Path

# Default metric set per issue #89 acceptance criteria.
# Each entry: (ncu metric name, short column label, "higher_better" hint)
METRICS = [
    ("sm__warps_active.avg.pct_of_peak_sustained_active",
     "occupancy_pct", True),
    ("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
     "tc_util_pct", True),
    ("l1tex__t_sector_hit_rate.pct",
     "l1_hit_pct", True),
    ("lts__t_sector_hit_rate.pct",
     "l2_hit_pct", True),
    ("dram__bytes_read.sum.per_second",
     "dram_read_bw", True),
    ("dram__bytes_write.sum.per_second",
     "dram_write_bw", True),
    ("smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio",
     "load_coalesce_bytes", True),
    # Stall reason histogram (per-issue cycles) — top contenders for SOL gap.
    ("smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
     "stall_long_sb", False),
    ("smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
     "stall_short_sb", False),
    ("smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
     "stall_wait", False),
    ("smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
     "stall_mio", False),
    ("smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
     "stall_lg_throttle", False),
    ("smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
     "stall_barrier", False),
    ("smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
     "stall_math_throttle", False),
    ("smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio",
     "stall_tex_throttle", False),
]

CUDA_BIN = "/usr/local/cuda/bin"
NCU = os.path.join(CUDA_BIN, "ncu")


def ensure_path() -> None:
    """Make sure /usr/local/cuda/bin is on PATH for child processes too."""
    if CUDA_BIN not in os.environ.get("PATH", ""):
        os.environ["PATH"] = CUDA_BIN + ":" + os.environ.get("PATH", "")


def run_ncu(kernel: str, bench: str, bench_args: list[str],
            launch_skip: int, launch_count: int) -> str:
    """Invoke ncu, return raw CSV output. Raises on permission/binary errors."""
    metric_arg = ",".join(m[0] for m in METRICS)
    cmd = [
        NCU,
        "--csv",
        "--kernel-name", kernel,
        "--launch-skip", str(launch_skip),
        "--launch-count", str(launch_count),
        "--metrics", metric_arg,
        "--target-processes", "all",
        bench,
    ] + bench_args

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(
            f"ncu failed (rc={proc.returncode}). "
            "Check ERR_NVGPUCTRPERM if counter permission error."
        )
    return proc.stdout


def parse_ncu_csv(raw: str) -> dict[str, str]:
    """
    NCU CSV is one row per (kernel-instance × metric). We collapse to a single
    dict keyed by metric name → value. If multiple kernel instances were
    captured, average numeric values; otherwise take the last.
    """
    # Strip ncu banner lines (anything before the first line containing "ID")
    lines = raw.splitlines()
    header_idx = next(
        (i for i, ln in enumerate(lines) if ln.startswith('"ID"') or ln.startswith("ID,")),
        None,
    )
    if header_idx is None:
        raise SystemExit("Could not find CSV header in ncu output")
    body = "\n".join(lines[header_idx:])
    reader = csv.DictReader(io.StringIO(body))
    accum: dict[str, list[float]] = {}
    raw_units: dict[str, str] = {}
    for row in reader:
        metric = row.get("Metric Name", "").strip()
        value_s = row.get("Metric Value", "").strip().replace(",", "")
        unit = row.get("Metric Unit", "").strip()
        if not metric:
            continue
        try:
            v = float(value_s)
        except ValueError:
            v = float("nan")
        accum.setdefault(metric, []).append(v)
        raw_units[metric] = unit
    out = {}
    for metric, vals in accum.items():
        finite = [v for v in vals if v == v]  # filter NaN
        avg = sum(finite) / len(finite) if finite else float("nan")
        out[metric] = (avg, raw_units.get(metric, ""))
    return out


def write_csv(path: Path, label: str, parsed: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            cols = ["label"] + [m[1] for m in METRICS]
            w.writerow(cols)
        row = [label]
        for metric, short, _ in METRICS:
            val, _unit = parsed.get(metric, (float("nan"), ""))
            row.append(f"{val:.4g}" if val == val else "NaN")
        w.writerow(row)


def print_markdown(label: str, parsed: dict) -> None:
    print(f"\n### {label}\n")
    print("| metric | value |")
    print("|---|---|")
    for metric, short, _ in METRICS:
        val, unit = parsed.get(metric, (float("nan"), ""))
        v_str = f"{val:.4g}" if val == val else "NaN"
        if unit:
            v_str = f"{v_str} {unit}"
        print(f"| `{short}` | {v_str} |")


def main() -> None:
    ensure_path()
    p = argparse.ArgumentParser()
    p.add_argument("--kernel", required=True,
                   help="Kernel name pattern (regex okay, see ncu --kernel-name)")
    p.add_argument("--bench", required=True,
                   help="Path to compiled bench binary")
    p.add_argument("--args", default="",
                   help="Space-separated bench args, e.g. '1024 8 8'")
    p.add_argument("--label", required=True,
                   help="Human-readable label for this configuration")
    p.add_argument("--launch-skip", type=int, default=5,
                   help="Skip first N kernel launches (warmup)")
    p.add_argument("--launch-count", type=int, default=1,
                   help="Capture this many launches and average")
    p.add_argument("--out", default="results/ncu/profile.csv",
                   help="Append CSV row here")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the ncu command that would run, don't execute")
    args = p.parse_args()

    if not Path(args.bench).is_file() or not os.access(args.bench, os.X_OK):
        raise SystemExit(f"Bench binary not found or not executable: {args.bench}")

    bench_args = args.args.split() if args.args else []
    if args.dry_run:
        metric_arg = ",".join(m[0] for m in METRICS)
        cmd = [NCU, "--csv", "--kernel-name", args.kernel,
               "--launch-skip", str(args.launch_skip),
               "--launch-count", str(args.launch_count),
               "--metrics", metric_arg,
               "--target-processes", "all",
               args.bench] + bench_args
        print(" ".join(cmd))
        return
    raw = run_ncu(args.kernel, args.bench, bench_args,
                  args.launch_skip, args.launch_count)
    parsed = parse_ncu_csv(raw)
    print_markdown(args.label, parsed)
    write_csv(Path(args.out), args.label, parsed)
    print(f"\nAppended row to {args.out}")


if __name__ == "__main__":
    main()
