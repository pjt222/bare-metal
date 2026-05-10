#!/usr/bin/env bash
# ncu_profile_all.sh — profile representative kernels across phase2-phase4.
#
# Requires: GPU performance counters enabled (see ncu_profile.R header).
# Output:   results/ncu/all.csv + per-row markdown to stdout.
#
# Usage:    bash scripts/profile/ncu_profile_all.sh
#           bash scripts/profile/ncu_profile_all.sh --tag fa_only   # subset

set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_CSV="$ROOT/results/ncu/all.csv"
mkdir -p "$(dirname "$OUT_CSV")"

# Each row: kernel_name | bench_path | bench_args | label | working_dir
# Working dir matters: many benches load cubins via relative paths.
ROWS=(
  # --- Flash Attention (phase 3) ---
  "flash_attn_br16_v2|kernels/attention/flash_attention/bench_v2_variants|1024 8 8|FA v2 baseline (seq=1024,b=8,h=8)|kernels/attention/flash_attention"
  "flash_attn_br16_v2_pipeline|kernels/attention/flash_attention/bench_v2_variants|1024 8 8|FA v2 pipeline (seq=1024,b=8,h=8)|kernels/attention/flash_attention"
  "flash_attn_v2_persistent|kernels/attention/flash_attention/bench_v2_variants|1024 8 8|FA v2 persistent (seq=1024,b=8,h=8)|kernels/attention/flash_attention"
  "flash_attn_br16_regpv|kernels/attention/flash_attention/bench_br16_regpv|1024 8 8|FA regpv (legacy, seq=1024,b=8,h=8)|kernels/attention/flash_attention"

  # --- HGEMM (phase 2) ---
  "hgemm_16warp|kernels/gemm/hgemm/bench|4096 4096 4096|HGEMM 16-warp (4096³)|kernels/gemm/hgemm"
  "hgemm_16warp_epi|kernels/gemm/hgemm/bench|4096 4096 4096|HGEMM 16-warp+epi (4096³)|kernels/gemm/hgemm"
  "hgemm_256x128|kernels/gemm/hgemm/bench|4096 4096 4096|HGEMM 256x128 (4096³)|kernels/gemm/hgemm"

  # --- IGEMM (phase 2) ---
  "igemm_sparse_tiled|kernels/gemm/igemm/bench_sparse|4096 4096 4096|Sparse INT8 GEMM (4096³)|kernels/gemm/igemm"

  # --- Cross-Attention (phase 4) ---
  "cross_attn_v2|kernels/attention/cross_attention/bench_v2|1024 256 8|Cross-attn v2 (1024 q, 256 kv, h=8)|kernels/attention/cross_attention"

  # --- ResBlock (phase 4) ---
  "implicit_gemm_conv|kernels/convolution/resblock/bench_implicit|1 320 32 32|ResBlock implicit GEMM (SD UNet 320ch)|kernels/convolution/resblock"
)

# Wipe old CSV so columns/rows match the current run.
rm -f "$OUT_CSV"

FAILED=0
for row in "${ROWS[@]}"; do
  IFS='|' read -r kernel bench args label wd <<<"$row"
  bench_abs="$ROOT/$bench"
  if [[ ! -x "$bench_abs" ]]; then
    echo "SKIP: $bench_abs not built"
    continue
  fi
  echo "=== $label ==="
  pushd "$ROOT/$wd" >/dev/null
  if ! Rscript "$ROOT/scripts/profile/ncu_profile.R" \
        --kernel "$kernel" \
        --bench "$bench_abs" \
        --args "$args" \
        --label "$label" \
        --launch-skip 5 --launch-count 1 \
        --out "$OUT_CSV"; then
    echo "FAIL: $label"
    FAILED=$((FAILED + 1))
  fi
  popd >/dev/null
done

echo
echo "Done. Output CSV: $OUT_CSV"
[[ $FAILED -eq 0 ]] || { echo "$FAILED row(s) failed"; exit 1; }
