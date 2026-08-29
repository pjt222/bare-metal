#!/usr/bin/env bash
# scripts/probe/gpu_power_state.sh
#
# WSL wrapper for gpu_power_state.ps1 (issue #207).
#
# Reports the platform state that governs the dGPU power envelope, so a
# benchmark session can prove it is comparable to a recorded baseline.
# `nvidia-smi`'s power.draw cannot show this -- it reports what the GPU
# drew, never what it was allowed to draw.
#
# The nvidia-smi and registry halves need no elevation. The Lenovo
# SmartFanMode half lives in root\WMI and returns "access denied"
# without Administrator, so --elevated re-launches the script through
# `Start-Process -Verb RunAs`, which raises a UAC prompt on the desktop.
# That prompt is interactive: the run blocks until it is answered.
#
# Usage:
#   scripts/probe/gpu_power_state.sh                    # report, no UAC
#   scripts/probe/gpu_power_state.sh --elevated         # full report, UAC
#   scripts/probe/gpu_power_state.sh --elevated --set-mode 3
#   scripts/probe/gpu_power_state.sh --json
#
# SmartFanMode: 1 Quiet, 2 Balanced, 3 Performance, 255 Custom.
#
# Exit status: 0 report produced; 1 the script could not be run; 2 the
# GPU is power-capped (enforced limit below the VBIOS ceiling) -- so a
# caller can gate a benchmark on `gpu_power_state.sh || exit`.

set -uo pipefail

ELEVATED=0
SET_MODE=0
JSON=0

while [ $# -gt 0 ]; do
  case "$1" in
    --elevated) ELEVATED=1; shift ;;
    --set-mode) SET_MODE="$2"; shift 2 ;;
    --json)     JSON=1; shift ;;
    -h|--help)  sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

here_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ps1_wsl="$here_dir/gpu_power_state.ps1"
if [ ! -f "$ps1_wsl" ]; then
  echo "not found: $ps1_wsl" >&2
  exit 1
fi

powershell_exe="$(command -v powershell.exe || true)"
if [ -z "$powershell_exe" ]; then
  echo "powershell.exe not on PATH -- this tool is WSL-on-Windows only." >&2
  exit 1
fi

# Windows needs a Windows path for -File.
ps1_win="$(wslpath -w "$ps1_wsl")"

# The elevated child is a separate process with its own stdout, which does
# not come back to this terminal -- so both paths route the report through
# the log file and we read that. TEMP is used rather than the repo so a
# diagnostic run never dirties the working tree.
log_win="$(powershell.exe -NoProfile -Command 'Write-Output $env:TEMP' 2>/dev/null | tr -d '\r')\\gpu_power_state.log"
log_wsl="$(wslpath -u "$log_win")"
rm -f "$log_wsl" 2>/dev/null || true

# An ARRAY, not a string. A quoted string expanded unquoted gets its
# quotes passed through as literal characters -- PowerShell then sees
# -File ""D:\...\gpu_power_state.ps1"" and refuses it with "Illegales
# Zeichen im Pfad". Expanded unquoted instead, any space in the repo path
# would split the argument. The array is correct on both counts.
args=(-NoProfile -ExecutionPolicy Bypass -File "$ps1_win")
if [ "$SET_MODE" != "0" ]; then
  args+=(-SetMode "$SET_MODE")
fi
if [ "$JSON" = "1" ]; then
  args+=(-Json)
fi

if [ "$ELEVATED" = "1" ]; then
  echo "Requesting elevation -- answer the UAC prompt on the desktop." >&2
  # -Wait so this returns only once the elevated child has finished and
  # the log is complete.
  powershell.exe -NoProfile -Command \
    "Start-Process powershell -Verb RunAs -Wait -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','$ps1_win'$( [ "$SET_MODE" != "0" ] && echo ",'-SetMode','$SET_MODE'" )" \
    >/dev/null 2>&1
  rc=$?
  if [ ! -f "$log_wsl" ]; then
    echo "no log at $log_wsl -- UAC was probably declined." >&2
    exit 1
  fi
  cat "$log_wsl"
else
  powershell.exe "${args[@]}" 2>&1 | tr -d '\r'
  rc=${PIPESTATUS[0]:-0}
fi

# Re-read the enforced limit here rather than parsing the log: this is the
# value the exit status promises, and reading it directly keeps the
# contract independent of the report's formatting.
read -r enforced maxw < <(nvidia-smi \
  --query-gpu=enforced.power.limit,power.max_limit \
  --format=csv,noheader,nounits 2>/dev/null | tr -d ',' | awk '{print $1, $2}')

if [ -n "${enforced:-}" ] && [ -n "${maxw:-}" ]; then
  if awk -v e="$enforced" -v m="$maxw" 'BEGIN{exit !(e < m - 1)}'; then
    echo "" >&2
    echo "POWER-CAPPED: enforced ${enforced} W < ceiling ${maxw} W." >&2
    echo "Benchmarks taken now are NOT comparable to recorded baselines." >&2
    exit 2
  fi
fi

exit 0
