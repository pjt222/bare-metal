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
# Usage (run with bash, as the repo's other .sh scripts are documented):
#   bash scripts/probe/gpu_power_state.sh                    # report, no UAC
#   bash scripts/probe/gpu_power_state.sh --elevated         # full report, UAC
#   bash scripts/probe/gpu_power_state.sh --elevated --set-mode 3
#   bash scripts/probe/gpu_power_state.sh --json
#   bash scripts/probe/gpu_power_state.sh --reference default
#
# Not `sh`: /bin/sh is dash here, which rejects `set -o pipefail` and exits
# 2 -- the same code as "capped", so a gate would misread the failure. The
# file is committed 100644 like every .sh in the repo; a bare invocation
# works only on the /mnt/d drvfs mount, which shows every file as
# executable.
#
# SmartFanMode: 1 Quiet, 2 Balanced, 3 Performance, 255 Custom.
#
# --reference max|default picks what "capped" is measured against
# (default: max). `max` fits this laptop, whose normal operating point is
# the 150 W VBIOS ceiling an OEM performance mode raises it to; there even
# the documented 115 W default is a 23% cut. `default` fits stock hardware
# whose normal operating point IS the default (e.g. a desktop card at
# 220 / 220 / 250 W), which `max` would report as capped forever. The same
# two references #207 records in the run record (power_below_max /
# power_below_default).
#
# Exit status: 0 report produced and the GPU is at its reference; 1 the
# script could not be run, or powershell.exe exited non-zero so the report
# may be partial; 2 the GPU is power-capped (enforced limit below the
# chosen reference); 3 the power limit could not be read, so nothing is
# certified.
# Once a report exists, 3 and 2 take precedence over a partial one (1).
# Failures BEFORE any power read -- bad arguments, an unresolvable %TEMP%,
# a declined or failed elevation -- exit 1 directly, so a declined UAC
# prompt on a capped GPU returns 1, not 2. Every non-zero status means
# "do not benchmark", so a caller can gate a run on
# `bash scripts/probe/gpu_power_state.sh || exit`.

set -uo pipefail

ELEVATED=0
SET_MODE=0
JSON=0
REFERENCE=max

while [ $# -gt 0 ]; do
  case "$1" in
    --elevated) ELEVATED=1; shift ;;
    --set-mode)
      # Validated here, not only by the .ps1's ValidateSet: the value is
      # handed to an ELEVATED process, and anything that reaches its command
      # line unvalidated is a way in.
      case "${2:-}" in
        1|2|3|255) SET_MODE="$2" ;;
        *) echo "--set-mode must be 1, 2, 3 or 255 (got '${2:-}')" >&2; exit 1 ;;
      esac
      shift 2 ;;
    --json)     JSON=1; shift ;;
    --reference)
      case "${2:-}" in
        max|default) REFERENCE="$2" ;;
        *) echo "--reference must be max or default (got '${2:-}')" >&2; exit 1 ;;
      esac
      shift 2 ;;
    -h|--help)  awk 'NR > 1 && /^#/ { print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

# Setting SmartFanMode needs root\WMI, which needs elevation. Without
# --elevated the .ps1 skips the WMI section, so --set-mode was silently
# ignored and the run exited 0 as if it had been applied.
if [ "$SET_MODE" != "0" ] && [ "$ELEVATED" != "1" ]; then
  echo "--set-mode needs --elevated (root\\WMI requires Administrator); nothing was changed." >&2
  exit 1
fi

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
# not come back to this terminal -- so the ELEVATED path reads the report
# back from the log (or, with --json, the JSON file) that the child writes.
# The non-elevated path prints PowerShell's stdout directly; it writes the
# log too but never reads it. TEMP is used rather than the repo so a
# diagnostic run never dirties the working tree.
temp_win="$(powershell.exe -NoProfile -Command 'Write-Output $env:TEMP' 2>/dev/null | tr -d '\r')"
# An empty or rootless TEMP must stop here. "\gpu_power_state.log" is a
# valid Windows path relative to the CURRENT DRIVE's root: wslpath maps it
# to /mnt/<drive>/gpu_power_state.log, and the rm lines below would delete
# files at a drive root. Accept only a drive path (C:\...) or a UNC path.
if ! [[ "$temp_win" =~ ^([A-Za-z]:\\|\\\\) ]]; then
  echo "could not resolve %TEMP% from Windows (got '${temp_win}')." >&2
  exit 1
fi
log_win="${temp_win}\\gpu_power_state.log"
log_wsl="$(wslpath -u "$log_win")"
rm -f "$log_wsl" 2>/dev/null || true
# The JSON counterpart, for --elevated --json: the elevated child writes
# its JSON object here because its stdout cannot reach this terminal.
json_win="${temp_win}\\gpu_power_state.json"
json_wsl="$(wslpath -u "$json_win")"
rm -f "$json_wsl" 2>/dev/null || true

# An ARRAY, not a string. A quoted string expanded unquoted gets its
# quotes passed through as literal characters -- PowerShell then sees
# -File ""D:\...\gpu_power_state.ps1"" and refuses it with "Illegales
# Zeichen im Pfad". Expanded unquoted instead, any space in the repo path
# would split the argument. The array is correct on both counts.
args=(-NoProfile -ExecutionPolicy Bypass -File "$ps1_win" -LogPath "$log_win")
if [ "$SET_MODE" != "0" ]; then
  args+=(-SetMode "$SET_MODE")
fi
if [ "$JSON" = "1" ]; then
  args+=(-Json)
fi
args+=(-Reference "$REFERENCE")

if [ "$ELEVATED" = "1" ]; then
  echo "Requesting elevation -- answer the UAC prompt on the desktop." >&2
  # Nothing from this shell is spliced into the outer -Command string. It
  # used to be: the repo path sat inside a PowerShell '...' literal, so a
  # ' in the path ended the string and the rest ran as code -- in the
  # process that builds the ELEVATED child's command line. Now the values
  # travel as environment variables (WSLENV flag /w = WSL -> Win32) and
  # the -Command text is a fixed bash single-quoted literal.
  #
  # Start-Process joins -ArgumentList with bare spaces and quotes nothing,
  # so the path is wrapped in double quotes ([char]34) for the child's
  # command line; a Windows path cannot itself contain a double quote.
  #
  # -Wait so this returns only once the elevated child has finished and
  # the log is complete.
  elevate_ps='
    $a = @("-NoProfile", "-ExecutionPolicy", "Bypass",
           "-File", ([char]34 + $env:GPS_PS1 + [char]34),
           "-LogPath", ([char]34 + $env:GPS_LOG_PATH + [char]34))
    if ($env:GPS_SET_MODE -and $env:GPS_SET_MODE -ne "0") {
        $a += @("-SetMode", $env:GPS_SET_MODE)
    }
    if ($env:GPS_JSON_PATH) {
        $a += @("-Json", "-JsonPath", ([char]34 + $env:GPS_JSON_PATH + [char]34))
    }
    $a += @("-Reference", $env:GPS_REFERENCE)
    Start-Process powershell -Verb RunAs -Wait -ArgumentList $a'
  # --json must survive the relaunch: it used to be dropped here, because
  # this branch built its own argument list and never added -Json.
  gps_json_path=""
  if [ "$JSON" = "1" ]; then gps_json_path="$json_win"; fi
  # stderr is kept, not discarded: when the relaunch fails, PowerShell's own
  # message (e.g. "The operation was canceled by the user") is the
  # diagnosis. rc is the OUTER process's status -- whether Start-Process
  # succeeded. The elevated child's own exit status is not propagated; the
  # log it writes is the evidence that it ran.
  # The log path is passed explicitly (GPS_LOG_PATH), like the JSON path,
  # rather than assumed: the child's default would be ITS %TEMP%, which
  # need not be this one (e.g. elevation as a different admin account).
  elevate_err="$(GPS_PS1="$ps1_win" GPS_SET_MODE="$SET_MODE" GPS_JSON_PATH="$gps_json_path" \
    GPS_REFERENCE="$REFERENCE" GPS_LOG_PATH="$log_win" \
    WSLENV="${WSLENV:+$WSLENV:}GPS_PS1/w:GPS_SET_MODE/w:GPS_JSON_PATH/w:GPS_REFERENCE/w:GPS_LOG_PATH/w" \
    powershell.exe -NoProfile -Command "$elevate_ps" 2>&1 >/dev/null)"
  rc=$?
  elevate_err="${elevate_err//$'\r'/}"
  if [ ! -f "$log_wsl" ]; then
    # "Did not complete", not "never ran": the wrapper cannot tell a
    # declined prompt from a child that ran and then failed to write the
    # log. The difference matters with --set-mode, which may already have
    # changed the thermal mode.
    echo "no log at $log_wsl -- the elevated child did not complete (powershell.exe" >&2
    echo "exit $rc). A declined UAC prompt is the usual cause; if it ran and failed to" >&2
    echo "write the log, a requested --set-mode may still have been applied." >&2
    if [ -n "$elevate_err" ]; then printf '%s\n' "$elevate_err" >&2; fi
    exit 1
  fi
  if [ "$JSON" = "1" ]; then
    if [ ! -f "$json_wsl" ]; then
      echo "no JSON at $json_wsl although the elevated child ran (see $log_wsl)." >&2
      exit 1
    fi
    cat "$json_wsl"; echo
  else
    cat "$log_wsl"
  fi
else
  # stdout and stderr stay separate (both CR-stripped). Merged with 2>&1,
  # any PowerShell error text landed in --json output ahead of the object.
  powershell.exe "${args[@]}" 2> >(tr -d '\r' >&2) | tr -d '\r'
  rc=${PIPESTATUS[0]:-0}
fi

# A non-zero PowerShell status means the report above may be partial (a
# parse error, a blocked execution policy). It used to be captured and
# never read, so a broken report still exited 0. It is reported now, and
# turns into exit 1 below unless the power read has something more
# specific to say (3 unreadable, 2 capped).
report_failed=0
if [ "${rc:-0}" != "0" ]; then
  echo "" >&2
  echo "REPORT-INCOMPLETE: powershell.exe exited $rc -- the report above may be partial." >&2
  if [ -n "${elevate_err:-}" ]; then printf '%s\n' "$elevate_err" >&2; fi
  report_failed=1
fi

# Re-read the enforced limit here rather than parsing the log: this is the
# value the exit status promises, and reading it directly keeps the
# contract independent of the report's formatting.
#
# Split on COMMAS, not whitespace. A field such as "[Not Supported]"
# contains a space, and a whitespace split shifts every later field:
# "50.00, [Not Supported], 150.00" read max as "Supported]". Spaces and
# CRs are stripped first, so that field arrives as "[NotSupported]", which
# the numeric check below rejects as it should.
IFS=, read -r enforced defaultw maxw < <(nvidia-smi \
  --query-gpu=enforced.power.limit,power.default_limit,power.max_limit \
  --format=csv,noheader,nounits 2>/dev/null | tr -d ' \r')

# Fail CLOSED when the limit cannot be read. "Could not observe" is not "not
# capped": a gate that exits 0 here certifies a run nobody checked, which
# inverts the NA-vs-FALSE convention #207 set for the run record. Numeric
# checks matter too -- a driver that reports "[N/A]" yields a non-empty
# string that awk would silently compare as 0.
is_number() { [[ "${1:-}" =~ ^[0-9]+(\.[0-9]+)?$ ]]; }
if [ "$REFERENCE" = "default" ]; then reference_w="${defaultw:-}"; else reference_w="${maxw:-}"; fi
# A reference of 0 W is not a limit, it is a driver that reported nothing
# useful. Compared as a number it would read "not capped" (e < 0 - 1 is
# false), so it is UNKNOWN here, as in the .ps1 ($refw -gt 0).
if ! is_number "${enforced:-}" || ! is_number "$reference_w" ||
   ! awk -v r="$reference_w" 'BEGIN{exit !(r > 0)}'; then
  echo "" >&2
  echo "POWER-UNKNOWN: could not read the enforced/${REFERENCE} power limit from nvidia-smi" >&2
  echo "(got enforced='${enforced:-}' ${REFERENCE}='${reference_w}'). Treat this as NOT comparable." >&2
  exit 3
fi

if awk -v e="$enforced" -v r="$reference_w" 'BEGIN{exit !(e < r - 1)}'; then
  echo "" >&2
  echo "POWER-CAPPED: enforced ${enforced} W < ${REFERENCE} ${reference_w} W." >&2
  echo "Benchmarks taken now are NOT comparable to recorded baselines." >&2
  exit 2
fi

if [ "$report_failed" = "1" ]; then
  exit 1
fi

exit 0
