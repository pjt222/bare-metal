<#
.SYNOPSIS
  Reusable elevated-shell driver for clock-locked GPU evaluations.

.DESCRIPTION
  Locks the SM clock host-side via nvidia-smi.exe -lgc, invokes
  scripts/bench/bench_regress.R inside WSL with --clock-locked, and
  restores the default clock policy in a finally block. Captures
  metadata (driver, GPU, mode, observed clock, repo commit) and the
  full bench stdout/stderr to a timestamped log; appends a structured
  one-line record to scripts/probe/eval_logs/results.jsonl.

  Must be run from an elevated (Administrator) PowerShell. Without
  admin rights nvidia-smi.exe -lgc returns "Insufficient Permissions".

.PARAMETER ClockMHz
  Locked SM clock target in MHz. Defaults to 1605 (the igemm 4096^3
  clock_lock entry).

.PARAMETER Kernel
  Kernel .cu path relative to the repo root. Defaults to
  kernels/gemm/igemm/igemm_sparse_tiled.cu.

.PARAMETER RepoRoot
  Windows path to the repo. Defaults to the directory two levels above
  this script.

.PARAMETER Distro
  WSL distro name. Empty -> default distro.

.PARAMETER NoLock
  Skip the clock lock/restore. Useful for dry-running the logging path
  or for measurements where the clock is already locked externally.

.EXAMPLE
  # default: lock 1605, measure igemm 4096^3
  .\run_locked_eval.ps1

.EXAMPLE
  .\run_locked_eval.ps1 -ClockMHz 1500 -Kernel kernels/gemm/igemm/igemm_sparse_tiled.cu
#>

[CmdletBinding()]
param(
  [int]    $ClockMHz = 1605,
  [string] $Kernel   = "kernels/gemm/igemm/igemm_sparse_tiled.cu",
  [string] $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
  [string] $Distro   = "",
  [switch] $NoLock
)

$ErrorActionPreference = "Stop"
# PS 7+: native exes (nvidia-smi.exe, wsl.exe) throw on non-zero exit
# instead of needing manual $LASTEXITCODE checks. No-op on PS 5.1.
$PSNativeCommandUseErrorActionPreference = $true

# ---------------------------------------------------------------------------
# Pre-flight: elevation + tool availability
# ---------------------------------------------------------------------------
function Assert-Elevated {
  $id = [Security.Principal.WindowsIdentity]::GetCurrent()
  $pr = New-Object Security.Principal.WindowsPrincipal($id)
  if (-not $pr.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "This script requires an elevated (Administrator) PowerShell. nvidia-smi.exe -lgc will be rejected otherwise."
  }
}

function Assert-Tool($name) {
  if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
    throw "Required tool not on PATH: $name"
  }
}

if (-not $NoLock) { Assert-Elevated }
Assert-Tool "nvidia-smi.exe"
Assert-Tool "wsl.exe"

if (-not (Test-Path $RepoRoot)) {
  throw "Repo root not found: $RepoRoot"
}

# ---------------------------------------------------------------------------
# Path translation Windows -> WSL
# ---------------------------------------------------------------------------
function ConvertTo-WslPath($winPath) {
  # Local transform avoids wsl.exe arg-passing backslash mangling.
  # D:\dev\p\bare-metal -> /mnt/d/dev/p/bare-metal
  $full = (Resolve-Path $winPath).Path
  if ($full -match '^([A-Za-z]):[\\/](.*)$') {
    $drive = $Matches[1].ToLower()
    $rest  = $Matches[2] -replace '\\', '/'
    return "/mnt/$drive/$rest"
  }
  throw "Cannot translate path to WSL form: $winPath"
}

$RepoWsl = ConvertTo-WslPath $RepoRoot
Write-Host "Repo (win): $RepoRoot"
Write-Host "Repo (wsl): $RepoWsl"

# ---------------------------------------------------------------------------
# Metadata capture helpers
# ---------------------------------------------------------------------------
function Get-NvidiaSmiQuery([string]$query) {
  try {
    $out = & nvidia-smi.exe --query-gpu=$query --format=csv,noheader,nounits 2>&1
    if ($LASTEXITCODE -ne 0) { return $null }
    return ($out | ForEach-Object { $_.Trim() }) -join "; "
  } catch {
    return $null
  }
}

function Get-GpuSnapshot {
  [PSCustomObject]@{
    driver_version = Get-NvidiaSmiQuery "driver_version"
    gpu_name       = Get-NvidiaSmiQuery "name"
    gpu_uuid       = Get-NvidiaSmiQuery "uuid"
    clock_sm_mhz   = Get-NvidiaSmiQuery "clocks.current.sm"
    clock_mem_mhz  = Get-NvidiaSmiQuery "clocks.current.memory"
    power_w        = Get-NvidiaSmiQuery "power.draw"
    power_limit_w  = Get-NvidiaSmiQuery "power.limit"
    temp_c         = Get-NvidiaSmiQuery "temperature.gpu"
    perf_state     = Get-NvidiaSmiQuery "pstate"
    throttle       = Get-NvidiaSmiQuery "clocks_throttle_reasons.active"
  }
}

function Get-RepoCommit {
  try {
    $a = @()
    if ($Distro) { $a += @("-d", $Distro) }
    $a += @("--cd", $RepoWsl, "--", "git", "rev-parse", "HEAD")
    $sha = & wsl.exe @a 2>$null
    if ($LASTEXITCODE -eq 0) { return $sha.Trim() } else { return $null }
  } catch { return $null }
}

function Get-GpuMode {
  try {
    $a = @()
    if ($Distro) { $a += @("-d", $Distro) }
    $a += @("--", "bash", "-lc", "echo `$BARE_METAL_GPU_MODE")
    $m = & wsl.exe @a 2>$null
    if ($LASTEXITCODE -eq 0) { return $m.Trim() } else { return $null }
  } catch { return $null }
}

# ---------------------------------------------------------------------------
# Log destination
# ---------------------------------------------------------------------------
$LogDir = Join-Path $RepoRoot "scripts\probe\eval_logs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

$Stamp     = Get-Date -Format "yyyyMMdd-HHmmss"
$KernelTag = ($Kernel -replace "[\\/]", "_" -replace "\.cu$", "")
$LogPath   = Join-Path $LogDir "$Stamp`_$KernelTag`_$ClockMHz.log"
$JsonlPath = Join-Path $LogDir "results.jsonl"

"Logging to: $LogPath" | Write-Host

# ---------------------------------------------------------------------------
# Header / metadata block
# ---------------------------------------------------------------------------
$Header = @()
$Header += "=============================================================="
$Header += "run_locked_eval — $Stamp"
$Header += "=============================================================="
$Header += "host_time_local : $(Get-Date -Format o)"
$Header += "host_time_utc   : $((Get-Date).ToUniversalTime().ToString('o'))"
$Header += "repo_root_win   : $RepoRoot"
$Header += "repo_root_wsl   : $RepoWsl"
$Header += "git_head        : $(Get-RepoCommit)"
$Header += "kernel          : $Kernel"
$Header += "clock_target_mhz: $ClockMHz"
$Header += "no_lock         : $NoLock"
$Header += "gpu_mode (wsl)  : $(Get-GpuMode)"
$Header += "distro          : $Distro"
$Header | Set-Content -Path $LogPath -Encoding UTF8

$PreSnap = Get-GpuSnapshot
"--- gpu pre-lock ---"           | Add-Content $LogPath
($PreSnap | Format-List | Out-String).TrimEnd() | Add-Content $LogPath

# ---------------------------------------------------------------------------
# Lock -> measure -> restore
# ---------------------------------------------------------------------------
$LockApplied = $false
$BenchExit   = $null
$DuringSnap  = $null
$Sw          = [System.Diagnostics.Stopwatch]::StartNew()

try {
  if (-not $NoLock) {
    "" | Add-Content $LogPath
    "--- applying clock lock $ClockMHz,$ClockMHz ---" | Add-Content $LogPath
    $lockOut = & nvidia-smi.exe -lgc "$ClockMHz,$ClockMHz" 2>&1
    $lockExit = $LASTEXITCODE
    $lockOut  | Add-Content $LogPath
    "lock_exit_code: $lockExit" | Add-Content $LogPath
    if ($lockExit -ne 0) {
      throw "nvidia-smi.exe -lgc failed (exit $lockExit): $lockOut"
    }
    $LockApplied = $true
    Start-Sleep -Milliseconds 500
  }

  $DuringSnap = Get-GpuSnapshot
  "" | Add-Content $LogPath
  "--- gpu post-lock ---" | Add-Content $LogPath
  ($DuringSnap | Format-List | Out-String).TrimEnd() | Add-Content $LogPath

  $RscriptArgs = @("scripts/bench/bench_regress.R",
                   "--kernel", $Kernel,
                   "--clock-locked", "$ClockMHz")
  $wslCmd = @("wsl.exe")
  if ($Distro) { $wslCmd += @("-d", $Distro) }
  $wslCmd += @("--cd", $RepoWsl, "--", "Rscript") + $RscriptArgs

  "" | Add-Content $LogPath
  "--- invoking bench ---" | Add-Content $LogPath
  "cmd: $($wslCmd -join ' ')" | Add-Content $LogPath
  "" | Add-Content $LogPath

  # Stream stdout+stderr line-by-line into the log AND the console.
  & $wslCmd[0] $wslCmd[1..($wslCmd.Length - 1)] 2>&1 | ForEach-Object {
    $_ | Tee-Object -FilePath $LogPath -Append | Write-Host
  }
  $BenchExit = $LASTEXITCODE
  "" | Add-Content $LogPath
  "bench_exit_code: $BenchExit" | Add-Content $LogPath
}
finally {
  $Sw.Stop()
  if ($LockApplied) {
    "" | Add-Content $LogPath
    "--- restoring clock (-rgc) ---" | Add-Content $LogPath
    $rgcOut  = & nvidia-smi.exe -rgc 2>&1
    $rgcExit = $LASTEXITCODE
    $rgcOut  | Add-Content $LogPath
    "rgc_exit_code: $rgcExit" | Add-Content $LogPath
    if ($rgcExit -ne 0) {
      Write-Warning "nvidia-smi.exe -rgc returned $rgcExit. Run manually to ensure default policy."
    }
  }
}

$PostSnap = Get-GpuSnapshot
"" | Add-Content $LogPath
"--- gpu post-restore ---" | Add-Content $LogPath
($PostSnap | Format-List | Out-String).TrimEnd() | Add-Content $LogPath
"" | Add-Content $LogPath
"elapsed_seconds: $([math]::Round($Sw.Elapsed.TotalSeconds, 3))" | Add-Content $LogPath

# ---------------------------------------------------------------------------
# Structured JSONL record
# ---------------------------------------------------------------------------
$record = [ordered]@{
  timestamp_utc    = (Get-Date).ToUniversalTime().ToString("o")
  git_head         = Get-RepoCommit
  kernel           = $Kernel
  clock_target_mhz = $ClockMHz
  no_lock          = [bool]$NoLock
  bench_exit_code  = $BenchExit
  elapsed_seconds  = [math]::Round($Sw.Elapsed.TotalSeconds, 3)
  log_path         = $LogPath
  gpu_pre          = $PreSnap
  gpu_during       = $DuringSnap
  gpu_post         = $PostSnap
  gpu_mode         = Get-GpuMode
  distro           = $Distro
}
($record | ConvertTo-Json -Compress -Depth 6) | Add-Content -Path $JsonlPath -Encoding UTF8

Write-Host ""
Write-Host "=============================================================="
Write-Host "Done. Exit $BenchExit. Log: $LogPath"
Write-Host "Index: $JsonlPath"
Write-Host "=============================================================="

exit $BenchExit
