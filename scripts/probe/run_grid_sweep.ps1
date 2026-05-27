<#
.SYNOPSIS
  Multi-kernel x clock grid sweep orchestrator (#135). PHASE 1
  SKELETON — Ctrl+C / lock-safety harness only. Inner measurement
  not yet wired.

.DESCRIPTION
  Build order (per #135 design):
    Phase 1 (this file, now)
      - Elevation assertion
      - Lock-state sentinel
      - [Console]::CancelKeyPress handler
      - Dummy inner (sleep) so Ctrl+C scenarios can be exercised
        before any real measurement code lands.
    Phase 2 — wire grid_measure.R planner + measure-one-cell.
    Phase 3 — resume, dry-run, no-lock.
    Phase 4 — grid_collect.R (JSONL -> RDS materialiser).
    Phase 5 — methodology doc.

  The Ctrl+C path is the load-bearing piece: PowerShell `finally` is
  not guaranteed to run on console cancel. We use both `finally`
  (covers exceptions) AND `[Console]::CancelKeyPress` (covers
  ConsoleCancelEventArgs.SpecialKey = ControlC). A sentinel file
  recorded the moment the lock is applied; the handler reads it on
  cancel to know whether `-rgc` is needed. If the script crashes
  hard, a future run sees the sentinel and refuses to re-lock until
  the operator resolves it.

  This skeleton must pass three manual Ctrl+C tests in the operator's
  elevated pwsh before Phase 2 is started — see TEST_PLAN at the
  bottom.

.PARAMETER ClockMHz
  Sentinel-test only: clock to lock during the dummy phase.
  Default 1605.

.PARAMETER SleepSeconds
  How long the dummy inner sleeps. Default 30. Long enough to
  Ctrl+C mid-sleep without sprinting.

.PARAMETER NoLock
  Skip the actual nvidia-smi.exe -lgc call. Lets the elevation /
  sentinel / handler paths be exercised without touching the GPU
  state. Useful for first-pass validation.

.PARAMETER ForceClearSentinel
  Delete a stale sentinel file before starting. Use only after
  manually verifying nvidia-smi.exe -rgc has been run.

.EXAMPLE
  # Real lock, dummy sleep, Ctrl+C test.
  pwsh -File scripts\probe\run_grid_sweep.ps1

.EXAMPLE
  # Sentinel/handler test without touching the GPU.
  pwsh -File scripts\probe\run_grid_sweep.ps1 -NoLock
#>

[CmdletBinding()]
param(
  [int]    $ClockMHz            = 1605,
  [int]    $SleepSeconds        = 30,
  [string] $RepoRoot            = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
  [switch] $NoLock,
  [switch] $ForceClearSentinel
)

$ErrorActionPreference = "Stop"
# PS 7+: native exes (nvidia-smi.exe, wsl.exe) throw on non-zero exit.
$PSNativeCommandUseErrorActionPreference = $true

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
function Assert-Elevated {
  $id = [Security.Principal.WindowsIdentity]::GetCurrent()
  $pr = New-Object Security.Principal.WindowsPrincipal($id)
  if (-not $pr.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Elevated PowerShell required. nvidia-smi.exe -lgc fails otherwise."
  }
}
function Assert-Tool($name) {
  if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
    throw "Tool not on PATH: $name"
  }
}

if (-not $NoLock) { Assert-Elevated }
Assert-Tool "nvidia-smi.exe"

$EvalDir     = Join-Path $RepoRoot "scripts\probe\eval_logs"
$SentinelPath = Join-Path $EvalDir ".LOCK_HELD"
if (-not (Test-Path $EvalDir)) { New-Item -ItemType Directory -Path $EvalDir | Out-Null }

# ---------------------------------------------------------------------------
# Stale sentinel handling
# ---------------------------------------------------------------------------
if (Test-Path $SentinelPath) {
  if ($ForceClearSentinel) {
    Write-Warning "Removing stale sentinel: $SentinelPath"
    Write-Warning "Verify GPU is unlocked: nvidia-smi.exe -rgc"
    Remove-Item -Force $SentinelPath
  } else {
    $content = Get-Content $SentinelPath -Raw
    Write-Error @"
Stale lock sentinel present: $SentinelPath
Contents:
$content

A previous run may have left the GPU clock-locked. Recover manually:
    nvidia-smi.exe -rgc
Then either:
    - delete $SentinelPath
    - or re-run with -ForceClearSentinel

Refusing to apply a new lock on top of unknown state.
"@
    exit 2
  }
}

# ---------------------------------------------------------------------------
# Cancel handler — fires on Ctrl+C, runs BEFORE the process is killed.
# Note: PowerShell does NOT guarantee `finally` runs on console cancel.
# This handler covers the gap. The handler is best-effort: if the
# process is hard-killed (taskkill /F), neither runs and the sentinel
# stays — next invocation sees it and refuses to relock.
# ---------------------------------------------------------------------------
$Script:LockApplied = $false

# Console cancel handler.
#
# Critical: PowerShell script-blocks attached to [Console]::CancelKeyPress
# fail with "no Runspace available" — the .NET cancel handler fires on a
# worker thread that has no PowerShell Runspace. T2 (#135 Phase 1)
# proved this empirically: PS crashed mid-sleep with an unhandled
# PSInvalidOperationException and the GPU stayed locked.
#
# Fix: implement the handler in C# via Add-Type. Pure .NET code runs
# on the worker thread without needing a Runspace. The PS-side
# `finally` block remains the normal-completion path; the C# handler
# is the Ctrl+C path. Both call the same cleanup work and are
# idempotent via a shared flag.
Add-Type -TypeDefinition @"
using System;
using System.Diagnostics;
using System.IO;
public static class GridSweepCleanup {
  public static string SentinelPath = "";
  public static bool LockApplied = false;
  public static bool Done = false;

  public static void Run() {
    if (Done) return;
    Done = true;
    if (LockApplied) {
      Console.WriteLine();
      Console.WriteLine("[cleanup] restoring default clock policy (-rgc)...");
      try {
        var psi = new ProcessStartInfo("nvidia-smi.exe", "-rgc");
        psi.UseShellExecute = false;
        psi.CreateNoWindow = true;
        psi.RedirectStandardOutput = true;
        psi.RedirectStandardError = true;
        var p = Process.Start(psi);
        p.WaitForExit();
        if (p.ExitCode == 0) Console.WriteLine("[cleanup] -rgc OK");
        else Console.WriteLine("[cleanup] -rgc returned exit " + p.ExitCode);
      } catch (Exception ex) {
        Console.WriteLine("[cleanup] -rgc threw: " + ex.Message);
        Console.WriteLine("[cleanup] run manually: nvidia-smi.exe -rgc");
      }
    }
    if (!string.IsNullOrEmpty(SentinelPath) && File.Exists(SentinelPath)) {
      try {
        File.Delete(SentinelPath);
        Console.WriteLine("[cleanup] sentinel cleared");
      } catch (Exception ex) {
        Console.WriteLine("[cleanup] sentinel removal failed: " + ex.Message);
      }
    }
  }

  public static void OnCancel(object sender, ConsoleCancelEventArgs e) {
    e.Cancel = true;
    Console.WriteLine();
    Console.WriteLine("[CancelKeyPress] Ctrl+C received -- running cleanup");
    try { Run(); } catch (Exception ex) {
      Console.WriteLine("[cleanup] handler threw: " + ex.Message);
    }
    Console.WriteLine("[CancelKeyPress] exit 130 (SIGINT)");
    Environment.Exit(130);
  }
}
"@ -ErrorAction Stop

[GridSweepCleanup]::SentinelPath = $SentinelPath
[Console]::add_CancelKeyPress([ConsoleCancelEventHandler]([GridSweepCleanup]::OnCancel))

function Invoke-Cleanup {
  # Normal-path cleanup (finally block). Delegates to the same C#
  # routine the cancel handler uses, so the two paths cannot diverge.
  [GridSweepCleanup]::LockApplied = $Script:LockApplied
  [GridSweepCleanup]::Run()
}

# ---------------------------------------------------------------------------
# Dummy phase: lock -> sleep -> restore
# ---------------------------------------------------------------------------
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
Write-Host "=============================================================="
Write-Host "run_grid_sweep (PHASE 1 SKELETON) — $Stamp"
Write-Host "=============================================================="
Write-Host "RepoRoot     : $RepoRoot"
Write-Host "EvalDir      : $EvalDir"
Write-Host "SentinelPath : $SentinelPath"
Write-Host "ClockMHz     : $ClockMHz"
Write-Host "SleepSeconds : $SleepSeconds"
Write-Host "NoLock       : $NoLock"
Write-Host ""

try {
  if (-not $NoLock) {
    Write-Host "[lock] nvidia-smi.exe -lgc $ClockMHz,$ClockMHz"
    & nvidia-smi.exe -lgc "$ClockMHz,$ClockMHz" | Out-Null
    $Script:LockApplied = $true
    # Mirror into C# static so the Ctrl+C handler (runs on a
    # non-Runspace thread) knows to call -rgc. PS Invoke-Cleanup also
    # mirrors this; we set it eagerly so an immediate Ctrl+C between
    # -lgc and sentinel-write still triggers restore.
    [GridSweepCleanup]::LockApplied = $true
    # Sentinel written AFTER -lgc returns OK.
    @"
pid: $PID
host_time_utc: $((Get-Date).ToUniversalTime().ToString('o'))
clock_mhz: $ClockMHz
script: $($MyInvocation.MyCommand.Path)
"@ | Set-Content -Path $SentinelPath -Encoding UTF8
    Write-Host "[lock] sentinel written"
  } else {
    Write-Host "[lock] skipped (-NoLock)"
  }

  Write-Host ""
  Write-Host "[dummy] sleeping $SleepSeconds s — Ctrl+C now to test handler"
  for ($i = 1; $i -le $SleepSeconds; $i++) {
    Start-Sleep -Seconds 1
    if ($i % 5 -eq 0) { Write-Host "[dummy] $i / $SleepSeconds" }
  }
  Write-Host "[dummy] sleep complete (no cancel)"
}
finally {
  Invoke-Cleanup
}

Write-Host ""
Write-Host "=============================================================="
Write-Host "Skeleton run complete. Sentinel + lock state should be clean."
Write-Host "Verify:"
Write-Host "  Test-Path '$SentinelPath'           # expect False"
Write-Host "  nvidia-smi.exe --query-gpu=clocks.current.sm --format=csv,noheader"
Write-Host "=============================================================="

<#
TEST_PLAN — manual validation before Phase 2

Goal: confirm that on Ctrl+C, the GPU is always restored to default
clock policy and the sentinel is cleared, regardless of when in the
script the cancel arrives.

T1. No-lock smoke (no GPU touch):
    pwsh -File scripts\probe\run_grid_sweep.ps1 -NoLock -SleepSeconds 10
    Let it run to completion. Expect "Skeleton run complete".
    Sentinel must not exist afterward.

T2. Real lock, cancel mid-sleep:
    (elevated) pwsh -File scripts\probe\run_grid_sweep.ps1 -SleepSeconds 30
    Wait until "sleeping 30 s" line; press Ctrl+C around the 10 s mark.
    Expect:
        [CancelKeyPress] Ctrl+C received — running cleanup
        [cleanup] restoring default clock policy (-rgc)...
        [cleanup] -rgc OK
        [cleanup] sentinel cleared
        [CancelKeyPress] exit 130 (SIGINT)
    Verify: nvidia-smi.exe --query-gpu=clocks.current.sm,pstate,clocks_throttle_reasons.active --format=csv,noheader
        SM clock should rise to native boost when load returns (P0).
    Verify: sentinel gone (Test-Path returns False).

T3. Real lock, cancel BEFORE sleep starts (race window):
    pwsh -File scripts\probe\run_grid_sweep.ps1 -SleepSeconds 60
    Press Ctrl+C the instant "[lock] sentinel written" appears.
    Same expected cleanup output. Same verification.

T4. Stale-sentinel refusal:
    Manually create sentinel: New-Item $SentinelPath -ItemType File -Force
    Re-run: pwsh -File scripts\probe\run_grid_sweep.ps1 -NoLock
    Expect: script refuses with exit 2 and a recovery message.
    Recover: nvidia-smi.exe -rgc; Remove-Item $SentinelPath
    OR re-run with -ForceClearSentinel.

If T1-T4 all pass, Phase 1 is validated and Phase 2 can land.
If T2 or T3 fails (clock stays locked OR sentinel remains), STOP
and redesign the handler before adding measurement code.
#>
