<#
.SYNOPSIS
  Multi-kernel x clock grid sweep orchestrator (#135).

.DESCRIPTION
  Reads scripts/probe/grid_sweep.yml via the R planner
  (scripts/probe/grid_measure.R --mode plan), iterates the planned
  cells grouped by clock target — one host-side nvidia-smi.exe -lgc
  per clock — and calls the R measurer for each cell. Per-sample
  rows are appended to scripts/probe/eval_logs/grid_sweep_samples.jsonl
  by the measurer; the orchestrator never writes data.

  Safety:
    - elevation assertion (skipped under -NoLock)
    - lock-state sentinel at eval_logs/.LOCK_HELD
    - C# CancelKeyPress handler (PS script-block handlers crash with
      "no Runspace" on the .NET cancel thread; see #135 Phase 1)
    - child Rscript process killed on Ctrl+C via tracked PID
    - PS `finally` covers normal completion; C# handler covers cancel
    - both paths invoke the same idempotent cleanup

  Build history:
    Phase 1 (commit 7bd55b4)  Ctrl+C harness, dummy sleep inner
    Phase 2 (this file)       YAML spec + R planner/measurer +
                              orchestrator iterates real cells
    Phase 3 (later)           --DryRun + --Resume polish + tests
    Phase 4 (later)           grid_collect.R (JSONL -> RDS)
    Phase 5 (later)           methodology doc

.PARAMETER Spec
  Path to the grid spec YAML. Default scripts/probe/grid_sweep.yml.

.PARAMETER Jsonl
  Path for the per-sample append-only store. Default
  scripts/probe/eval_logs/grid_sweep_samples.jsonl.

.PARAMETER Resume
  Pass the existing JSONL to the planner; cells whose
  (git_head, clock_target, cell_id) already appear are skipped.

.PARAMETER DryRun
  Run the planner and dump the plan, but skip all locks and
  measurements. Verifies spec validity + the cancel-handler /
  sentinel paths without spending GPU time.

.PARAMETER NoLock
  Skip nvidia-smi.exe -lgc calls. Cells whose clock_target is an
  int run anyway — useful for testing the measurement path under
  whatever clock the GPU picks. Band-checks will reject most
  samples; this is by design.

.PARAMETER ForceClearSentinel
  Delete a stale sentinel before starting. Use only after manually
  running nvidia-smi.exe -rgc.

.EXAMPLE
  pwsh -File scripts\probe\run_grid_sweep.ps1 -DryRun

.EXAMPLE
  pwsh -File scripts\probe\run_grid_sweep.ps1 -Resume
#>

[CmdletBinding()]
param(
  [string] $RepoRoot             = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
  [string] $Spec                 = "scripts\probe\grid_sweep.yml",
  [string] $Jsonl                = "scripts\probe\eval_logs\grid_sweep_samples.jsonl",
  [switch] $Resume,
  [switch] $DryRun,
  [switch] $NoLock,
  [switch] $ForceClearSentinel,
  [string] $Distro               = "",
  [string] $OnlyCellId           = ""
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
function Assert-Elevated {
  $id = [Security.Principal.WindowsIdentity]::GetCurrent()
  $pr = New-Object Security.Principal.WindowsPrincipal($id)
  if (-not $pr.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Elevated PowerShell required. nvidia-smi.exe -lgc fails otherwise. Use -NoLock or -DryRun for non-elevated testing."
  }
}
function Assert-Tool($name) {
  if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
    throw "Tool not on PATH: $name"
  }
}

if (-not ($NoLock -or $DryRun)) { Assert-Elevated }
Assert-Tool "nvidia-smi.exe"
Assert-Tool "wsl.exe"

# Resolve repo paths
if (-not (Test-Path $RepoRoot)) { throw "Repo root not found: $RepoRoot" }
$SpecAbs  = if ([IO.Path]::IsPathRooted($Spec))  { $Spec }  else { Join-Path $RepoRoot $Spec }
$JsonlAbs = if ([IO.Path]::IsPathRooted($Jsonl)) { $Jsonl } else { Join-Path $RepoRoot $Jsonl }
if (-not (Test-Path $SpecAbs)) { throw "Spec not found: $SpecAbs" }

$EvalDir     = Split-Path $JsonlAbs -Parent
$SentinelPath = Join-Path $EvalDir ".LOCK_HELD"
if (-not (Test-Path $EvalDir)) { New-Item -ItemType Directory -Path $EvalDir | Out-Null }

# Win -> WSL path translation. Local transform avoids wsl.exe arg-passing
# backslash mangling (#131 run_locked_eval.ps1 hit this on first attempt).
function ConvertTo-WslPath($winPath) {
  $full = (Resolve-Path $winPath).Path
  if ($full -match '^([A-Za-z]):[\\/](.*)$') {
    $drive = $Matches[1].ToLower()
    $rest  = $Matches[2] -replace '\\', '/'
    return "/mnt/$drive/$rest"
  }
  throw "Cannot translate path to WSL form: $winPath"
}

$RepoWsl  = ConvertTo-WslPath $RepoRoot
$SpecWsl  = ConvertTo-WslPath $SpecAbs
# JSONL may not exist yet on first run — translate parent + append name.
$JsonlDirAbs = Split-Path $JsonlAbs -Parent
$JsonlWsl    = (ConvertTo-WslPath $JsonlDirAbs) + "/" + (Split-Path $JsonlAbs -Leaf)

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
# C# cleanup class (Add-Type) — runs without a PS Runspace.
# PS script-block handlers crashed in Phase 1 testing with
# "no Runspace available". C# delegate works on the cancel thread.
# ---------------------------------------------------------------------------
Add-Type -TypeDefinition @"
using System;
using System.Diagnostics;
using System.IO;
public static class GridSweepCleanup {
  public static string SentinelPath = "";
  public static bool LockApplied = false;
  public static int ChildPid = 0;
  public static bool Done = false;

  public static void KillChild() {
    int pid = ChildPid;
    if (pid > 0) {
      try {
        var p = Process.GetProcessById(pid);
        // Kill(true) terminates the Windows process tree (.NET 6+).
        p.Kill(true);
        p.WaitForExit(3000);
        Console.WriteLine("[cleanup] child PID " + pid + " killed (Windows tree)");
      } catch (Exception ex) {
        Console.WriteLine("[cleanup] child PID " + pid + " kill skipped: " + ex.Message);
      }
      ChildPid = 0;
    }

    // WSL2 Linux processes are children of init inside the WSL VM,
    // NOT Windows children of wsl.exe. Process.Kill(tree=true) does
    // not reach them. Issue an explicit pkill inside WSL targeting
    // our script + the kernel bench binaries. Best-effort; runs even
    // if Windows-side kill already cleaned up cleanly.
    try {
      var psi = new ProcessStartInfo("wsl.exe");
      psi.ArgumentList.Add("-e");
      psi.ArgumentList.Add("pkill");
      psi.ArgumentList.Add("-9");
      psi.ArgumentList.Add("-f");
      psi.ArgumentList.Add("grid_measure.R");
      psi.UseShellExecute = false;
      psi.CreateNoWindow = true;
      psi.RedirectStandardOutput = true;
      psi.RedirectStandardError = true;
      var p = Process.Start(psi);
      p.WaitForExit(3000);
      Console.WriteLine("[cleanup] WSL pkill grid_measure.R: exit " + p.ExitCode);
    } catch (Exception ex) {
      Console.WriteLine("[cleanup] WSL pkill failed: " + ex.Message);
    }
  }

  public static void Run() {
    if (Done) return;
    Done = true;
    KillChild();
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

$Script:LockApplied = $false

function Invoke-Cleanup {
  [GridSweepCleanup]::LockApplied = $Script:LockApplied
  [GridSweepCleanup]::Run()
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Build the wsl.exe arg array for a Rscript invocation.
function New-WslRscriptArgs([string[]]$RscriptArgs) {
  $a = @()
  if ($Distro) { $a += @("-d", $Distro) }
  $a += @("--cd", $RepoWsl, "--", "Rscript") + $RscriptArgs
  return ,$a
}

# Run a child wsl.exe with PID tracking + console streaming.
# Throws on non-zero exit code.
function Invoke-WslChild([string[]]$WslArgs, [string]$Label) {
  Write-Host "[$Label] wsl $($WslArgs -join ' ')" -ForegroundColor DarkGray
  $psi = New-Object System.Diagnostics.ProcessStartInfo
  $psi.FileName = "wsl.exe"
  foreach ($a in $WslArgs) { $psi.ArgumentList.Add($a) | Out-Null }
  $psi.UseShellExecute = $false
  # Inherit console streams — no redirection. Output appears live.
  $proc = [System.Diagnostics.Process]::Start($psi)
  [GridSweepCleanup]::ChildPid = $proc.Id
  try {
    $proc.WaitForExit()
  } finally {
    [GridSweepCleanup]::ChildPid = 0
  }
  if ($proc.ExitCode -ne 0) {
    throw "[$Label] child exited $($proc.ExitCode)"
  }
}

# Capture child stdout (used for planner JSON parse).
function Invoke-WslChildCapture([string[]]$WslArgs, [string]$Label) {
  Write-Host "[$Label] wsl $($WslArgs -join ' ')" -ForegroundColor DarkGray
  $psi = New-Object System.Diagnostics.ProcessStartInfo
  $psi.FileName = "wsl.exe"
  foreach ($a in $WslArgs) { $psi.ArgumentList.Add($a) | Out-Null }
  $psi.UseShellExecute = $false
  $psi.RedirectStandardOutput = $true
  $psi.RedirectStandardError  = $true
  $proc = [System.Diagnostics.Process]::Start($psi)
  [GridSweepCleanup]::ChildPid = $proc.Id
  try {
    $stdout = $proc.StandardOutput.ReadToEnd()
    $stderr = $proc.StandardError.ReadToEnd()
    $proc.WaitForExit()
  } finally {
    [GridSweepCleanup]::ChildPid = 0
  }
  if ($proc.ExitCode -ne 0) {
    Write-Host "--- stderr ---" -ForegroundColor Yellow
    Write-Host $stderr
    throw "[$Label] child exited $($proc.ExitCode)"
  }
  if ($stderr) { Write-Host $stderr -ForegroundColor DarkYellow }
  return $stdout
}

# Apply / restore SM clock lock.
function Apply-Lock([int]$ClockMHz) {
  Write-Host "[lock] nvidia-smi.exe -lgc $ClockMHz,$ClockMHz" -ForegroundColor Cyan
  & nvidia-smi.exe -lgc "$ClockMHz,$ClockMHz" | Out-Null
  $Script:LockApplied = $true
  [GridSweepCleanup]::LockApplied = $true
  @"
pid: $PID
host_time_utc: $((Get-Date).ToUniversalTime().ToString('o'))
clock_mhz: $ClockMHz
script: $($MyInvocation.MyCommand.Path)
"@ | Set-Content -Path $SentinelPath -Encoding UTF8
}

function Release-Lock() {
  if (-not $Script:LockApplied) { return }
  Write-Host "[lock] nvidia-smi.exe -rgc (between groups)" -ForegroundColor Cyan
  & nvidia-smi.exe -rgc | Out-Null
  $Script:LockApplied = $false
  [GridSweepCleanup]::LockApplied = $false
  if (Test-Path $SentinelPath) { Remove-Item -Force $SentinelPath }
}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$RunId = "grid-$Stamp"
Write-Host "=============================================================="
Write-Host "run_grid_sweep — $Stamp"
Write-Host "=============================================================="
Write-Host "RepoRoot     : $RepoRoot"
Write-Host "Spec         : $SpecAbs"
Write-Host "Jsonl        : $JsonlAbs"
Write-Host "RunId        : $RunId"
Write-Host "Resume       : $Resume"
Write-Host "DryRun       : $DryRun"
Write-Host "NoLock       : $NoLock"
Write-Host ""

try {
  # ---------------------------------------------------------------------
  # Plan
  # ---------------------------------------------------------------------
  # Plan JSON via file handoff. R's stdout carries renv NOTEs and
  # other startup noise that pollute pipe-based capture; --out path
  # is the clean contract.
  $planFile = Join-Path $EvalDir "grid_plan_$Stamp.json"
  $planFileWsl = (ConvertTo-WslPath $EvalDir) + "/" + (Split-Path $planFile -Leaf)
  $planArgs = @("scripts/probe/grid_measure.R", "--mode", "plan",
                "--spec", $SpecWsl, "--out", $planFileWsl)
  if ($Resume) {
    $planArgs += @("--resume-jsonl", $JsonlWsl)
  }
  Invoke-WslChild (New-WslRscriptArgs $planArgs) "plan"
  if (-not (Test-Path $planFile)) {
    throw "Planner did not produce $planFile"
  }
  $plan = Get-Content $planFile -Raw | ConvertFrom-Json

  Write-Host ""
  Write-Host "[plan] git_head    : $($plan.git_head)"
  Write-Host "[plan] n_cells     : $($plan.n_cells)"
  Write-Host "[plan] n_pending   : $($plan.n_pending)"
  Write-Host ""

  if ($plan.n_pending -eq 0) {
    Write-Host "Nothing to measure (all cells already in JSONL at this git_head)."
    return
  }

  $pendingCells = @($plan.cells | Where-Object { -not $_.already_done })
  if ($OnlyCellId) {
    $pendingCells = @($pendingCells | Where-Object { $_.cell_id -eq $OnlyCellId })
    Write-Host "[plan] filtered to OnlyCellId='$OnlyCellId': $($pendingCells.Count) cells" -ForegroundColor Yellow
    if ($pendingCells.Count -eq 0) {
      Write-Warning "OnlyCellId '$OnlyCellId' matches no pending cells. Exiting."
      return
    }
  }

  # Group by clock_target_mhz. Null = native (no lock).
  $groups = $pendingCells | Group-Object { if ($null -eq $_.clock_target_mhz) { "native" } else { [int]$_.clock_target_mhz } }
  # Sort: native first, then ascending int clock.
  $groups = $groups | Sort-Object { if ($_.Name -eq "native") { -1 } else { [int]$_.Name } }

  Write-Host "[plan] groups:"
  foreach ($g in $groups) {
    Write-Host ("  {0,8} : {1} cells" -f $g.Name, $g.Count)
  }
  Write-Host ""

  if ($DryRun) {
    Write-Host "[dry-run] skipping locks + measurements. Plan validated."
    return
  }

  # ---------------------------------------------------------------------
  # Iterate
  # ---------------------------------------------------------------------
  $cellIdx = 0
  foreach ($g in $groups) {
    $clockLabel = $g.Name
    $needsLock  = ($clockLabel -ne "native") -and (-not $NoLock)
    Write-Host ""
    Write-Host "=== group clock=$clockLabel  ($($g.Count) cells) ===" -ForegroundColor Magenta

    if ($needsLock) {
      Apply-Lock([int]$clockLabel)
    }

    foreach ($cell in $g.Group) {
      $cellIdx++
      Write-Host ""
      Write-Host "[$cellIdx/$($plan.n_pending)] $($cell.cell_id) @ $clockLabel" -ForegroundColor Green

      $measureArgs = @(
        "scripts/probe/grid_measure.R", "--mode", "measure",
        "--spec", $SpecWsl,
        "--cell-id", $cell.cell_id,
        "--clock-target", $clockLabel,
        "--jsonl", $JsonlWsl,
        "--run-id", $RunId
      )
      try {
        Invoke-WslChild (New-WslRscriptArgs $measureArgs) "measure"
      } catch {
        # One cell failing should not abort the sweep — the JSONL still
        # has partial data and the operator can investigate per-cell.
        Write-Warning "Cell failed: $_"
      }
    }

    if ($needsLock) {
      Release-Lock
    }
  }
}
finally {
  Invoke-Cleanup
}

Write-Host ""
Write-Host "=============================================================="
Write-Host "Sweep complete."
Write-Host "  JSONL : $JsonlAbs"
Write-Host "  Verify:"
Write-Host "    Test-Path '$SentinelPath'   # expect False"
Write-Host "=============================================================="

<#
TEST_PLAN — runtime validation

Phase 1 (lock-safety harness) tests T1, T2, T2-clean, T4, T4b are
preserved in commit 7bd55b4. Phase 2+ additions:

P2-1. Dry run (no elevation, no GPU touch):
        pwsh -File scripts\probe\run_grid_sweep.ps1 -DryRun
      Expect: plan loaded, groups listed, "[dry-run] skipping" message,
              no sentinel, no .jsonl modifications.

P2-2. Single-cell measure under -NoLock:
        Remove-Item scripts\probe\eval_logs\grid_sweep_samples.jsonl -ErrorAction SilentlyContinue
        # Edit grid_sweep.yml to a 1-kernel, native-only subset, or
        # let it run all native-regime cells (~5 min).
        pwsh -File scripts\probe\run_grid_sweep.ps1 -NoLock
      Expect: native-group cells measured; locked-group cells run but
              their band-checks reject samples; JSONL grows; sentinel
              never appears.

P2-3. Resume:
        # After a partial run.
        pwsh -File scripts\probe\run_grid_sweep.ps1 -Resume -DryRun
      Expect: n_pending < n_cells; only un-measured cells in groups.

P2-4. Full elevated sweep (the real artifact). Allow ~hour.

P2-5. Ctrl+C mid-measure: same expectations as Phase 1 T2 plus the
      child Rscript process tree must be killed (not orphaned).
      Verify: nvidia-smi.exe --query-gpu=clocks.current.sm  # not stuck
      Verify: Get-Process Rscript -ErrorAction SilentlyContinue  # empty
#>
