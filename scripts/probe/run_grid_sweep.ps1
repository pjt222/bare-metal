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

# Resolve the project's renv Linux library ONCE — for the child
# autoloader bypass in New-WslRscriptArgs (#135 P2-5). The glob keeps
# it R-version / distro agnostic; an empty result makes the bypass
# fall back to plain Rscript (correct, just slower / renv-bootstrapped).
$renvGlobArgs = @()
if ($Distro) { $renvGlobArgs += @("-d", $Distro) }
$renvGlobArgs += @("--cd", $RepoWsl, "--", "bash", "-c",
                   "ls -d renv/library/*/*/x86_64-pc-linux-gnu 2>/dev/null | head -1")
$RenvLibWsl = (& wsl.exe @renvGlobArgs | Select-Object -First 1)
if ($RenvLibWsl) {
  $RenvLibWsl = "$RenvLibWsl".Trim()
  # Glob ran under --cd $RepoWsl, so it's repo-relative; make it
  # absolute so R_LIBS_USER is unambiguous regardless of child cwd.
  if ($RenvLibWsl -notmatch '^/') { $RenvLibWsl = "$RepoWsl/$RenvLibWsl" }
  Write-Host "[renv] child autoloader bypass -> R_LIBS_USER=$RenvLibWsl" -ForegroundColor DarkGray
} else {
  Write-Host "[renv] lib glob empty -> children use default Rscript (renv-bootstrapped, slower)" -ForegroundColor Yellow
}

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
using System.Text;
using System.Runtime.InteropServices;
using System.ComponentModel;
public static class GridSweepCleanup {
  public static string SentinelPath = "";
  public static bool LockApplied = false;
  public static int ChildPid = 0;
  public static bool Done = false;
  // Set true on any cancel (CancelKeyPress, or pwsh seeing the child exit
  // 130). Distinguishes a clean completion (skip the bench pkill so it
  // cannot hit a concurrent unrelated bench) from a cancel, where we MUST
  // pkill even if R already exited 130 mid-sample and orphaned its bench.
  public static bool Cancelled = false;
  // Substring of the WSL-side cmdline for pkill'ing an orphaned kernel
  // bench on cancel. Set from PowerShell to "<repoWsl>/kernels/". With
  // Route B the child runs in a new process group and does NOT receive
  // the console Ctrl+C, so the bench keeps running until killed here --
  // without this it orphans and wedges CUDA (the known hazard).
  public static string BenchKillPattern = "";

  // --- Win32 CreateProcess in a NEW PROCESS GROUP (#135 P2-5, Route B)
  // A console Ctrl+C is a CTRL_C_EVENT broadcast to every process in the
  // console's process GROUP. If wsl.exe shares our group, it + R + the
  // bench all get the signal and die in ways that defeat a clean abort
  // (R's system2 wait returns rc=0; the sweep reads "success" and
  // continues). Launching wsl.exe with CREATE_NEW_PROCESS_GROUP disables
  // Ctrl+C delivery to it, so ONLY pwsh receives CTRL_C_EVENT -> OnCancel
  // fires and OWNS the abort (kill child tree + -rgc + sentinel),
  // independent of the child exit code. The child shares this console
  // (no CREATE_NEW_CONSOLE) so output still streams live.
  const uint CREATE_NEW_PROCESS_GROUP = 0x00000200;
  const uint INFINITE = 0xFFFFFFFF;

  [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
  struct STARTUPINFO {
    public int cb;
    public string lpReserved, lpDesktop, lpTitle;
    public int dwX, dwY, dwXSize, dwYSize, dwXCountChars, dwYCountChars, dwFillAttribute, dwFlags;
    public short wShowWindow, cbReserved2;
    public IntPtr lpReserved2, hStdInput, hStdOutput, hStdError;
  }
  [StructLayout(LayoutKind.Sequential)]
  struct PROCESS_INFORMATION { public IntPtr hProcess, hThread; public int dwProcessId, dwThreadId; }

  [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
  static extern bool CreateProcess(string app, StringBuilder cmd, IntPtr pa, IntPtr ta,
    bool inherit, uint flags, IntPtr env, string cwd, ref STARTUPINFO si, out PROCESS_INFORMATION pi);
  [DllImport("kernel32.dll", SetLastError = true)]
  static extern uint WaitForSingleObject(IntPtr h, uint ms);
  [DllImport("kernel32.dll", SetLastError = true)]
  static extern bool GetExitCodeProcess(IntPtr h, out uint code);
  [DllImport("kernel32.dll", SetLastError = true)]
  static extern bool CloseHandle(IntPtr h);

  // Quote one argument per the CommandLineToArgvW rules.
  static string QuoteArg(string s) {
    if (s.Length > 0 && s.IndexOfAny(new char[] { ' ', '\t', '"' }) < 0) return s;
    var sb = new StringBuilder();
    sb.Append('"');
    int bs = 0;
    foreach (char c in s) {
      if (c == '\\') { bs++; }
      else if (c == '"') { sb.Append('\\', bs * 2 + 1); sb.Append('"'); bs = 0; }
      else { if (bs > 0) { sb.Append('\\', bs); bs = 0; } sb.Append(c); }
    }
    sb.Append('\\', bs * 2);
    sb.Append('"');
    return sb.ToString();
  }
  public static string BuildCommandLine(string exe, string[] args) {
    var sb = new StringBuilder(QuoteArg(exe));
    foreach (var a in args) { sb.Append(' '); sb.Append(QuoteArg(a)); }
    return sb.ToString();
  }

  // Launch the command line (wsl.exe ...) in a new process group sharing
  // this console; block until exit; return the exit code. Sets ChildPid
  // for the cancel handler's KillChild.
  public static int RunNewGroup(string commandLine) {
    var si = new STARTUPINFO();
    si.cb = Marshal.SizeOf(typeof(STARTUPINFO));
    PROCESS_INFORMATION pi;
    var cmd = new StringBuilder(commandLine);
    bool ok = CreateProcess(null, cmd, IntPtr.Zero, IntPtr.Zero,
                            true, CREATE_NEW_PROCESS_GROUP, IntPtr.Zero, null, ref si, out pi);
    if (!ok) throw new Win32Exception(Marshal.GetLastWin32Error(),
                       "CreateProcess(NEW_PROCESS_GROUP) failed for: " + commandLine);
    ChildPid = pi.dwProcessId;
    try {
      WaitForSingleObject(pi.hProcess, INFINITE);
      uint code;
      if (!GetExitCodeProcess(pi.hProcess, out code)) code = 1;
      return unchecked((int)code);
    } finally {
      ChildPid = 0;
      CloseHandle(pi.hThread);
      CloseHandle(pi.hProcess);
    }
  }

  static void WslPkill(string pattern) {
    try {
      var psi = new ProcessStartInfo("wsl.exe");
      psi.ArgumentList.Add("-e");
      psi.ArgumentList.Add("pkill");
      psi.ArgumentList.Add("-9");
      psi.ArgumentList.Add("-f");
      psi.ArgumentList.Add(pattern);
      psi.UseShellExecute = false;
      psi.CreateNoWindow = true;
      psi.RedirectStandardOutput = true;
      psi.RedirectStandardError = true;
      var p = Process.Start(psi);
      p.WaitForExit(3000);
      Console.WriteLine("[cleanup] WSL pkill -f '" + pattern + "': exit " + p.ExitCode);
    } catch (Exception ex) {
      Console.WriteLine("[cleanup] WSL pkill -f '" + pattern + "' failed: " + ex.Message);
    }
  }

  public static void KillChild() {
    int pid = ChildPid;
    // Clean completion (no active child AND not a cancel): skip entirely
    // so the broad bench pattern cannot hit a concurrent unrelated bench.
    if (pid <= 0 && !Cancelled) return;
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

    // On a cancel, pkill the WSL-side script AND the kernel bench binary.
    // WSL2 Linux processes are children of init inside the VM, NOT Windows
    // children of wsl.exe -- Process.Kill(tree) does not reach them. This
    // runs even when R already exited 130 mid-sample (pid==0 here) and
    // left its bench orphaned, which would otherwise spin and wedge CUDA.
    WslPkill("grid_measure.R");
    if (!string.IsNullOrEmpty(BenchKillPattern)) WslPkill(BenchKillPattern);
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
    Cancelled = true;
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
# Orphan-bench kill pattern for the cancel path (Route B): any process
# whose WSL cmdline contains "<repo>/kernels/" — i.e. a running kernel
# bench launched by the measurer. Killed on Ctrl+C so it cannot spin and
# wedge CUDA after the measurer is gone.
[GridSweepCleanup]::BenchKillPattern = "$RepoWsl/kernels/"
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
  # #135 P2-5: bypass the renv autoloader in the child. `.Rprofile ->
  # renv/activate.R` costs ~26s/child ("dependency discovery") AND a
  # Ctrl+C during that window halts R with exit 1, NOT 130 — which the
  # sweep loop reads as a cell failure and CONTINUES instead of
  # aborting. `--no-init-file` skips .Rprofile (keeps .Renviron);
  # R_LIBS_USER points packages at the renv library directly, so the
  # child reaches grid_measure.R's SIGINT trap in ~2-3s and every
  # Ctrl+C yields 130. Empty glob -> plain Rscript (correct, slower).
  if ($RenvLibWsl) {
    $a += @("--cd", $RepoWsl, "--",
            "env", "R_LIBS_USER=$RenvLibWsl",
            "Rscript", "--no-init-file") + $RscriptArgs
  } else {
    $a += @("--cd", $RepoWsl, "--", "Rscript") + $RscriptArgs
  }
  return ,$a
}

# Run a child wsl.exe with PID tracking + console streaming.
# Returns the child exit code. Caller decides how to respond — the
# distinction between 130 (R-side SIGINT) and other non-zero codes
# matters for cancel-vs-failure handling in the sweep loop.
function Invoke-WslChild([string[]]$WslArgs, [string]$Label) {
  Write-Host "[$Label] wsl $($WslArgs -join ' ')" -ForegroundColor DarkGray
  # Route B (#135 P2-5): launch wsl.exe in a NEW PROCESS GROUP so a
  # console Ctrl+C is NOT delivered to the child (it would otherwise make
  # R's system2 wait return rc=0 and the sweep continue). Only pwsh gets
  # CTRL_C_EVENT -> OnCancel owns the abort. RunNewGroup shares this
  # console (live output), tracks ChildPid, and returns the exit code.
  $cmd = [GridSweepCleanup]::BuildCommandLine("wsl.exe", $WslArgs)
  return [GridSweepCleanup]::RunNewGroup($cmd)
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
  $planExit = Invoke-WslChild (New-WslRscriptArgs $planArgs) "plan"
  if ($planExit -ne 0) {
    throw "Planner exited $planExit"
  }
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
  $cancelled = $false
  :groupLoop foreach ($g in $groups) {
    if ($cancelled) { break }
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
      $cellExit = Invoke-WslChild (New-WslRscriptArgs $measureArgs) "measure"

      if ($cellExit -eq 130) {
        # R-side SIGINT trap (Route A). User pressed Ctrl+C; R caught it
        # and exited 130. Abort the sweep cleanly — do NOT continue. Flag
        # the cancel so the finally's KillChild still pkills a bench R may
        # have orphaned if the press landed mid-sample (else it spins and
        # wedges CUDA).
        Write-Warning "Cell cancelled by user (R exit 130). Aborting sweep."
        [GridSweepCleanup]::Cancelled = $true
        $cancelled = $true
        break groupLoop
      }
      if ($cellExit -ne 0) {
        # A real cell failure: log + continue. The JSONL still has
        # partial data for this cell; the operator can investigate.
        Write-Warning "Cell failed (exit $cellExit). Continuing sweep."
      }
    }

    if ($needsLock) {
      Release-Lock
    }
  }
  if ($cancelled) {
    Write-Host ""
    Write-Host "Sweep aborted on user cancel." -ForegroundColor Yellow
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
