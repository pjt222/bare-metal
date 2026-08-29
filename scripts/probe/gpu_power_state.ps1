<#
.SYNOPSIS
Report — and optionally set — the platform state that governs the dGPU's
power envelope, so a benchmark session can prove it is comparable.

.DESCRIPTION
A measurement on this machine is only comparable to a recorded baseline if
the GPU was allowed its full power envelope. `nvidia-smi`'s `power.draw`
cannot show that: it reports what the GPU *drew*, never what it was
*allowed* to draw. The enforced limit is set by the platform, and on
2026-08-13 it sat at 50 W against a 115 W default and a 150 W VBIOS max,
which put `conv2d_implicit_gemm` at 51% of its normal throughput with
nothing in the run record to explain it (issue #207).

Three separate levers were suspected and only some are observable without
Administrator, so this tool gathers all of them in one place:

  - enforced / default / max power limit   (nvidia-smi, no elevation)
  - memory + SM clock ceiling              (nvidia-smi, no elevation)
  - Windows power-mode overlay             (registry, no elevation)
  - Lenovo SmartFanMode                    (root\WMI, ELEVATION REQUIRED)
  - Lenovo GameZone method surface         (root\WMI, ELEVATION REQUIRED)

Measured on this machine 2026-08-13: the Windows overlay and the Lenovo
thermal mode were BOTH already at their performance settings while the
enforced limit stayed at 50 W, so neither is sufficient on its own. Report
everything; do not assume one lever explains a cap.

Elevation is checked and reported, never assumed. Deliberately no
`#Requires -RunAsAdministrator`: that refuses to launch the script, so
nothing reaches the log and the elevation branch becomes dead code. This
is the convention used by the sibling `fancontrol` project's `tools/`,
which is the reference implementation for the Lenovo WMI surface.

.PARAMETER SetMode
Set the Lenovo SmartFanMode: 1 Quiet, 2 Balanced, 3 Performance,
255 Custom. Requires elevation. Omit to report only.

.PARAMETER LogPath
Where to write the transcript. Defaults to gpu_power_state.log in TEMP.
Written UTF-8; PowerShell's default redirection would produce UTF-16,
which reads as spaced-out garbage from WSL.

.PARAMETER Json
Emit a single JSON object on stdout instead of the human report, so a
harness can consume it.

.EXAMPLE
  # Report only, no elevation needed for the nvidia-smi half
  powershell.exe -ExecutionPolicy Bypass -File scripts/probe/gpu_power_state.ps1

.EXAMPLE
  # From WSL, with a UAC prompt, and set Performance mode
  scripts/probe/gpu_power_state.sh --elevated --set-mode 3
#>
[CmdletBinding()]
param(
    [ValidateSet(1, 2, 3, 255)]
    [int]$SetMode = 0,
    [string]$LogPath,
    [switch]$Json
)

$ErrorActionPreference = 'Continue'
Set-StrictMode -Version Latest

if (-not $LogPath) { $LogPath = Join-Path $env:TEMP 'gpu_power_state.log' }

$script:Lines = New-Object System.Collections.Generic.List[string]
function Log([string]$m) {
    $script:Lines.Add($m) | Out-Null
    if (-not $Json) { Write-Output $m }
}

$result = [ordered]@{
    timestamp             = (Get-Date -Format o)
    elevated              = $false
    power_limit_w         = $null
    power_limit_default_w = $null
    power_limit_max_w     = $null
    clock_max_sm_mhz      = $null
    clock_max_mem_mhz     = $null
    overlay_ac            = $null
    overlay_dc            = $null
    overlay_name          = $null
    smart_fan_mode        = $null
    smart_fan_mode_name   = $null
    gamezone_methods      = @()
    set_mode_requested    = $(if ($SetMode -gt 0) { $SetMode } else { $null })
    set_mode_applied      = $null
    notes                 = @()
}

Log "=== GPU power state ==="
Log ("timestamp : " + $result.timestamp)

# ---- elevation ------------------------------------------------------------
$identity  = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
$result.elevated = $principal.IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
Log ("elevated  : " + $result.elevated)

# ---- nvidia-smi (no elevation) --------------------------------------------
function Get-SmiFields {
    try {
        $raw = & nvidia-smi --query-gpu=enforced.power.limit,power.default_limit,power.max_limit,clocks.max.sm,clocks.max.mem --format=csv,noheader,nounits 2>$null
        if (-not $raw) { return $null }
        ($raw -split ',') | ForEach-Object { $_.Trim() }
    } catch { $null }
}

$smi = Get-SmiFields
if ($smi -and $smi.Count -ge 5) {
    $result.power_limit_w         = $smi[0]
    $result.power_limit_default_w = $smi[1]
    $result.power_limit_max_w     = $smi[2]
    $result.clock_max_sm_mhz      = $smi[3]
    $result.clock_max_mem_mhz     = $smi[4]
    Log ("power limit (enforced/default/max) : {0} / {1} / {2} W" -f $smi[0], $smi[1], $smi[2])
    Log ("max clocks  (sm/mem)               : {0} / {1} MHz" -f $smi[3], $smi[4])

    # InvariantCulture, explicitly. nvidia-smi always emits "50.00" with a
    # dot, but the shell locale here is de-DE, where the bare
    # [double]::TryParse overload reads "." as a THOUSANDS separator and
    # turns 50.00 W into 5000 W. Observed: "enforced 5000 W is below the
    # 15000 W ceiling". The verdict survived (both sides scale alike) but
    # every number in the report was wrong by 100x.
    $inv = [System.Globalization.CultureInfo]::InvariantCulture
    $sty = [System.Globalization.NumberStyles]::Float
    $enforced = 0.0; $maxw = 0.0
    if ([double]::TryParse($smi[0], $sty, $inv, [ref]$enforced) -and
        [double]::TryParse($smi[2], $sty, $inv, [ref]$maxw) -and $maxw -gt 0) {
        if ($enforced -lt ($maxw - 1)) {
            $msg = "CAPPED: enforced $enforced W is below the $maxw W ceiling. " +
                   "Measurements taken now are NOT comparable to baselines."
            Log ("  !! " + $msg)
            $result.notes += $msg
        } else {
            Log "  ok: enforced limit is at the ceiling."
        }
    }
} else {
    Log "nvidia-smi: unavailable"
    $result.notes += 'nvidia-smi unavailable'
}

# ---- Windows power overlay (no elevation) ---------------------------------
$overlays = @{
    '961cc777-2547-4f9d-8174-7d86181b8a7a' = 'best-power-efficiency'
    '00000000-0000-0000-0000-000000000000' = 'balanced'
    'ded574b5-45a0-4f42-8737-46345c09c238' = 'best-performance'
    '3af9b8d9-7c97-431d-ad78-34a8bfea439f' = 'better-performance'
}
try {
    $key = 'HKLM:\SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes'
    $p = Get-ItemProperty $key -ErrorAction Stop
    $result.overlay_ac = "$($p.ActiveOverlayAcPowerScheme)".ToLower()
    $result.overlay_dc = "$($p.ActiveOverlayDcPowerScheme)".ToLower()
    $result.overlay_name = $overlays[$result.overlay_ac]
    if (-not $result.overlay_name) { $result.overlay_name = $result.overlay_ac }
    Log ("overlay AC : {0} ({1})" -f $result.overlay_ac, $result.overlay_name)
    Log ("overlay DC : {0}" -f $result.overlay_dc)
} catch {
    Log ("overlay    : unreadable (" + $_.Exception.Message + ")")
}

# ---- Lenovo WMI (elevation required) --------------------------------------
$modeNames = @{ 1 = 'quiet'; 2 = 'balanced'; 3 = 'performance'; 255 = 'custom' }

if (-not $result.elevated) {
    Log "Lenovo WMI : SKIPPED (needs Administrator; root\WMI returns access denied)"
    $result.notes += 'lenovo wmi skipped: not elevated'
} else {
    try {
        $gz = Get-WmiObject -Namespace root\WMI -Class LENOVO_GAMEZONE_DATA -ErrorAction Stop

        # Method surface, for discovery. The available methods differ by BIOS
        # and this is the cheapest way to see what this firmware exposes.
        $result.gamezone_methods = @(
            $gz.GetType().GetMethods() | ForEach-Object { $_.Name } |
                Where-Object { $_ -match '^(Get|Set|Is)' } | Sort-Object -Unique)
        if (-not $result.gamezone_methods) {
            $result.gamezone_methods = @(
                (Get-CimClass -Namespace root\WMI -ClassName LENOVO_GAMEZONE_DATA
                    ).CimClassMethods | ForEach-Object { $_.Name } | Sort-Object)
        }
        Log ("gamezone methods: " + ($result.gamezone_methods -join ', '))

        try {
            $m = $gz.GetSmartFanMode().Data
            $result.smart_fan_mode = $m
            $result.smart_fan_mode_name = $modeNames[[int]$m]
            Log ("SmartFanMode: {0} ({1})" -f $m, $result.smart_fan_mode_name)
        } catch {
            Log ("SmartFanMode: unreadable (" + $_.Exception.Message + ")")
        }

        if ($SetMode -gt 0) {
            try {
                $null = $gz.SetSmartFanMode($SetMode)
                Start-Sleep -Seconds 4
                $gz2 = Get-WmiObject -Namespace root\WMI -Class LENOVO_GAMEZONE_DATA
                $after = $gz2.GetSmartFanMode().Data
                $result.set_mode_applied = $after
                Log ("SetSmartFanMode({0}) -> mode now {1}" -f $SetMode, $after)

                $smi2 = Get-SmiFields
                if ($smi2 -and $smi2.Count -ge 3) {
                    Log ("power limit after : {0} / {1} / {2} W" -f $smi2[0], $smi2[1], $smi2[2])
                    if ($smi2[0] -eq $result.power_limit_w) {
                        $msg = "thermal mode changed but the enforced limit did NOT move " +
                               "($($smi2[0]) W) -- the cap is set by something else " +
                               "(adapter wattage / MUX mode / firmware)."
                        Log ("  !! " + $msg)
                        $result.notes += $msg
                    }
                    $result.power_limit_w = $smi2[0]
                }
            } catch {
                Log ("SetSmartFanMode failed: " + $_.Exception.Message)
                $result.notes += "set failed: $($_.Exception.Message)"
            }
        }
    } catch {
        Log ("Lenovo WMI : unreachable (" + $_.Exception.Message + ")")
        $result.notes += "lenovo wmi unreachable: $($_.Exception.Message)"
    }
}

Log "=== done ==="

# UTF-8 without a BOM: the log is read from WSL, and PowerShell's default
# redirection encoding (UTF-16LE) renders there as spaced-out garbage.
try {
    $enc = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllLines($LogPath, $script:Lines, $enc)
} catch {
    Write-Warning "could not write log to $LogPath : $($_.Exception.Message)"
}

if ($Json) { $result | ConvertTo-Json -Depth 4 -Compress }
