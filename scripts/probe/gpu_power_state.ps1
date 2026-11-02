<#
.SYNOPSIS
Report -- and optionally set -- the platform state that governs the dGPU's
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
This CHANGES THE MACHINE, not just the report: it switches the physical
thermal profile (fan curve and the platform's power behaviour), and the
setting persists after the script exits. Every baseline in
data/baselines.json was recorded in Performance mode. Changing the mode in
the middle of a measurement campaign breaks comparability with what came
before, so set it before a campaign or not at all.

.PARAMETER LogPath
Where to write the transcript. Defaults to gpu_power_state.log in TEMP.
Written UTF-8; PowerShell's default redirection would produce UTF-16,
which reads as spaced-out garbage from WSL.

.PARAMETER Json
Emit a single JSON object on stdout instead of the human report, so a
harness can consume it.

.PARAMETER JsonPath
With -Json, also write the JSON object to this file (UTF-8, no BOM). The
elevated relaunch needs it: an elevated child's stdout never reaches the
WSL terminal that started it, so the wrapper reads this file instead.

.PARAMETER Reference
What "capped" is measured against: 'max' (the VBIOS ceiling, the default,
right for this laptop whose normal point is the 150 W an OEM performance
mode raises it to) or 'default' (right for stock hardware whose normal
operating point IS the default limit). Matches the wrapper's --reference.

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
    [switch]$Json,
    [string]$JsonPath,
    [ValidateSet('max', 'default')]
    [string]$Reference = 'max'
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
    power_limit_before_set_w = $null
    power_reference       = $Reference
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

# The verdict on one nvidia-smi reading: enforced vs the chosen -Reference.
# A function because it runs twice: on the initial reading, and again
# after -SetMode with the reading taken after the change (labelled by
# $when), so the report never carries a verdict about a state that no
# longer holds.
#
# InvariantCulture, explicitly. nvidia-smi always emits "50.00" with a
# dot, but the shell locale here is de-DE, where the bare
# [double]::TryParse overload reads "." as a THOUSANDS separator and
# turns 50.00 W into 5000 W. Observed: "enforced 5000 W is below the
# 15000 W ceiling". The verdict survived (both sides scale alike) but
# every number in the report was wrong by 100x.
function Write-PowerVerdict([object[]]$fields, [string]$when) {
    $inv = [System.Globalization.CultureInfo]::InvariantCulture
    $sty = [System.Globalization.NumberStyles]::Float
    $enforced = 0.0; $refw = 0.0
    $refField = $(if ($Reference -eq 'default') { $fields[1] } else { $fields[2] })
    if ([double]::TryParse($fields[0], $sty, $inv, [ref]$enforced) -and
        [double]::TryParse($refField, $sty, $inv, [ref]$refw) -and $refw -gt 0) {
        if ($enforced -lt ($refw - 1)) {
            $msg = "CAPPED${when}: enforced $enforced W is below the $refw W $Reference reference. " +
                   "Measurements taken now are NOT comparable to baselines."
            Log ("  !! " + $msg)
            $result.notes += $msg
        } else {
            Log "  ok${when}: enforced limit is at or above the $Reference reference ($refw W)."
        }
    } else {
        $msg = "UNKNOWN${when}: enforced or $Reference limit is not a positive number " +
               "(enforced='$($fields[0])', $Reference='$refField')."
        Log ("  !! " + $msg)
        $result.notes += $msg
    }
}

# @(...): a single value comes back from the pipeline as a bare string, and
# .Count on a string throws PropertyNotFoundStrict under Set-StrictMode
# Latest -- which skipped this whole block without saying "unavailable".
$smi = @(Get-SmiFields)
if ($smi.Count -ge 5) {
    $result.power_limit_w         = $smi[0]
    $result.power_limit_default_w = $smi[1]
    $result.power_limit_max_w     = $smi[2]
    $result.clock_max_sm_mhz      = $smi[3]
    $result.clock_max_mem_mhz     = $smi[4]
    Log ("power limit (enforced/default/max) : {0} / {1} / {2} W" -f $smi[0], $smi[1], $smi[2])
    Log ("max clocks  (sm/mem)               : {0} / {1} MHz" -f $smi[3], $smi[4])
    Write-PowerVerdict $smi ''
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
    # Each value read on its own, through PSObject.Properties: under
    # Set-StrictMode -Version Latest, `$p.ActiveOverlayDcPowerScheme` THROWS
    # when the value does not exist, which used to discard an AC value
    # already read and report the whole overlay as unreadable.
    $acProp = $p.PSObject.Properties['ActiveOverlayAcPowerScheme']
    $dcProp = $p.PSObject.Properties['ActiveOverlayDcPowerScheme']
    $result.overlay_ac = $(if ($acProp) { "$($acProp.Value)".ToLower() } else { 'absent' })
    $result.overlay_dc = $(if ($dcProp) { "$($dcProp.Value)".ToLower() } else { 'absent' })
    $result.overlay_name = $(if ($overlays.ContainsKey($result.overlay_ac)) { $overlays[$result.overlay_ac] } else { $result.overlay_ac })
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
        #
        # Get-CimClass, not $gz.GetType().GetMethods(): the latter lists the
        # .NET members of System.Management.ManagementObject, never the
        # firmware's WMI methods, so GetSmartFanMode could not appear in it.
        # Its own try: discovery must not cost us the SmartFanMode read below.
        try {
            $result.gamezone_methods = @(
                (Get-CimClass -Namespace root\WMI -ClassName LENOVO_GAMEZONE_DATA -ErrorAction Stop
                    ).CimClassMethods | ForEach-Object { $_.Name } | Sort-Object)
            Log ("gamezone methods: " + ($result.gamezone_methods -join ', '))
        } catch {
            Log ("gamezone methods: unreadable (" + $_.Exception.Message + ")")
            $result.notes += "gamezone method discovery failed: $($_.Exception.Message)"
        }

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

                $smi2 = @(Get-SmiFields)
                if ($smi2.Count -ge 3) {
                    Log ("power limit after : {0} / {1} / {2} W" -f $smi2[0], $smi2[1], $smi2[2])
                    if ($smi2[0] -eq $result.power_limit_w) {
                        $msg = "thermal mode changed but the enforced limit did NOT move " +
                               "($($smi2[0]) W) -- the cap is set by something else " +
                               "(adapter wattage / MUX mode / firmware)."
                        Log ("  !! " + $msg)
                        $result.notes += $msg
                    }
                    # Refresh ALL three limits and re-run the verdict on
                    # this reading. Only power_limit_w used to be updated,
                    # so the JSON could say CAPPED about a state the mode
                    # change had already fixed, while the wrapper (which
                    # re-reads after this) exited 0 -- or the reverse.
                    $result.power_limit_before_set_w = $result.power_limit_w
                    $result.power_limit_w            = $smi2[0]
                    $result.power_limit_default_w    = $smi2[1]
                    $result.power_limit_max_w        = $smi2[2]
                    Write-PowerVerdict $smi2 ' (after set-mode)'
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
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

# Failures below go to STDERR and into the record, never to stdout. With
# -Json, stdout must carry the JSON object and nothing else. Write-Warning
# used to put text there ahead of it.
try {
    [System.IO.File]::WriteAllLines($LogPath, $script:Lines, $utf8NoBom)
} catch {
    $m = "could not write log to $LogPath : $($_.Exception.Message)"
    $result.notes += $m
    [Console]::Error.WriteLine($m)
}

if ($Json) {
    $jsonText = $result | ConvertTo-Json -Depth 4 -Compress
    if ($JsonPath) {
        try {
            [System.IO.File]::WriteAllText($JsonPath, $jsonText, $utf8NoBom)
        } catch {
            $m = "could not write JSON to $JsonPath : $($_.Exception.Message)"
            [Console]::Error.WriteLine($m)
            # In the elevated case this console closes with the child, and the
            # wrapper points the user at the log when the JSON is missing. So
            # the reason goes into the log too.
            try {
                [System.IO.File]::AppendAllText($LogPath, $m + [Environment]::NewLine, $utf8NoBom)
            } catch { }
        }
    }
    $jsonText
}
