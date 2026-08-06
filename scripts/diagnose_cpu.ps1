<#
.SYNOPSIS
    Measure how much CPU the head tracker and Mach4 actually consume.

.DESCRIPTION
    The head tracking loop is camera-rate-limited at ~30 Hz, and most of its
    largest stage (frame acquire + align) is spent blocked waiting for the next
    frame rather than computing. That makes "the tracker is eating the CPU" a
    hypothesis rather than a fact.

    This samples each process's accumulated processor time once a second and
    converts it to cores consumed. Mean tells you the sustained load; peak tells
    you whether the native thread pools burst wide, which is what disrupts a
    12 ms real-time loop even when the average looks harmless.

.EXAMPLE
    .\scripts\diagnose_cpu.ps1 -Seconds 30

.EXAMPLE
    .\scripts\diagnose_cpu.ps1 -TrackerPid 12345
#>
param(
    [int]$Seconds = 20,
    [int]$TrackerPid = 0,
    [int]$Mach4Pid = 0
)

$ErrorActionPreference = 'Stop'

function Resolve-Target {
    param([string]$Label, [int]$ExplicitPid, [string[]]$Names, [string]$CommandLineMatch)

    if ($ExplicitPid -gt 0) {
        return Get-Process -Id $ExplicitPid
    }
    foreach ($name in $Names) {
        $found = Get-Process -Name $name -ErrorAction SilentlyContinue
        if ($found) {
            return ($found | Sort-Object CPU -Descending | Select-Object -First 1)
        }
    }
    if ($CommandLineMatch) {
        $cim = Get-CimInstance Win32_Process |
            Where-Object { $_.CommandLine -and $_.CommandLine -match $CommandLineMatch } |
            Select-Object -First 1
        if ($cim) { return Get-Process -Id $cim.ProcessId }
    }
    Write-Warning "$Label not found; skipping."
    return $null
}

$cores = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
Write-Host "logical processors: $cores" -ForegroundColor Cyan

$tracker = Resolve-Target -Label 'tracker' -ExplicitPid $TrackerPid `
    -Names @('orbbec-head-stream-cnc') -CommandLineMatch 'stream_cnc|orbbec-head-stream-cnc'
$mach4 = Resolve-Target -Label 'Mach4' -ExplicitPid $Mach4Pid `
    -Names @('Mach4GUI', 'Mach4Hobby', 'Mach4Industrial') -CommandLineMatch $null

if (-not $tracker -and -not $mach4) {
    throw 'Neither process is running. Start them, then run this again.'
}

foreach ($proc in @($tracker, $mach4)) {
    if ($proc) { Write-Host ("watching {0} (pid {1})" -f $proc.ProcessName, $proc.Id) }
}

$samples = @{}
foreach ($key in 'tracker', 'mach4') { $samples[$key] = New-Object System.Collections.ArrayList }

function Get-CpuSeconds {
    param($Proc)
    if (-not $Proc) { return $null }
    try { return (Get-Process -Id $Proc.Id).TotalProcessorTime.TotalSeconds }
    catch { return $null }
}

$prevTracker = Get-CpuSeconds $tracker
$prevMach4 = Get-CpuSeconds $mach4
$prevClock = Get-Date

Write-Host ''
Write-Host ('{0,6}  {1,14}  {2,14}' -f 'sec', 'tracker cores', 'mach4 cores')

for ($i = 1; $i -le $Seconds; $i++) {
    Start-Sleep -Milliseconds 1000
    $now = Get-Date
    $elapsed = ($now - $prevClock).TotalSeconds
    $prevClock = $now

    $curTracker = Get-CpuSeconds $tracker
    $curMach4 = Get-CpuSeconds $mach4

    $tCores = $null
    if ($null -ne $curTracker -and $null -ne $prevTracker) {
        $tCores = ($curTracker - $prevTracker) / $elapsed
        [void]$samples['tracker'].Add($tCores)
    }
    $mCores = $null
    if ($null -ne $curMach4 -and $null -ne $prevMach4) {
        $mCores = ($curMach4 - $prevMach4) / $elapsed
        [void]$samples['mach4'].Add($mCores)
    }
    $prevTracker = $curTracker
    $prevMach4 = $curMach4

    $tText = if ($null -ne $tCores) { '{0,14:N2}' -f $tCores } else { '{0,14}' -f '-' }
    $mText = if ($null -ne $mCores) { '{0,14:N2}' -f $mCores } else { '{0,14}' -f '-' }
    Write-Host ('{0,6}  {1}  {2}' -f $i, $tText, $mText)
}

Write-Host ''
foreach ($key in 'tracker', 'mach4') {
    $values = $samples[$key]
    if ($values.Count -eq 0) { continue }
    $stats = $values | Measure-Object -Average -Maximum
    Write-Host ("{0,-8} mean {1,5:N2} cores  peak {2,5:N2} cores  ({3,4:N1}% / {4,4:N1}% of machine)" -f `
            $key, $stats.Average, $stats.Maximum, `
        (100 * $stats.Average / $cores), (100 * $stats.Maximum / $cores)) -ForegroundColor Green
}

Write-Host ''
Write-Host 'Reading the result:' -ForegroundColor Cyan
Write-Host '  tracker mean near 1-2 cores  -> CPU contention is essentially ruled out;'
Write-Host '                                  look at DPC latency and the USB stack instead.'
Write-Host '  tracker peak much above mean -> native thread pools burst wide; the affinity'
Write-Host '                                  split is worth testing.'
Write-Host '  tracker mean above ~6 cores  -> sustained load; pin it off Mach4 P-cores.'
