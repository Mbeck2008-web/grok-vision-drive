# Long-run clip supervisor for collect_clips.py.
#
# scripts/collect_clips.py defaults --long-clips to 0 so a short edge-case
# batch is unchanged unless the supervisor passes the flag. This loop is the
# long-run config: it passes --long-clips and --long-clip-s 600,900 so each
# batch records 10-15 minute regular-driving clips.
#
# Start-Process ExitCode can come back $null even after a successful batch
# (that used to trip `$rc -ne 0` and count as a halt, ending the loop with
# "3 consecutive halted batches"). Invoke-Batch launches with -PassThru, caches
# $p.Handle, WaitForExit()s, then reads $p.ExitCode. A null code is unknown and
# is checked against the session.json of THIS launch (--session-id), never the
# newest folder: a batch that died before writing its session.json is a halt.
#
# Dot-source this file to get the functions (the loop does not run then).

[CmdletBinding()]
param(
    [int] $LongClips = 8,
    [string] $LongClipRange = "600,900",
    [int] $Clips = 60,
    [string] $Out = "E:\GVD_data\clips_v2",
    [string] $Maps = "west_coast_usa,italy,east_coast_usa",
    [string] $Python = "py",
    [string] $Collect = "",
    [int] $HaltLimit = 3,
    [string] $Repo = ""
)

$ErrorActionPreference = "Stop"

function Get-BatchArgs {
    param([string] $Collect, [string] $Out, [int] $Clips, [string] $Maps,
          [int] $LongClips, [string] $LongClipRange, [string] $SessionId)
    return @(
        $Collect,
        "--out", $Out,
        "--clips", "$Clips",
        "--maps", $Maps,
        "--long-clips", "$LongClips",
        "--long-clip-s", $LongClipRange,
        "--session-id", $SessionId
    )
}

function Invoke-Batch {
    param([string] $Python, [string[]] $ArgList)
    $p = Start-Process -FilePath $Python -ArgumentList $ArgList -PassThru -NoNewWindow
    # Caching Handle makes ExitCode readable after the process has exited.
    $null = $p.Handle
    $p.WaitForExit()
    return $p.ExitCode
}

function Get-SessionOutcome {
    param([string] $OutRoot, [string] $SessionId, $ExitCode)
    $file = Join-Path (Join-Path $OutRoot "clips_$SessionId") "session.json"
    $manifest = $null
    if (Test-Path $file) {
        $manifest = Get-Content $file -Raw | ConvertFrom-Json
        if ([string] $manifest.session_id -ne $SessionId) { $manifest = $null }
    }
    if ($null -eq $manifest) {
        return [pscustomobject]@{ Kind = "halt"; Reason = "no session.json for this launch ($SessionId); ExitCode=$ExitCode"; Halt = $true }
    }
    $got = @($manifest.results).Count
    $want = [int] $manifest.clips
    $stop = $manifest.stop_reason
    if ($null -eq $ExitCode) {
        if ($stop) {
            return [pscustomobject]@{ Kind = "unknown"; Reason = "null ExitCode; session-stop=$stop clips=$got/$want"; Halt = $true }
        }
        if ($got -ge $want -and $want -gt 0) {
            return [pscustomobject]@{ Kind = "unknown"; Reason = "null ExitCode; session completed $got/$want clips"; Halt = $false }
        }
        return [pscustomobject]@{ Kind = "unknown"; Reason = "null ExitCode; incomplete session $got/$want clips"; Halt = $true }
    }
    if ($ExitCode -ne 0) {
        return [pscustomobject]@{ Kind = "halt"; Reason = "ExitCode=$ExitCode stop=$stop clips=$got/$want"; Halt = $true }
    }
    return [pscustomobject]@{ Kind = "ok"; Reason = "ExitCode=0 clips=$got/$want"; Halt = $false }
}

if ($MyInvocation.InvocationName -eq ".") { return }

if (-not $Collect) {
    if (-not $Repo) { $Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path }
    $Collect = Join-Path $Repo "scripts\collect_clips.py"
}
Write-Host "rec_loop: long-run --long-clips $LongClips --long-clip-s $LongClipRange (10-15 min regular driving)"
$halts = 0
$n = 0
while ($true) {
    if (Test-Path (Join-Path $Out "STOP")) {
        Write-Host "rec_loop: STOP file present; exiting"
        break
    }
    $n++
    $sid = "{0}_{1}" -f (Get-Date -Format "yyyyMMdd_HHmmss"), $n
    $argList = Get-BatchArgs -Collect $Collect -Out $Out -Clips $Clips -Maps $Maps `
        -LongClips $LongClips -LongClipRange $LongClipRange -SessionId $sid
    $rc = Invoke-Batch -Python $Python -ArgList $argList
    $outcome = Get-SessionOutcome -OutRoot $Out -SessionId $sid -ExitCode $rc
    Write-Host "rec_loop: $($outcome.Kind) $($outcome.Reason)"
    if ($outcome.Halt) {
        $halts++
        Write-Host "rec_loop: halt $halts / $HaltLimit"
        if ($halts -ge $HaltLimit) {
            Write-Host "rec_loop: $HaltLimit consecutive halted batches"
            exit 2
        }
    } else {
        $halts = 0
    }
}
