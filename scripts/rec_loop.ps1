# Long-run clip supervisor for collect_clips.py.
#
# scripts/collect_clips.py defaults --long-clips to 0 so a short edge-case
# batch is unchanged unless the supervisor passes the flag. This loop is the
# long-run config: it MUST pass --long-clips and --long-clip-s 600,900 so
# each batch actually records 10-15 minute regular-driving clips.
#
# Start-Process ExitCode can come back $null even after a successful batch
# (that used to trip `$rc -ne 0` and count as a halt, ending the loop with
# "3 consecutive halted batches"). Launch with -PassThru, cache $p.Handle,
# WaitForExit(), then read $p.ExitCode. A null code is unknown: verify from
# the session clip count / session-stop line instead of counting a halt.

[CmdletBinding()]
param(
    [int] $LongClips = 8,
    [string] $LongClipS = "600,900",
    [int] $Clips = 60,
    [string] $Out = "E:\GVD_data\clips_v2",
    [string] $Maps = "west_coast_usa,italy,east_coast_usa",
    [string] $Python = "py",
    [int] $HaltLimit = 3,
    [string] $Repo = ""
)

$ErrorActionPreference = "Stop"
if (-not $Repo) {
    $Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
$collect = Join-Path $Repo "scripts\collect_clips.py"

function Get-SessionOutcome {
    param([string] $OutRoot, $ExitCode)
    $sessions = @(Get-ChildItem -Path $OutRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { Test-Path (Join-Path $_.FullName "session.json") } |
        Sort-Object LastWriteTime -Descending)
    if ($sessions.Count -lt 1) {
        return [pscustomobject]@{ Kind = "unknown"; Reason = "no session.json under $OutRoot"; Halt = $false }
    }
    $session = $sessions[0].FullName
    $manifest = Get-Content (Join-Path $session "session.json") -Raw | ConvertFrom-Json
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

Write-Host "rec_loop: long-run --long-clips $LongClips --long-clip-s $LongClipS (10-15 min regular driving)"
$halts = 0
while ($true) {
    if (Test-Path (Join-Path $Out "STOP")) {
        Write-Host "rec_loop: STOP file present; exiting"
        break
    }
    $argList = @(
        $collect,
        "--out", $Out,
        "--clips", "$Clips",
        "--maps", $Maps,
        "--long-clips", "$LongClips",
        "--long-clip-s", $LongClipS
    )
    $p = Start-Process -FilePath $Python -ArgumentList $argList -PassThru -Wait:$false
    # Caching Handle makes ExitCode available after the process has exited.
    $null = $p.Handle
    $p.WaitForExit()
    $rc = $p.ExitCode
    $outcome = Get-SessionOutcome -OutRoot $Out -ExitCode $rc
    Write-Host "rec_loop: $($outcome.Kind) $($outcome.Reason)"
    if ($outcome.Halt) {
        $halts++
        Write-Host "rec_loop: halt $halts / $HaltLimit"
        if ($halts -ge $HaltLimit) {
            Write-Error "rec_loop: $HaltLimit consecutive halted batches"
            exit 2
        }
    } else {
        $halts = 0
    }
}
