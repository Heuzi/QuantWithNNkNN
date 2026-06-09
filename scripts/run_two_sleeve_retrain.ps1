param(
    [ValidateSet("both", "conservative", "momentum_breakout")]
    [string]$Sleeve = "both",
    [string]$DatasetRoot = "data/eodhd_us_equities_30y",
    [string]$StartDate = "1995-01-01",
    [string]$EndDate = "",
    [switch]$Resume,
    [switch]$SkipDataRefresh,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Write-Stage {
    param(
        [string]$Name,
        [string]$State
    )
    $timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"
    Write-Output "=== $State $Name $timestamp ==="
}

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$Command
    )
    Write-Stage -Name $Name -State "START"
    if ($DryRun) {
        Write-Output ("DRYRUN " + ($Command -join " "))
        Write-Stage -Name $Name -State "END"
        return
    }
    & $Command[0] $Command[1..($Command.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        throw "Step '$Name' failed with exit code $LASTEXITCODE."
    }
    Write-Stage -Name $Name -State "END"
}

function Invoke-Profile {
    param([string]$Profile)
    $command = @("py", "-3.11", "scripts/run_v1_pipeline.py", "--profile", $Profile)
    if ($Resume) {
        $command += "--resume"
    }
    Invoke-Step -Name $Profile -Command $command
}

function Selected-Sleeves {
    if ($Sleeve -eq "both") {
        return @("conservative", "momentum_breakout")
    }
    return @($Sleeve)
}

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class KeepAwakeNative {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@

[uint32]$ES_CONTINUOUS = [Convert]::ToUInt32("80000000", 16)
[uint32]$ES_SYSTEM_REQUIRED = [Convert]::ToUInt32("00000001", 16)
[uint32]$ES_AWAYMODE_REQUIRED = [Convert]::ToUInt32("00000040", 16)
[KeepAwakeNative]::SetThreadExecutionState([uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED -bor $ES_AWAYMODE_REQUIRED)) | Out-Null

try {
    if (-not $SkipDataRefresh) {
        $refresh = @(
            "py", "-3.11",
            "scripts/update_eodhd_daily_dataset.py",
            "--dataset-root", $DatasetRoot,
            "--start-date", $StartDate,
            "--max-tickers", "0",
            "--fetch-only",
            "--recent-overlap-days", "7"
        )
        if (-not [string]::IsNullOrWhiteSpace($EndDate)) {
            $refresh += @("--end-date", $EndDate)
        }
        Invoke-Step -Name "refresh_root_raw" -Command $refresh

        $featureCommand = @(
            "py", "-3.11",
            "scripts/build_eodhd_daily_features_chunked.py",
            "--dataset-root", $DatasetRoot,
            "--max-tickers", "0"
        )
        if ($Resume) {
            $featureCommand += "--resume"
        } else {
            $featureCommand += "--force"
        }
        Invoke-Step -Name "rebuild_root_features" -Command $featureCommand

        Invoke-Step -Name "merge_incremental_feature_updates" -Command @(
            "py", "-3.11",
            "scripts/merge_incremental_feature_updates.py",
            "--dataset-root", $DatasetRoot
        )

        Invoke-Step -Name "rebuild_root_normalized" -Command @(
            "py", "-3.11",
            "scripts/build_normalized_features_from_processed.py",
            "--dataset-root", $DatasetRoot,
            "--resume",
            "--bucket-max-open-files", "256"
        )
    }

    foreach ($selectedSleeve in Selected-Sleeves) {
        if ($selectedSleeve -eq "momentum_breakout") {
            Invoke-Profile -Profile "configs/v1_runs/eodhd_sleeve_momentum_breakout_model_selection_cache.json"
            Invoke-Profile -Profile "configs/v1_runs/eodhd_sleeve_momentum_breakout_model_selection.json"
        } else {
            Invoke-Profile -Profile "configs/v1_runs/eodhd_sleeve_${selectedSleeve}_cache.json"
            Invoke-Profile -Profile "configs/v1_runs/eodhd_sleeve_${selectedSleeve}_xgboost.json"
            Invoke-Profile -Profile "configs/v1_runs/eodhd_sleeve_${selectedSleeve}_torch_mlp.json"
            Invoke-Profile -Profile "configs/v1_runs/eodhd_sleeve_${selectedSleeve}_torch_seq_static.json"
        }
    }
}
finally {
    [KeepAwakeNative]::SetThreadExecutionState([uint32]$ES_CONTINUOUS) | Out-Null
}
