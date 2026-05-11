param(
    [string]$DatasetRoot = "data/eodhd_us_equities_30y",
    [string]$EndDate = "2026-05-08",
    [string]$LogRoot = "logs/full_universe_retrain",
    [switch]$Resume,
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
    Write-Stage -Name $Name -State "END"
}

function Remove-IfExists {
    param([string]$PathText)
    $path = Join-Path $repoRoot $PathText
    if (Test-Path $path) {
        Remove-Item -LiteralPath $path -Recurse -Force
    }
}

function Invoke-Cleanup {
    param([string]$Name)
    Write-Stage -Name $Name -State "START"
    if ($DryRun) {
        Write-Output "DRYRUN cleanup true-full outputs"
        Write-Stage -Name $Name -State "END"
        return
    }
    $paths = @(
        "data/eodhd_training_panels/eodhd_true_full_walk_forward",
        "artifacts/v1_baselines/eodhd_true_full_walk_forward",
        "artifacts/v1_baselines/eodhd_true_full_xgboost",
        "artifacts/v1_baselines/eodhd_true_full_torch_mlp",
        "artifacts/v1_baselines/eodhd_true_full_torch_seq_static",
        "logs/v1_pipeline_state/eodhd_true_full_walk_forward.json",
        "logs/v1_pipeline_state/eodhd_true_full_xgboost.json",
        "logs/v1_pipeline_state/eodhd_true_full_torch_mlp.json",
        "logs/v1_pipeline_state/eodhd_true_full_torch_seq_static.json"
    )
    foreach ($relative in $paths) {
        Remove-IfExists -PathText $relative
    }
    Write-Stage -Name $Name -State "END"
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
    $python = "py"
    $basePyArgs = @("-3.11")
    $pipelineProfile = "configs/v1_runs/eodhd_true_full_walk_forward.json"
    $xgbProfile = "configs/v1_runs/eodhd_true_full_xgboost.json"
    $mlpProfile = "configs/v1_runs/eodhd_true_full_torch_mlp.json"
    $seqProfile = "configs/v1_runs/eodhd_true_full_torch_seq_static.json"

    if (-not $Resume) {
        Invoke-Cleanup -Name "cleanup_true_full_outputs"
    }

    Invoke-Step -Name "refresh_root_raw" -Command (
        @($python) + $basePyArgs +
        @(
            "scripts/update_eodhd_daily_dataset.py",
            "--dataset-root", $DatasetRoot,
            "--start-date", "1995-01-01",
            "--end-date", $EndDate,
            "--max-tickers", "0",
            "--fetch-only",
            "--recent-overlap-days", "7"
        )
    )

    if ($Resume) {
        Invoke-Step -Name "rebuild_root_features_resume" -Command (
            @($python) + $basePyArgs +
            @(
                "scripts/build_eodhd_daily_features_chunked.py",
                "--dataset-root", $DatasetRoot,
                "--max-tickers", "0",
                "--resume"
            )
        )
    } else {
        Invoke-Step -Name "rebuild_root_features" -Command (
            @($python) + $basePyArgs +
            @(
                "scripts/build_eodhd_daily_features_chunked.py",
                "--dataset-root", $DatasetRoot,
                "--max-tickers", "0",
                "--force"
            )
        )
    }

    Invoke-Step -Name "rebuild_root_normalized" -Command (
        @($python) + $basePyArgs +
        @(
            "scripts/build_normalized_features_from_processed.py",
            "--dataset-root", $DatasetRoot
        )
    )

    $resumeArgs = @()
    if ($Resume) {
        $resumeArgs = @("--resume")
    }

    Invoke-Step -Name "materialize_panel" -Command (
        @($python) + $basePyArgs +
        @(
            "scripts/run_v1_pipeline.py",
            "--profile", $pipelineProfile,
            "--stage", "materialize_panel"
        ) + $resumeArgs
    )

    Invoke-Step -Name "materialize_cache" -Command (
        @($python) + $basePyArgs +
        @(
            "scripts/run_v1_pipeline.py",
            "--profile", $pipelineProfile,
            "--stage", "materialize_cache"
        ) + $resumeArgs
    )

    Invoke-Step -Name "train_xgboost" -Command (
        @($python) + $basePyArgs +
        @(
            "scripts/run_v1_pipeline.py",
            "--profile", $xgbProfile
        ) + $resumeArgs
    )

    Invoke-Step -Name "train_torch_mlp" -Command (
        @($python) + $basePyArgs +
        @(
            "scripts/run_v1_pipeline.py",
            "--profile", $mlpProfile
        ) + $resumeArgs
    )

    Invoke-Step -Name "train_torch_seq_static" -Command (
        @($python) + $basePyArgs +
        @(
            "scripts/run_v1_pipeline.py",
            "--profile", $seqProfile
        ) + $resumeArgs
    )
}
finally {
    [KeepAwakeNative]::SetThreadExecutionState([uint32]$ES_CONTINUOUS) | Out-Null
}
