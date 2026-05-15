param(
    [string]$DatasetRoot = "data/eodhd_us_equities_30y",
    [string]$EndDate = "2026-05-08",
    [string]$FastRoot = "",
    [string]$LogRoot = "logs/full_universe_retrain",
    [switch]$Resume,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
if ([string]::IsNullOrWhiteSpace($FastRoot)) {
    $FastRoot = $repoRoot
}

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
        [string[]]$Command,
        [string]$ResumeStage = ""
    )
    if ($Resume -and $ResumeStage -and (Test-StageOutput -Stage $ResumeStage)) {
        Write-Stage -Name $Name -State "SKIP"
        Write-Output "Skipping completed stage via -Resume: $Name"
        return
    }
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

function Remove-IfExists {
    param([string]$PathText)
    if ([System.IO.Path]::IsPathRooted($PathText)) {
        $path = $PathText
    } else {
        $path = Join-Path $repoRoot $PathText
    }
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
        "$FastRoot/data/eodhd_training_panels/eodhd_true_full_walk_forward",
        "$FastRoot/artifacts/v1_baselines/eodhd_true_full_walk_forward",
        "$FastRoot/artifacts/v1_baselines/eodhd_true_full_xgboost",
        "$FastRoot/artifacts/v1_baselines/eodhd_true_full_torch_mlp",
        "$FastRoot/artifacts/v1_baselines/eodhd_true_full_torch_seq_static",
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

function Test-StageOutput {
    param(
        [ValidateSet("raw_refresh","feature_build","normalized","panel","cache","train_xgboost","train_torch_mlp","train_torch_seq_static")]
        [string]$Stage
    )
    switch ($Stage) {
        "raw_refresh" {
            $path = Join-Path $repoRoot "$DatasetRoot/raw/eodhd_fetch_manifest.json"
            return (Test-Path $path)
        }
        "feature_build" {
            $path = Join-Path $repoRoot "$DatasetRoot/processed/daily_features_chunked_manifest.json"
            return (Test-Path $path)
        }
        "normalized" {
            $manifest = Join-Path $repoRoot "$DatasetRoot/processed/daily_features_normalized_manifest.json"
            $data = Join-Path $repoRoot "$DatasetRoot/processed/daily_features_normalized.csv"
            $features = Join-Path $repoRoot "$DatasetRoot/processed/daily_features.csv"
            if (-not ((Test-Path $manifest) -and (Test-Path $data))) {
                return $false
            }
            if (Test-Path $features) {
                return ((Get-Item $data).LastWriteTime -ge (Get-Item $features).LastWriteTime) -and ((Get-Item $manifest).LastWriteTime -ge (Get-Item $features).LastWriteTime)
            }
            return $true
        }
        "panel" {
            $path = Join-Path $FastRoot "data/eodhd_training_panels/eodhd_true_full_walk_forward/processed/materialized_panel_manifest.json"
            return (Test-Path $path)
        }
        "cache" {
            $path = Join-Path $FastRoot "data/eodhd_training_panels/eodhd_true_full_walk_forward/episode_cache/manifest.json"
            return (Test-Path $path)
        }
        "train_xgboost" {
            $path = Join-Path $FastRoot "artifacts/v1_baselines/eodhd_true_full_xgboost/final_models.json"
            return (Test-Path $path)
        }
        "train_torch_mlp" {
            $path = Join-Path $FastRoot "artifacts/v1_baselines/eodhd_true_full_torch_mlp/final_models.json"
            return (Test-Path $path)
        }
        "train_torch_seq_static" {
            $path = Join-Path $FastRoot "artifacts/v1_baselines/eodhd_true_full_torch_seq_static/final_models.json"
            return (Test-Path $path)
        }
    }
    return $false
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
    ) -ResumeStage "raw_refresh"

    if ($Resume) {
        Invoke-Step -Name "rebuild_root_features_resume" -Command (
            @($python) + $basePyArgs +
            @(
                "scripts/build_eodhd_daily_features_chunked.py",
                "--dataset-root", $DatasetRoot,
                "--max-tickers", "0",
                "--resume"
            )
        ) -ResumeStage "feature_build"
    } else {
        Invoke-Step -Name "rebuild_root_features" -Command (
            @($python) + $basePyArgs +
            @(
                "scripts/build_eodhd_daily_features_chunked.py",
                "--dataset-root", $DatasetRoot,
                "--max-tickers", "0",
                "--force"
            )
        ) -ResumeStage "feature_build"
    }

    Invoke-Step -Name "merge_incremental_feature_updates" -Command (
        @($python) + $basePyArgs +
        @(
            "scripts/merge_incremental_feature_updates.py",
            "--dataset-root", $DatasetRoot
        )
    )

    Invoke-Step -Name "rebuild_root_normalized" -Command (
        @($python) + $basePyArgs +
        @(
            "scripts/build_normalized_features_from_processed.py",
            "--dataset-root", $DatasetRoot,
            "--resume",
            "--bucket-max-open-files", "256"
        )
    ) -ResumeStage "normalized"

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
    ) -ResumeStage "panel"

    Invoke-Step -Name "materialize_cache" -Command (
        @($python) + $basePyArgs +
        @(
            "scripts/run_v1_pipeline.py",
            "--profile", $pipelineProfile,
            "--stage", "materialize_cache"
        ) + $resumeArgs
    ) -ResumeStage "cache"

    Invoke-Step -Name "train_xgboost" -Command (
        @($python) + $basePyArgs +
        @(
            "scripts/run_v1_pipeline.py",
            "--profile", $xgbProfile
        ) + $resumeArgs
    ) -ResumeStage "train_xgboost"

    Invoke-Step -Name "train_torch_mlp" -Command (
        @($python) + $basePyArgs +
        @(
            "scripts/run_v1_pipeline.py",
            "--profile", $mlpProfile
        ) + $resumeArgs
    ) -ResumeStage "train_torch_mlp"

    Invoke-Step -Name "train_torch_seq_static" -Command (
        @($python) + $basePyArgs +
        @(
            "scripts/run_v1_pipeline.py",
            "--profile", $seqProfile
        ) + $resumeArgs
    ) -ResumeStage "train_torch_seq_static"
}
finally {
    [KeepAwakeNative]::SetThreadExecutionState([uint32]$ES_CONTINUOUS) | Out-Null
}
