param(
    [string]$DatasetRoot = "data\eodhd_us_equities_30y",
    [string]$CredentialsPath = "EODHD_api_key",
    [string]$OutputRoot = "artifacts\production_reports",
    [string]$ReportName = "",
    [string]$AnchorDate = "",
    [string]$FetchEndDate = "",
    [string]$PositionLedger = "data\open_positions.csv",
    [int]$MaxTickers = 0,
    [switch]$SkipFetch,
    [switch]$ForceRebuildLatestInference,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

if ([string]::IsNullOrWhiteSpace($ReportName)) {
    $ReportName = "latest_two_sleeve_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
}

$conservativeReport = "${ReportName}_conservative"
$momentumReport = "${ReportName}_momentum_breakout"
$combinedDir = Join-Path $OutputRoot $ReportName

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$Command
    )
    $timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"
    Write-Output "=== START $Name $timestamp ==="
    Write-Output ($Command -join " ")
    if ($DryRun) {
        Write-Output "DRYRUN"
        return
    }
    & $Command[0] $Command[1..($Command.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        throw "Step '$Name' failed with exit code $LASTEXITCODE."
    }
    $timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"
    Write-Output "=== END $Name $timestamp ==="
}

function Add-SharedArgs {
    param([string[]]$Command)
    $out = $Command + @(
        "--dataset-root", $DatasetRoot,
        "--credentials-path", $CredentialsPath,
        "--output-root", $OutputRoot,
        "--position-ledger", $PositionLedger
    )
    if ($MaxTickers -gt 0) {
        $out += @("--max-tickers", [string]$MaxTickers)
    }
    if (-not [string]::IsNullOrWhiteSpace($AnchorDate)) {
        $out += @("--anchor-date", $AnchorDate)
    }
    if (-not [string]::IsNullOrWhiteSpace($FetchEndDate)) {
        $out += @("--fetch-end-date", $FetchEndDate)
    }
    return $out
}

$conservative = Add-SharedArgs @(
    "py", "-3.11", "scripts\run_trading_strategy.py",
    "--report-name", $conservativeReport,
    "--leaderboard-rank", "1",
    "--run-dir", "artifacts\v1_baselines\eodhd_true_full_xgboost",
    "--run-dir", "artifacts\v1_baselines\eodhd_true_full_ablation_torch_mlp",
    "--run-dir", "artifacts\v1_baselines\eodhd_true_full_ablation_torch_seq_static"
)
if ($SkipFetch) {
    $conservative += "--skip-fetch"
} elseif ($ForceRebuildLatestInference) {
    $conservative += "--force-rebuild-latest-inference"
}

Invoke-Step -Name "conservative_trading_strategy" -Command $conservative

$momentum = Add-SharedArgs @(
    "py", "-3.11", "scripts\run_trading_strategy.py",
    "--report-name", $momentumReport,
    "--skip-fetch",
    "--leaderboard-top-k", "3",
    "--run-dir", "artifacts\v1_baselines\eodhd_sleeve_momentum_breakout_model_selection"
)
Invoke-Step -Name "momentum_breakout_trading_strategy" -Command $momentum

Invoke-Step -Name "combine_sleeve_reports" -Command @(
    "py", "-3.11", "scripts\combine_sleeve_trading_reports.py",
    "--output-dir", $combinedDir,
    "--sleeve-report", "conservative=$(Join-Path $OutputRoot $conservativeReport)",
    "--sleeve-report", "momentum_breakout=$(Join-Path $OutputRoot $momentumReport)"
)

Write-Output "Combined report: $combinedDir"
