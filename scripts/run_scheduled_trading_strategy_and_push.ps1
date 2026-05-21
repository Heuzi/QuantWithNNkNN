param(
    [string]$DatasetRoot = "data\eodhd_us_equities_30y",
    [string]$CredentialsPath = "EODHD_api_key",
    [string]$OutputRoot = "artifacts\production_reports",
    [string]$Remote = "origin",
    [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

$runDate = Get-Date -Format "yyyyMMdd"
$reportName = "latest_best3_$runDate"
$reportDir = Join-Path $OutputRoot $reportName
$logDir = Join-Path "logs" "trading_strategy"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logPath = Join-Path $logDir "scheduled_$reportName.log"

function Write-Log {
    param([string]$Message)
    $line = "$(Get-Date -Format o) | $Message"
    $line | Tee-Object -FilePath $logPath -Append
}

Write-Log "scheduled trading strategy run starting"
Write-Log "repo=$repoRoot report=$reportDir"

if (-not (Test-Path $CredentialsPath)) {
    throw "Missing credentials file: $CredentialsPath"
}

Write-Log "syncing $Remote/$Branch with --ff-only --autostash"
git pull --ff-only --autostash $Remote $Branch 2>&1 | Tee-Object -FilePath $logPath -Append
if ($LASTEXITCODE -ne 0) {
    throw "git pull failed with exit code $LASTEXITCODE"
}

$args = @(
    "scripts\run_trading_strategy.py",
    "--dataset-root", $DatasetRoot,
    "--credentials-path", $CredentialsPath,
    "--force-rebuild-latest-inference",
    "--leaderboard-rank", "1",
    "--report-name", $reportName
)

Write-Log "running trading strategy: py -3.11 $($args -join ' ')"
& py -3.11 @args 2>&1 | Tee-Object -FilePath $logPath -Append
if ($LASTEXITCODE -ne 0) {
    throw "trading strategy failed with exit code $LASTEXITCODE"
}

$manifestPath = Join-Path $reportDir "run_manifest.json"
if (-not (Test-Path $manifestPath)) {
    throw "Report manifest was not created: $manifestPath"
}

$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
if ([int]$manifest.model_count -ne 3) {
    throw "Expected model_count=3 in report manifest, got $($manifest.model_count)"
}
if (-not $manifest.latest_inference_manifest.local_data_end_date) {
    throw "Report manifest is missing latest_inference_manifest.local_data_end_date"
}

Write-Log "report validated: model_count=$($manifest.model_count) data_max_date=$($manifest.data_max_date)"

git add $reportDir
if ($LASTEXITCODE -ne 0) {
    throw "git add failed with exit code $LASTEXITCODE"
}

$staged = git diff --cached --name-only -- $reportDir
if (-not $staged) {
    Write-Log "no report changes staged; nothing to commit"
    exit 0
}

$commitMessage = "Add trading strategy report $reportName"
Write-Log "committing report: $commitMessage"
git commit -m $commitMessage 2>&1 | Tee-Object -FilePath $logPath -Append
if ($LASTEXITCODE -ne 0) {
    throw "git commit failed with exit code $LASTEXITCODE"
}

Write-Log "pushing $Branch to $Remote"
git push $Remote $Branch 2>&1 | Tee-Object -FilePath $logPath -Append
if ($LASTEXITCODE -ne 0) {
    throw "git push failed with exit code $LASTEXITCODE"
}

Write-Log "scheduled trading strategy run complete"
