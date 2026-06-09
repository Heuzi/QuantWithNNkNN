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
$reportName = "latest_two_sleeve_$runDate"
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
    "scripts\run_two_sleeve_trading_strategy.ps1",
    "-DatasetRoot", $DatasetRoot,
    "-CredentialsPath", $CredentialsPath,
    "-OutputRoot", $OutputRoot,
    "-ReportName", $reportName,
    "-ForceRebuildLatestInference"
)

Write-Log "running trading strategy: powershell -ExecutionPolicy Bypass -File $($args -join ' ')"
& powershell -ExecutionPolicy Bypass -File @args 2>&1 | Tee-Object -FilePath $logPath -Append
if ($LASTEXITCODE -ne 0) {
    throw "trading strategy failed with exit code $LASTEXITCODE"
}

$manifestPath = Join-Path $reportDir "run_manifest.json"
if (-not (Test-Path $manifestPath)) {
    throw "Report manifest was not created: $manifestPath"
}

$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
if ([int]$manifest.sleeve_count -ne 2) {
    throw "Expected sleeve_count=2 in combined report manifest, got $($manifest.sleeve_count)"
}
if ([int]$manifest.model_count -lt 6) {
    throw "Expected at least 6 scored models in combined report manifest, got $($manifest.model_count)"
}

Write-Log "report validated: sleeve_count=$($manifest.sleeve_count) model_count=$($manifest.model_count)"

$sourceDirs = @(
    $reportDir,
    (Join-Path $OutputRoot "${reportName}_conservative"),
    (Join-Path $OutputRoot "${reportName}_momentum_breakout")
)
git add @sourceDirs
if ($LASTEXITCODE -ne 0) {
    throw "git add failed with exit code $LASTEXITCODE"
}

$staged = git diff --cached --name-only -- @sourceDirs
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
