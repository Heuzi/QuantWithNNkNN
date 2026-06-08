[CmdletBinding()]
param(
    [ValidateSet("eodhd", "massive")]
    [string]$Vendor = "eodhd",

    [string]$DatasetRoot = "data\eodhd_us_equities_30y",
    [string]$StartDate = "1995-01-01",
    [string]$EndDate = "",
    [string]$Tickers = "",
    [int]$MaxTickers = 25,
    [int]$FetchWorkers = 4,
    [switch]$FullUniverse,
    [switch]$FetchOnly,
    [switch]$SkipFundamentals,
    [switch]$SkipSentiment
)

$ErrorActionPreference = "Stop"

function Add-Arg {
    param(
        [System.Collections.Generic.List[string]]$Args,
        [string]$Name,
        [object]$Value
    )

    if ($null -ne $Value -and "$Value" -ne "") {
        $Args.Add($Name)
        $Args.Add("$Value")
    }
}

if ($Vendor -eq "eodhd") {
    if (-not $env:EODHD_API_KEY -and -not (Test-Path -LiteralPath "EODHD_api_key")) {
        throw "Set EODHD_API_KEY or create a local ignored EODHD_api_key file before fetching EODHD data."
    }

    $argsList = [System.Collections.Generic.List[string]]::new()
    $argsList.Add("scripts\update_eodhd_daily_dataset.py")
    Add-Arg $argsList "--dataset-root" $DatasetRoot
    Add-Arg $argsList "--start-date" $StartDate
    Add-Arg $argsList "--end-date" $EndDate
    Add-Arg $argsList "--tickers" $Tickers
    Add-Arg $argsList "--fetch-workers" $FetchWorkers

    if ($FullUniverse) {
        Add-Arg $argsList "--max-tickers" 0
    } else {
        Add-Arg $argsList "--max-tickers" $MaxTickers
    }
    if ($FetchOnly) { $argsList.Add("--fetch-only") }
    if ($SkipFundamentals) { $argsList.Add("--skip-fundamentals") }
    if ($SkipSentiment) { $argsList.Add("--skip-sentiment") }

    & py -3.11 @argsList
    exit $LASTEXITCODE
}

if (-not $env:MASSIVE_API_KEY -and -not (Test-Path -LiteralPath "MassiveApiKey")) {
    throw "Set MASSIVE_API_KEY or create a local ignored MassiveApiKey file before fetching Massive data."
}

$massiveArgs = [System.Collections.Generic.List[string]]::new()
$massiveArgs.Add("scripts\update_massive_daily_dataset.py")
Add-Arg $massiveArgs "--dataset-root" $DatasetRoot
Add-Arg $massiveArgs "--start-date" $StartDate
Add-Arg $massiveArgs "--end-date" $EndDate
Add-Arg $massiveArgs "--tickers" $Tickers

& py -3.11 @massiveArgs
exit $LASTEXITCODE
