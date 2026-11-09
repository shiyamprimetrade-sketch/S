param(
  [Parameter(Mandatory=$true)][string]$ExportFile,
  [string]$Title = "",
  [switch]$KeepExposedSecrets
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path -LiteralPath $ExportFile)) { throw "ChatGPT export not found: $ExportFile" }
if ($KeepExposedSecrets) {
  Write-Warning "Exposed tokens will be retained in your PRIVATE local SQLite archive. Never commit data/."
  & py -3 -m continuum.cli import-chatgpt $ExportFile --title $Title --unredacted
} else {
  & py -3 -m continuum.cli import-chatgpt $ExportFile --title $Title
}
if ($LASTEXITCODE -ne 0) { throw "ChatGPT import failed with code $LASTEXITCODE" }
