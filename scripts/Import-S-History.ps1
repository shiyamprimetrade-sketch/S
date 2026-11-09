param([Parameter(Mandatory=$true)][string]$ExportZip,[string]$Title='')
$ErrorActionPreference='Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path -LiteralPath $ExportZip)) { throw "Export not found: $ExportZip" }
if ($Title) {py -3 -m continuum.cli import-chatgpt $ExportZip --title $Title}
else {py -3 -m continuum.cli import-chatgpt $ExportZip}
if ($LASTEXITCODE -ne 0) { throw 'Import did not finish' }
Write-Host 'Imported locally. Back up your S database and keep the ZIP private.' -ForegroundColor Green
