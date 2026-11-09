# Run from the extracted S v0.5 source on your own Windows PC.
# Intentionally does not terminate applications or overwrite private database.
$ErrorActionPreference = 'Stop'
$source = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Push-Location $source
try {
  if (-not (Get-Command py -ErrorAction SilentlyContinue)) { throw 'Python launcher missing. Install Python 3.11+.' }
  Write-Host '1/4 Checking source and running regression tests' -ForegroundColor Cyan
  & py -3 -m unittest discover -s tests -q
  if ($LASTEXITCODE -ne 0) { throw 'Regression tests failed. Existing S installation was not changed.' }
  Write-Host '2/4 Backing up private S database' -ForegroundColor Cyan
  $backupDir = Join-Path $HOME 'Documents\S-backups'
  New-Item -ItemType Directory -Force -Path $backupDir | Out-Null
  $backupPath = Join-Path $backupDir ('S-before-v0.5-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.sqlite3')
  & py -3 -m continuum.cli backup --output $backupPath
  if ($LASTEXITCODE -ne 0) { throw 'Backup failed. Existing S installation was not changed.' }
  Write-Host '3/4 Upgrading installed program files without touching private memory' -ForegroundColor Cyan
  & (Join-Path $PSScriptRoot 'Install-S.ps1')
  if (-not $?) { throw 'Installer failed; backups are preserved.' }
  Write-Host '4/4 Upgrade complete; restart S from your desktop shortcut.' -ForegroundColor Green
  Write-Host ('Backup: ' + $backupPath) -ForegroundColor Green
} finally { Pop-Location }