param([string]$Destination = '')
$ErrorActionPreference = 'Stop'
$source = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $Destination) {
  $profileInstall = Join-Path $env:LOCALAPPDATA 'S-Computer'
  if (Test-Path -LiteralPath $profileInstall) { $Destination = $profileInstall }
  elseif (Test-Path -LiteralPath 'C:\S') { $Destination = 'C:\S' }
  else { $Destination = $profileInstall }
}
$backupDir = Join-Path $HOME 'Documents\S-backups'
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$programBackup = Join-Path $backupDir ('S-program-before-v0.7-' + $stamp)
$dbBackup = Join-Path $backupDir ('S-memory-before-v0.7-' + $stamp + '.sqlite3')
$database = Join-Path $HOME '.s-computer\s.sqlite3'
Push-Location $source
try {
  if (-not (Get-Command py -ErrorAction SilentlyContinue)) { throw 'Python launcher not found; install Python 3.11+ first.' }
  Write-Host '1/5: Verify v0.7 package before changing installed S' -ForegroundColor Cyan
  & py -3 -m unittest discover -s tests -q
  if ($LASTEXITCODE -ne 0) { throw 'Regression tests failed; installed S was not changed.' }
  & py -3 -m examples.evaluate_meaning | Out-Null
  if ($LASTEXITCODE -ne 0) { throw 'Meaning evaluation failed.' }
  $tcp = [System.Net.Sockets.TcpClient]::new()
  try {
    $connect = $tcp.BeginConnect('127.0.0.1',8765,$null,$null)
    if ($connect.AsyncWaitHandle.WaitOne(400,$false)) {
      try { $tcp.EndConnect($connect); throw 'S is running on port 8765. Stop it before upgrading.' }
      catch [System.Net.Sockets.SocketException] { }
    }
  } finally { $tcp.Close() }
  Write-Host '2/5: Back up local memory before installation' -ForegroundColor Cyan
  New-Item -ItemType Directory -Force -Path $backupDir | Out-Null
  if (Test-Path -LiteralPath $database) {
    & py -3 -m continuum.cli backup --output $dbBackup
    if ($LASTEXITCODE -ne 0) { throw 'Database backup failed. Installed program not changed.' }
  } else {
    Write-Host 'No existing default S database was found; no private backup needed.' -ForegroundColor Yellow
  }
  Write-Host '3/5: Preserve installed application files for rollback' -ForegroundColor Cyan
  if (Test-Path -LiteralPath $Destination) {
    Copy-Item -LiteralPath $Destination -Destination $programBackup -Recurse -Force
  }
  Write-Host '4/5: Update program files; keep private database in user profile' -ForegroundColor Cyan
  if ([System.IO.Path]::GetFullPath($source).TrimEnd('\') -ne [System.IO.Path]::GetFullPath($Destination).TrimEnd('\')) {
    try {
      & (Join-Path $PSScriptRoot 'Install-S.ps1') -Destination $Destination
      if (-not $?) { throw 'Installer reported failure' }
    } catch {
      if (Test-Path -LiteralPath $programBackup) {
        Copy-Item -Path (Join-Path $programBackup '*') -Destination $Destination -Force -Recurse
      }
      throw 'Program update failed. Previous files were restored where possible; backup preserved. ' + $_.Exception.Message
    }
  } else {
    Write-Host 'Package is already located in destination. Skipping self-copy.' -ForegroundColor Yellow
  }
  Write-Host '5/5: Upgrade files copied, restart S and verify local browser' -ForegroundColor Green
  Write-Host ('Installation: ' + $Destination)
  if (Test-Path -LiteralPath $dbBackup) { Write-Host ('Database backup: ' + $dbBackup) }
  if (Test-Path -LiteralPath $programBackup) { Write-Host ('Program backup: ' + $programBackup) }
  Write-Host 'This script did not modify the operating system or start an external AI service.'
} finally { Pop-Location }
