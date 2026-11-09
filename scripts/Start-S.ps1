$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
Write-Host 'S — Local Human Computer' -ForegroundColor Green
Write-Host 'Open http://127.0.0.1:8765 in your browser; Ctrl+C to stop.'
py -3 -m continuum.cli serve
