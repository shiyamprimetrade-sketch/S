# Safe local-first launcher. Run from PowerShell: powershell -ExecutionPolicy Bypass -File .\scripts\Start-Continuum.ps1
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    Write-Host 'Python launcher missing. Install Python 3.11+ (e.g. winget install -e --id Python.Python.3.12) and reopen PowerShell.' -ForegroundColor Yellow
    exit 2
}
Write-Host 'Checking local model engine...' -ForegroundColor Cyan
try {
    $models=Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/tags' -TimeoutSec 4
    $names=@($models.models | ForEach-Object { $_.name })
    if ($names.Count -eq 0) { Write-Warning 'Ollama is running but no models installed. Try: ollama pull qwen2.5:1.5b' }
    else { Write-Host ('Installed models: ' + ($names -join ', ')) }
} catch {
    Write-Warning 'Ollama not running. Install Ollama, start it, and run: ollama pull qwen2.5:1.5b'
    Write-Host 'The web UI will still open, and existing conversations remain accessible.'
}
Write-Host 'Starting AxiomFlow on 127.0.0.1:8765 ...' -ForegroundColor Green
Start-Process 'http://127.0.0.1:8765'
py -3 -m continuum.cli serve --port 8765
