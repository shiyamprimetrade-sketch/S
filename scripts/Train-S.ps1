param([int]$Steps=250,[string]$ProjectId='')
$ErrorActionPreference='Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
py -3 -c 'import torch; print("PyTorch",torch.__version__)'
if ($LASTEXITCODE -ne 0) { throw 'Install PyTorch: py -3 -m pip install torch' }
if ($ProjectId) { py -3 -m continuum.charllm train --corpus private --db "$HOME\.s-computer\s.sqlite3" --project $ProjectId --steps $Steps }
else { py -3 -m continuum.charllm train --steps $Steps }
if ($LASTEXITCODE -ne 0) { throw 'S model training failed' }
Write-Host 'Local checkpoint created. Evaluate unseen prompts before claiming improvement.' -ForegroundColor Green
