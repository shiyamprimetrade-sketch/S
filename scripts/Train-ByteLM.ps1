param(
  [ValidateRange(1,20000)][int]$Steps = 250,
  [ValidateSet("synthetic", "private")][string]$Corpus = "synthetic",
  [string]$ProjectId = ""
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
& py -3 -c "import torch; print('PyTorch',torch.__version__)"
if ($LASTEXITCODE -ne 0) { throw "PyTorch is missing; run: py -3 -m pip install torch" }
if ($Corpus -eq "private") {
  if (-not $ProjectId) {throw "Private training requires a ProjectId from /api/projects"}
  & py -3 -m continuum.charllm train --corpus private --db .\data\continuum.sqlite3 --project $ProjectId --steps $Steps
} else {
  & py -3 -m continuum.charllm train --steps $Steps
}
if ($LASTEXITCODE -ne 0) { throw "Training failed with code $LASTEXITCODE" }
