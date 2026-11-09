$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
Set-Location $root
py -3 -m unittest discover -s tests -v
if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}
py -3 -m continuum.cli doctor
