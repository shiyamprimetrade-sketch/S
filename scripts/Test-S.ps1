$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
py -3 -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw 'S local automated tests failed' }
py -3 -m continuum.cli doctor
