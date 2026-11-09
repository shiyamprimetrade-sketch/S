$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
py -3 -m unittest discover -s tests -p 'test_human_computer.py' -v