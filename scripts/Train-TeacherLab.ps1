$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
Set-Location $root
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
  throw 'Python 3.11+ required. Install Python from Windows before continuing.'
}
py -3 -m continuum.lab_cli build --simulations 4000 --seed 20261109
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
py -3 -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host 'TeacherLab training and unit tests completed; inspect reports/teacher_student_evaluation.json' -ForegroundColor Green
