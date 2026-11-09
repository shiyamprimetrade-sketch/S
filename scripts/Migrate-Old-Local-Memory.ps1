param([Parameter(Mandatory=$true)][switch]$ConfirmMigration,[string]$OldContinuumDb='')
$ErrorActionPreference = 'Stop'
if (-not $ConfirmMigration) { throw 'Add -ConfirmMigration to copy private local data.' }
$dst = Join-Path $HOME '.s-computer'
New-Item -ItemType Directory -Path $dst -Force | Out-Null
if ($OldContinuumDb) {
    if (-not (Test-Path -LiteralPath $OldContinuumDb)) { throw "Old database not found: $OldContinuumDb" }
    $new = Join-Path $dst 's.sqlite3'
    if (Test-Path -LiteralPath $new) { throw "Refusing to overwrite existing S data: $new. Export/import the old project instead." }
    py -3 -c "import sqlite3,sys; d=sqlite3.connect(sys.argv[1]); print('Old DB integrity:',d.execute('PRAGMA integrity_check').fetchone()[0]); d.close()" $OldContinuumDb
    if ($LASTEXITCODE -ne 0) { throw 'Old database cannot be read.' }
    Copy-Item -LiteralPath $OldContinuumDb -Destination $new
    Write-Host "Old Continuum database copied without overwriting: $new"
}
$oldHuman = Join-Path $HOME '.axiom-human-computer'
$newHuman = Join-Path $dst 'human'
if (Test-Path -LiteralPath $oldHuman) {
    if (Test-Path -LiteralPath $newHuman) { throw "Refusing to overwrite existing human-computer workspace: $newHuman" }
    Copy-Item -Path $oldHuman -Destination $newHuman -Recurse
    $oldDb = Join-Path $newHuman 'axiom_pc.sqlite3'
    if (Test-Path -LiteralPath $oldDb) { Rename-Item -LiteralPath $oldDb -NewName 's_pc.sqlite3' }
    Write-Host "Copied prior human-computer research into $newHuman" -ForegroundColor Green
}
Write-Host 'Migration copy complete. Original Axiom data is unchanged. Run S tests and create a backup.' -ForegroundColor Green
