$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$python = "python"
if (Test-Path "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe") {
    $python = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
}

$script = @'
import sqlite3
from pathlib import Path

db_path = Path("data/cache.sqlite3")
if not db_path.exists():
    raise SystemExit("data/cache.sqlite3 not found")

with sqlite3.connect(db_path) as conn:
    result = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
    conn.execute("VACUUM")

print(f"SQLite prepared: {db_path} ({db_path.stat().st_size:,} bytes)")
print(f"WAL checkpoint: {result}")
'@

$script | & $python -

Write-Host "Data is ready for Git/LFS. Next:"
Write-Host "  git add .gitattributes .gitignore app.py src data/cache.sqlite3 data/*.json data/universe_*.csv requirements.txt COMPANY_SETUP_KO.md scripts"
Write-Host "  git commit -m `"Sync stock helper app and local learning database`""
Write-Host "  git push"
