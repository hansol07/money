$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$python = "python"
if (Test-Path "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe") {
    $python = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
}

& $python -m streamlit run app.py --server.port=8501 --browser.gatherUsageStats=false

