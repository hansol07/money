$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

# Some Codex/sandbox shells inject a dead proxy (127.0.0.1:9). If Streamlit
# inherits it, all market-data requests are routed into a closed local port.
foreach ($name in @("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")) {
    $value = [Environment]::GetEnvironmentVariable($name, "Process")
    if ($value -match "127\.0\.0\.1:9") {
        [Environment]::SetEnvironmentVariable($name, $null, "Process")
    }
}

$python = "python"
if (Test-Path "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe") {
    $python = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
}

& $python -m streamlit run app.py --server.port=8501 --browser.gatherUsageStats=false
