$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
$pythonCommand = Get-Command py -ErrorAction SilentlyContinue
if (-not (Test-Path ".venv/Scripts/python.exe")) {
    if ($pythonCommand) { & py -3.11 -m venv .venv } else { & python -m venv .venv }
}
$python = Join-Path $repoRoot ".venv/Scripts/python.exe"
if (-not (Test-Path $python)) { throw "Could not find virtual-environment Python at $python" }
$pythonVersion = & $python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ([version]$pythonVersion -lt [version]"3.11") { throw "Python 3.11 or newer is required; found $pythonVersion." }
& $python -m pip install -e ".[dev]"
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
Write-Host "Capture Studio: http://127.0.0.1:8000/"
Write-Host "API docs:       http://127.0.0.1:8000/docs"
Write-Host "Press Ctrl+C to stop."
& $python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
