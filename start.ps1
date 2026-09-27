# Launch wordsearch-bot from PowerShell (uv creates the venv on first run).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv is not installed. Install it with:"
    Write-Host '  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
    exit 1
}
uv run wsbot @args
