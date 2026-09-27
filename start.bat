@echo off
rem Launch wordsearch-bot on Windows (uv creates the venv on first run).
cd /d "%~dp0"
where uv >nul 2>nul || (
  echo uv is not installed. Install it with:
  echo   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  exit /b 1
)
uv run wsbot %*
