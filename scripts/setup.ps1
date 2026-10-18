# One-time setup: install dependencies, ingest all data, train models, run backtest.
# Usage (PowerShell, from repo root):  .\scripts\setup.ps1          (add -Quick to skip long backfills)
param([switch]$Quick)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location "$root\backend"
python -m pip install -r requirements.txt
if ($Quick) { python -m ipopilot.cli bootstrap --quick } else { python -m ipopilot.cli bootstrap }
python -m ipopilot.cli news
python -m ipopilot.cli train
python -m ipopilot.cli backtest
Set-Location "$root\frontend"
npm install
Write-Host "Setup complete. Start everything with .\scripts\start.ps1" -ForegroundColor Green
