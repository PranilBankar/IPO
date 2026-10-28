# Start IPOPilot AI Streamlit Web App
$ErrorActionPreference = "Stop"

Write-Host "🚀 Launching IPOPilot AI on Streamlit..." -ForegroundColor Cyan

# Set working directory to project root
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
Set-Location $projectRoot

# Ensure python environment is used
python -m streamlit run streamlit_app.py
