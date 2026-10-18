# Start the API (port 8000) and the web app (port 3000) in two windows.
# Usage: .\scripts\start.ps1            (add -Prod to run an optimized production build)
param([switch]$Prod)
$root = Split-Path -Parent $PSScriptRoot
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$root\backend'; python -m ipopilot.cli serve --host 127.0.0.1 --port 8000"
if ($Prod) {
  Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$root\frontend'; npm run build; npm run start"
} else {
  Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$root\frontend'; npm run dev"
}
Start-Sleep -Seconds 6
Start-Process "http://localhost:3000"
