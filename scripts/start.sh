#!/usr/bin/env bash
# Start API (:8000) and web app (:3000). Ctrl+C stops both.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
(cd "$ROOT/backend" && python -m ipopilot.cli serve --host 127.0.0.1 --port 8000) &
API=$!
trap 'kill $API' EXIT
cd "$ROOT/frontend" && npm run dev
