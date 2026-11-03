#!/usr/bin/env bash
# Serve the built web viewer over HTTP (WebGPU requires a secure context;
# localhost counts as secure).
set -euo pipefail
cd "$(dirname "$0")/web"
PORT="${PORT:-8000}"
echo "Serving $(pwd) on http://localhost:$PORT/"
echo "Use a browser with WebGPU enabled (tested with Chromium on Metal)."
exec python3 -m http.server "$PORT" --bind 127.0.0.1
