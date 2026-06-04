#!/usr/bin/env bash
# Serve the built web viewer over HTTP (WebGPU requires a secure context;
# localhost counts as secure).
set -euo pipefail
cd "$(dirname "$0")/web"
PORT="${PORT:-8000}"
echo "Serving $(pwd) on http://localhost:$PORT/"
echo "Open in:"
echo "  - Chrome 113+ / Edge 113+ (WebGPU on by default)"
echo "  - Safari 18+ (macOS Sonoma+ — WebGPU on by default)"
echo "  - Safari Technology Preview / Firefox Nightly (with WebGPU flag)"
exec python3 -m http.server "$PORT"
