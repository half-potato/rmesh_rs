#!/usr/bin/env bash
# Build the wasm viewer and generate JS bindings.
# Usage: ./build.sh [--release | --debug]   (default: --release)
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"

MODE="${1:---release}"
case "$MODE" in
  --release) PROFILE=release; CARGO_FLAG=--release ;;
  --debug)   PROFILE=debug;   CARGO_FLAG= ;;
  *) echo "usage: $0 [--release|--debug]"; exit 1 ;;
esac

# Use the rustup toolchain so wasm32-unknown-unknown is available
# (Homebrew rustc on PATH doesn't ship cross targets).
if [[ -x "$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin/cargo" ]]; then
  export PATH="$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin:$HOME/.cargo/bin:$PATH"
else
  export PATH="$HOME/.cargo/bin:$PATH"
fi

command -v wasm-bindgen >/dev/null || {
  echo "wasm-bindgen-cli not found. Install with:"
  echo "  cargo install wasm-bindgen-cli --version 0.2.113 --locked"
  exit 1
}

echo "==> cargo build $CARGO_FLAG --target wasm32-unknown-unknown"
(cd "$ROOT" && cargo build -p rmesh-viewer-web --target wasm32-unknown-unknown $CARGO_FLAG)

WASM="$ROOT/target/wasm32-unknown-unknown/$PROFILE/rmesh_viewer_web.wasm"
OUT="$(pwd)/web/pkg"
mkdir -p "$OUT"

echo "==> wasm-bindgen --target web -> $OUT"
wasm-bindgen --target web --no-typescript --out-dir "$OUT" "$WASM"

echo
echo "Built. To serve:"
echo "  ./serve.sh"
echo "Then open http://localhost:8000/ in Chrome or Safari (WebGPU enabled)."
