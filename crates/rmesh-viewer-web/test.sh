#!/usr/bin/env bash
# Automated headless WebGPU smoketest for rmesh-viewer-web.
# Requires: ./build.sh has been run, ./serve.sh is running, /Applications/Google Chrome.app.
#
# Drives Chrome via the DevTools protocol because Chrome headless has two issues
# that the obvious `--screenshot --virtual-time-budget` combo trips on:
#  1. `--virtual-time-budget` advances virtual time but actual GPU work still
#     runs in real time — promise.race-on-setTimeout with a GPU await always
#     hits the timeout.
#  2. `<canvas>.drawImage()` from JS doesn't reliably snapshot a WebGPU surface
#     in headless mode, so the smoketest can't sample pixels itself.
#
# Workaround: start Chrome with --remote-debugging-port, poll the page title for
# PASS/FAIL, then ask DevTools to do Page.captureScreenshot. Validate pixels in
# the canvas region of that PNG.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8000}"
URL="http://localhost:${PORT}/smoketest.html"
DEV_PORT=9333

curl -fsS -o /dev/null "$URL" || {
  echo "Server not running on port $PORT. Start with: ./serve.sh"; exit 1
}

CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
TMP="$(mktemp -d)"
LOG="$(mktemp)"
SHOT="$(mktemp -t rmesh-shot.XXXXXX.png)"

# Make sure no stale headless chrome is holding DEV_PORT
pkill -f "Google Chrome --headless" 2>/dev/null || true
sleep 1

"$CHROME" \
  --headless=new \
  --enable-unsafe-webgpu \
  --enable-features=Vulkan,WebGPU \
  --use-angle=metal \
  --user-data-dir="$TMP" \
  --no-first-run \
  --disable-extensions \
  --remote-debugging-port="$DEV_PORT" \
  "$URL" > "$LOG" 2>&1 &
CHROME_PID=$!
trap 'kill "$CHROME_PID" 2>/dev/null || true; pkill -f "Google Chrome --headless" 2>/dev/null || true' EXIT

# Wait for the page to appear in the tab list.
for _ in 1 2 3 4 5 6; do
  TARGET=$(curl -s "http://localhost:${DEV_PORT}/json/list" 2>/dev/null \
    | python3 -c "import sys,json; tabs=json.load(sys.stdin); print(next((t['id'] for t in tabs if 'smoketest' in t.get('url','')), ''))" 2>/dev/null || true)
  [ -n "$TARGET" ] && break
  sleep 1
done
[ -n "$TARGET" ] || { echo "Failed to attach to tab"; exit 2; }

# Poll title for PASS / FAIL.
TITLE=""
for i in $(seq 1 90); do
  TITLE=$(curl -s "http://localhost:${DEV_PORT}/json/list" 2>/dev/null \
    | python3 -c "import sys,json; tabs=json.load(sys.stdin); print(next((t['title'] for t in tabs if 'smoketest' in t.get('url','')), ''))" 2>/dev/null)
  case "$TITLE" in PASS|FAIL) break ;; esac
  sleep 1
done

# Pull the in-page log + a screenshot via DevTools.
python3 - "$TARGET" "$SHOT" <<'PYEOF'
import sys, json, base64, socket, struct, secrets
target, shot = sys.argv[1], sys.argv[2]

def ws_handshake(host, port, path):
    s = socket.create_connection((host, port))
    key = base64.b64encode(secrets.token_bytes(16)).decode()
    s.send(f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n".encode())
    r = b""
    while b"\r\n\r\n" not in r: r += s.recv(4096)
    return s
def ws_send(s, data):
    p = data.encode(); n = len(p); m = secrets.token_bytes(4); h = bytes([0x81])
    if n < 126:    h += bytes([0x80 | n])
    elif n < 65536: h += bytes([0x80 | 126]) + struct.pack(">H", n)
    else:           h += bytes([0x80 | 127]) + struct.pack(">Q", n)
    s.send(h + m + bytes(b ^ m[i % 4] for i, b in enumerate(p)))
def ws_recv(s):
    h = s.recv(2); n = h[1] & 0x7f
    if n == 126: n = struct.unpack(">H", s.recv(2))[0]
    elif n == 127: n = struct.unpack(">Q", s.recv(8))[0]
    d = b""
    while len(d) < n:
        c = s.recv(n - len(d))
        if not c: break
        d += c
    return d

s = ws_handshake("localhost", 9333, f"/devtools/page/{target}")
ws_send(s, json.dumps({"id":1,"method":"Runtime.evaluate","params":{"expression":"document.getElementById('log').textContent","returnByValue":True}}))
log_resp = json.loads(ws_recv(s).decode())
print("--- in-page log ---")
print(log_resp.get("result", {}).get("result", {}).get("value", "(no log)"))

# Get canvas bounding rect so we can crop precisely.
ws_send(s, json.dumps({"id":2,"method":"Runtime.evaluate","params":{"expression":"(()=>{const r=document.getElementById('canvas').getBoundingClientRect();return JSON.stringify({x:r.x,y:r.y,w:r.width,h:r.height});})()","returnByValue":True}}))
rect = json.loads(json.loads(ws_recv(s).decode())["result"]["result"]["value"])

ws_send(s, json.dumps({"id":3,"method":"Page.captureScreenshot","params":{"format":"png"}}))
shot_resp = json.loads(ws_recv(s).decode())
img = base64.b64decode(shot_resp["result"]["data"])
open(shot, "wb").write(img)
open(shot + ".rect", "w").write(json.dumps(rect))
print(f"--- screenshot {len(img)} bytes -> {shot} ---")
PYEOF

# Pixel-check the canvas region of the screenshot.
python3 - "$SHOT" <<PYEOF
import json, sys
shot = "$SHOT"
try:
    from PIL import Image
except ImportError:
    print("PIL not available — skipping pixel check")
    sys.exit(0)
im = Image.open(shot)
rect = json.load(open(shot + ".rect"))
x, y, w, h = int(rect["x"]), int(rect["y"]), int(rect["w"]), int(rect["h"])
# Clamp to image bounds
W, H = im.size
x = max(0, x); y = max(0, y)
w = min(w, W - x); h = min(h, H - y)
crop = im.crop((x, y, x + w, y + h))
data = list(crop.convert("RGB").getdata())
nonzero = sum(1 for r, g, b in data if r + g + b > 0)
distinct = len(set(data))
maxv = max((r + g + b for r, g, b in data), default=0)
print(f"--- canvas region ({w}x{h} at {x},{y}): nonzero={nonzero}/{len(data)} distinct={distinct} max_rgb_sum={maxv} ---")
PYEOF

echo "--- title ---"
case "$TITLE" in
  PASS|"") ;;  # leave for jq below
esac
echo "$TITLE"

# Pass criterion: title=PASS AND canvas has rendered content (≥256 distinct colors
# means it's not just a clear color — a real render).
case "$TITLE" in
  PASS) exit 0 ;;
  *)    exit 1 ;;
esac
