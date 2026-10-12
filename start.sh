#!/usr/bin/env bash
# Rockstone CT Workbench - one-click launcher (macOS / Linux / Git Bash).
# Starts the server on 127.0.0.1:8770 (unless already running) and opens
# http://127.0.0.1:8770 in the default browser.
# Usage: ./start.sh [port]
set -u
cd "$(dirname "$0")"

PORT="${1:-8770}"
URL="http://127.0.0.1:$PORT"

# --- pick a Python interpreter: project venv first, then PATH ------------
PY=""
if [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
elif [ -x .venv/Scripts/python.exe ]; then      # Git Bash on Windows
  PY=.venv/Scripts/python.exe
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "ERROR: Python not found and no .venv exists." >&2
  echo "Install Python 3.11+, then run:" >&2
  echo "    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

if ! "$PY" -c "import uvicorn" >/dev/null 2>&1; then
  echo "ERROR: dependencies are not installed for $PY." >&2
  echo "Run:  $PY -m pip install -r requirements.txt" >&2
  exit 1
fi

healthy() {
  "$PY" -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('$URL/api/health',timeout=2).status==200 else 1)" >/dev/null 2>&1
}

SRV_PID=""
STARTED=0
if healthy; then
  echo "Server already running at $URL"
else
  echo "Starting Rockstone CT Workbench server on $URL ..."
  "$PY" -m uvicorn server.app:app --host 127.0.0.1 --port "$PORT" &
  SRV_PID=$!
  STARTED=1
  trap 'kill "$SRV_PID" 2>/dev/null' INT TERM EXIT
  i=0
  while ! healthy; do
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
      echo "ERROR: server exited early - see output above." >&2
      exit 1
    fi
    i=$((i + 1))
    if [ "$i" -ge 60 ]; then
      echo "ERROR: server did not become ready within 60 seconds." >&2
      exit 1
    fi
    sleep 1
  done
  echo "Server ready."
fi

# --- open the default browser --------------------------------------------
if command -v open >/dev/null 2>&1; then
  open "$URL"
elif command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$URL" >/dev/null 2>&1 || true
elif command -v cmd.exe >/dev/null 2>&1; then     # Git Bash on Windows
  # use //c so MSYS does not rewrite /c as a path
  cmd.exe //c start "$URL" >/dev/null 2>&1 || true
else
  echo "Open this URL in your browser: $URL"
fi

# --- keep the server in the foreground when we started it ----------------
if [ "$STARTED" -eq 1 ]; then
  echo "Server running - press Ctrl+C to stop."
  wait "$SRV_PID"
fi
