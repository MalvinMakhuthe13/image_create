#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ ! -x .venv/bin/python ]]; then python3.11 -m venv .venv; fi
.venv/bin/python -m pip install -e ".[dev]"
echo "Capture Studio: http://127.0.0.1:8000/"
echo "API docs:       http://127.0.0.1:8000/docs"
exec .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
