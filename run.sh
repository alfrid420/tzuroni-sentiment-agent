#!/usr/bin/env bash
# Run the TZURONI sentiment pipeline. Creates the venv on first run.
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi
.venv/bin/python main.py
