#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python download_models.py --profile all
.venv/bin/python verify_models.py
echo "Start: .venv/bin/python run_laptop.py --profile balanced"
