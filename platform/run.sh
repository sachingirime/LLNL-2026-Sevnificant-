#!/usr/bin/env bash
# Launches the Lattice NDE Platform locally.
# Usage: ./run.sh   (from anywhere; resolves its own directory)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment in platform/.venv ..."
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -r requirements.txt

echo ""
echo "Starting Lattice NDE Platform at http://127.0.0.1:8420"
echo "(Ctrl+C to stop)"
python backend/main.py
