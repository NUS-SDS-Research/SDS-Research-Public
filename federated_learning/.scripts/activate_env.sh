#!/usr/bin/env bash
# FLaas — create (if needed) and activate the project virtual environment

VENV_DIR="$(git rev-parse --show-toplevel)/federated_learning/.venv"

[ ! -d "$VENV_DIR" ] && python3 -m venv "$VENV_DIR" && pip install -r "$(git rev-parse --show-toplevel)/federated_learning/requirements.txt" -q

source "$VENV_DIR/bin/activate"
echo "[FLaas] venv active → $(python --version)"
