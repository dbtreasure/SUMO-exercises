#!/bin/bash
set -e

CONFIG_PATH="${1:?Usage: $0 <config_path>}"

echo "========================================"
echo "Running ablation: $CONFIG_PATH"
echo "========================================"

cd "$(dirname "$0")/.."
uv run python main.py --config "$CONFIG_PATH"
