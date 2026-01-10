#!/usr/bin/env bash
# Build the single intersection network from plain XML files
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
SCENARIO_DIR="$REPO_ROOT/scenarios/single_int"

netconvert \
    --node-files="$SCENARIO_DIR/plain/nodes.nod.xml" \
    --edge-files="$SCENARIO_DIR/plain/edges.edg.xml" \
    --output-file="$SCENARIO_DIR/net.net.xml" \
    --tls.guess \
    --tls.default-type=static

echo "Network built: $SCENARIO_DIR/net.net.xml"
