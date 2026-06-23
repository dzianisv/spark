#!/bin/sh
# Build the spark+Aurora unified sandbox image.
# Run once; rebuilding is only needed after Aurora or Dockerfile updates.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
docker build -t spark-local "$SCRIPT_DIR"

echo ""
echo "Built spark-local. Usage:"
echo "  # Run with ChatGPT backend (Aurora starts automatically):"
echo "  CHATGPT_ACCESS_TOKEN=eyJ... bun spark.ts --sandbox"
echo ""
echo "  # Run with Ollama (existing behavior, no Aurora):"
echo "  bun spark.ts --sandbox"
