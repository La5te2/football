#!/usr/bin/env bash
set -euo pipefail

JACKAROO_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$JACKAROO_DIR/../.." && pwd)"
BUILD_DIR="$ROOT/.local/build/jackaroo-linux"

cmake -S "$JACKAROO_DIR" -B "$BUILD_DIR"
cmake --build "$BUILD_DIR" --parallel --target _gfootball_env
