#!/usr/bin/env bash
set -euo pipefail

TRAINING_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$TRAINING_DIR/.." && pwd)"
BUILD_DIR="$ROOT/.local/build/training-linux"

cmake -S "$TRAINING_DIR" -B "$BUILD_DIR"
cmake --build "$BUILD_DIR" --parallel --target _gfootball_env
