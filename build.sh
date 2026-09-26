#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$ROOT/.local/build/native-linux"
RUNTIME_DIR="$ROOT/bin"

cmake -S "$ROOT" -B "$BUILD_DIR"
cmake --build "$BUILD_DIR" --parallel

mkdir -p "$RUNTIME_DIR" "$ROOT/models/tamakeri"
cp "$BUILD_DIR/runtime/gfootball" "$RUNTIME_DIR/"
cp "$BUILD_DIR/runtime/replay" "$RUNTIME_DIR/"
cp "$BUILD_DIR/runtime"/libtamakeri.* "$ROOT/models/tamakeri/"
cp -R "$ROOT/engine/data" "$RUNTIME_DIR/"
cp -R "$ROOT/engine/fonts" "$RUNTIME_DIR/"
echo "Native runtime created in $RUNTIME_DIR"
