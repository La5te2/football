#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_DIR="$ROOT/runs"
DATA="$RUN_DIR/tamakeri.h5"
CHECKPOINT="$RUN_DIR/jackaroo.pt"
PYTHON_BIN="${PYTHON:-python}"

cd "$ROOT"
mkdir -p "$RUN_DIR"

if [[ ! -f "$CHECKPOINT" ]]; then
  if [[ ! -f "$DATA" ]]; then
    "$PYTHON_BIN" -u -m training.jackaroo.prepare \
      --games 256 \
      --flight 16 \
      --maximum-steps 3000 \
      --sequence-length 32 \
      --device auto \
      --seed 1 \
      --output "$DATA"
  fi
  "$PYTHON_BIN" -u -m training.jackaroo.pretrain \
    --data "$DATA" \
    --checkpoint "$CHECKPOINT" \
    --epochs 2 \
    --batch-size 4096 \
    --maximum-steps 3000 \
    --device auto \
    --seed 1
fi

exec "$PYTHON_BIN" -u -m training.jackaroo.train \
  --device auto \
  --updates 3000 \
  --games-per-update 256 \
  --sequence-length 32 \
  --burn-in 32 \
  --ppo-epochs 4 \
  --structure-epochs 1 \
  --ppo-batch-size 2048 \
  --value-clip 0.2 \
  --learning-rate 0.0001 \
  --entropy-coefficient 0.001 \
  --space-coefficient 0.05 \
  --flight 16 \
  --maximum-steps 3000 \
  --validation-seed 42 \
  --validation-directory "$RUN_DIR/validation" \
  --seed 1 \
  --resume "$CHECKPOINT" \
  --checkpoint "$CHECKPOINT" \
  "$@"
