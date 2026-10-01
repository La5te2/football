#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_DIR="$ROOT/runs"
CHECKPOINT="$RUN_DIR/jackaroo.pt"
LOG="$RUN_DIR/jackaroo.stdout.log"
PID_FILE="$RUN_DIR/jackaroo.pid"
PYTHON_BIN="${PYTHON:-python}"

if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(<"$PID_FILE")"
  if kill -0 "$existing_pid" 2>/dev/null; then
    echo "Jackaroo is already running with PID $existing_pid."
    exit 1
  fi
  rm -f "$PID_FILE"
fi

if ! command -v g++ >/dev/null 2>&1; then
  echo "g++ is required to select the system C++ runtime."
  exit 1
fi

SYSTEM_LIBSTDCPP="$(g++ -print-file-name=libstdc++.so.6)"
if [[ ! -f "$SYSTEM_LIBSTDCPP" ]]; then
  echo "The system libstdc++.so.6 could not be found."
  exit 1
fi

PRELOAD="$SYSTEM_LIBSTDCPP${LD_PRELOAD:+:$LD_PRELOAD}"
cd "$ROOT"
if ! env LD_PRELOAD="$PRELOAD" "$PYTHON_BIN" -c \
  "import torch; from training.jackaroo.env import SelfPlayVectorFootballEnv; e = SelfPlayVectorFootballEnv(1, 1, 1); e.reset(); e.step([[0, 0]], [False])"; then
  echo "Torch or the current native vector environment could not be loaded."
  echo "Build it first with: bash training/jackaroo/build.sh"
  exit 1
fi

mkdir -p "$RUN_DIR"

nohup env LD_PRELOAD="$PRELOAD" "$PYTHON_BIN" -u \
  -m training.jackaroo.train \
  --device auto \
  --updates 3000 \
  --pretraining-games 256 \
  --pretraining-epochs 2 \
  --games-per-update 256 \
  --sequence-length 32 \
  --ppo-epochs 4 \
  --ppo-batch-size 256 \
  --learning-rate 0.0001 \
  --entropy-coefficient 0.001 \
  --space-coefficient 0.05 \
  --flight 16 \
  --maximum-steps 3000 \
  --validation-seed 42 \
  --validation-directory "$RUN_DIR/validation" \
  --seed 1 \
  --checkpoint "$CHECKPOINT" \
  "$@" \
  > "$LOG" 2>&1 &

training_pid=$!
echo "$training_pid" > "$PID_FILE"
echo "Jackaroo started with PID $training_pid using a 16-match flight."
echo "Checkpoint: $CHECKPOINT"
echo "Log: $LOG"
echo "Follow the log with: tail -f runs/jackaroo.stdout.log"
