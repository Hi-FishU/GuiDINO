#!/usr/bin/env bash

# Queue runner for MedToken run scripts.
# Edit RUNFILES below to choose which scripts to execute, in order.
# Usage:
#   bash run_batch_queue.sh
# Optional env vars:
#   STOP_ON_ERROR=1      # stop queue when a script fails (default: 0)
#   LOG_DIR=logs/queue   # directory for log files

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/queue}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"

mkdir -p "$LOG_DIR"

# ------------------------------------------------------------
# Edit this list to define your queue.
# Paths can be relative to this file's directory.
# ------------------------------------------------------------
RUNFILES=(
  "run_guidennwnet_TN3K.sh"
  "run_guidennwnet_ISIC.sh"
  "run_guidino_ISIC.sh"
  "run_guidedino_lora_ISIC.sh"
  "run_guidino_TN3K.sh"
  "run_guidennwnet_lora_TN3K.sh"
)

if [ "${#RUNFILES[@]}" -eq 0 ]; then
  echo "[ERROR] RUNFILES is empty. Edit run_batch_queue.sh and add scripts to RUNFILES."
  exit 1
fi

echo "[INFO] Root dir      : $ROOT_DIR"
echo "[INFO] Log dir       : $LOG_DIR"
echo "[INFO] Stop on error : $STOP_ON_ERROR"
echo "[INFO] Queue size    : ${#RUNFILES[@]}"

total=${#RUNFILES[@]}
passed=0
failed=0

for i in "${!RUNFILES[@]}"; do
  runfile="${RUNFILES[$i]}"
  runpath="$ROOT_DIR/$runfile"
  name="$(basename "$runfile" .sh)"
  logfile="$LOG_DIR/${TIMESTAMP}_$((i + 1))_${name}.log"

  echo ""
  echo "========== [$((i + 1))/$total] START: $runfile =========="

  if [ ! -f "$runpath" ]; then
    echo "[FAIL] File not found: $runpath"
    failed=$((failed + 1))
    if [ "$STOP_ON_ERROR" = "1" ]; then
      break
    fi
    continue
  fi

  (
    cd "$ROOT_DIR" || exit 1
    bash "$runpath"
  ) 2>&1 | tee "$logfile"

  exit_code=${PIPESTATUS[0]}

  if [ "$exit_code" -eq 0 ]; then
    echo "[PASS] $runfile"
    passed=$((passed + 1))
  else
    echo "[FAIL] $runfile (exit code: $exit_code)"
    failed=$((failed + 1))
    if [ "$STOP_ON_ERROR" = "1" ]; then
      echo "[INFO] STOP_ON_ERROR=1 -> stopping queue."
      break
    fi
  fi

done

echo ""
echo "========== QUEUE SUMMARY =========="
echo "Passed: $passed"
echo "Failed: $failed"
echo "Logs  : $LOG_DIR"

if [ "$failed" -gt 0 ]; then
  exit 1
fi
