#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-submit}" # submit, status, download, or merge

INPUT_PATH="${INPUT_PATH:-data/speeches_1990_2022.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/noshot_4omini_batches}"
START="${START:-0}"
STOP="${STOP:-427014}"
CHUNK_SIZE="${CHUNK_SIZE:-25000}"

if [[ "$ACTION" == "merge" ]]; then
  python rr/run_noshot_batches.py merge --output-dir "$OUTPUT_DIR"
  exit 0
fi

for ((start = START; start < STOP; start += CHUNK_SIZE)); do
  stop=$((start + CHUNK_SIZE))
  if (( stop > STOP )); then
    stop="$STOP"
  fi

  echo "$ACTION rows $start..$stop"

  if [[ "$ACTION" == "submit" ]]; then
    python rr/run_noshot_batches.py submit \
      --prepare-first \
      --input-path "$INPUT_PATH" \
      --output-dir "$OUTPUT_DIR" \
      --start "$start" \
      --stop "$stop"
  else
    python rr/run_noshot_batches.py "$ACTION" \
      --output-dir "$OUTPUT_DIR" \
      --start "$start" \
      --stop "$stop"
  fi
done
