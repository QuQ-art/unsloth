#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ID="${1:-0}"
if (($#)); then shift; fi
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ID="$(date +%Y%m%d_%H%M%S)_gpu${GPU_ID}_$$"
RUN_DIR="${RESULTS_DIR:-$SCRIPT_DIR/output}/$RUN_ID"
mkdir -p "$RUN_DIR"
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
cp "$SCRIPT_DIR"/*.py "$SCRIPT_DIR/run.sh" "$RUN_DIR/"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$SCRIPT_DIR/.cache/datasets}"
export UNSLOTH_COMPILE_LOCATION="${UNSLOTH_COMPILE_LOCATION:-$SCRIPT_DIR/.cache/unsloth_compiled_cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$SCRIPT_DIR/.cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SCRIPT_DIR/.cache/triton}"

echo "GPU=$GPU_ID, output=$RUN_DIR"
nvidia-smi -i "$GPU_ID" --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu --format=csv -l 1 > "$RUN_DIR/gpu.csv" &
MON_PID=$!
cleanup(){ kill "$MON_PID" 2>/dev/null || true; wait "$MON_PID" 2>/dev/null || true; }
trap cleanup EXIT

set +e
"$PYTHON_BIN" "$SCRIPT_DIR/train.py" "$@" --output-dir "$RUN_DIR" 2>&1 | tee "$RUN_DIR/train.log"
status=${PIPESTATUS[0]}
set -e
printf 'exit_code=%s\n' "$status" > "$RUN_DIR/exit_status.txt"
exit "$status"
