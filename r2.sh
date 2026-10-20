#!/usr/bin/env bash
# Stage B experiment configuration. run.sh owns execution and performance logs.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if (($# > 1)); then
    echo "Usage: bash r2.sh [GPU_ID]" >&2
    exit 2
fi
GPU_ID="${1:-0}"

# Local resources; override these environment variables for another machine.
MODEL_ROOT="${MODEL_ROOT:-/home/xxf/Distill/models/OPD}"
STUDENT="${STUDENT:-$MODEL_ROOT/MixSFT}"
TEACHER_MATH="${TEACHER_MATH:-$MODEL_ROOT/Math}"
TEACHER_CODE="${TEACHER_CODE:-$MODEL_ROOT/Code}"
TEACHER_IF="${TEACHER_IF:-$MODEL_ROOT/IF}"
DATASET="${DATASET:-$MODEL_ROOT/data/rl_prompt_mix/train.parquet}"
export PYTHON_BIN="${PYTHON_BIN:-/home/zst/miniconda3/envs/unsloth_opd/bin/python}"
# Use this fork's implementation, matching the validated experiment.
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

# Fixed Stage B coverage experiment: full BF16 student, three offloaded teachers.
# train.py fixes the optimizer, weight decay, scheduler and gradient clipping.
EXPERIMENT_ARGS=(
    --student "$STUDENT"
    --teacher "$TEACHER_MATH"
    --teacher-code "$TEACHER_CODE"
    --teacher-if "$TEACHER_IF"
    --dataset "$DATASET"
    --steps 6
    --batch-size 1
    --grad-accum 1
    --max-prompt-length 512
    --max-new-tokens 1024
    --max-length 1536
    --data-rows 0
    --learning-rate 1e-6
    --lmbda 1.0
    --beta 1.0
    --seed 42
)

exec bash "$ROOT/examples/mopd-baseline/run.sh" "$GPU_ID" "${EXPERIMENT_ARGS[@]}"
