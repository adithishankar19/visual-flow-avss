#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${CHECKPOINT:?Set CHECKPOINT=/path/to/best.pt or last.pt.}"

CKPT_DIR="$(cd "$(dirname "${CHECKPOINT}")" && pwd)"
CONFIG="${CONFIG:-${CKPT_DIR}/config.resolved.yaml}"
OUT_CSV="${OUT_CSV:-${CKPT_DIR}/eval_${VISUAL_MODE:-correct}_$(date +%Y%m%d_%H%M%S).csv}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

ARGS=(
  --config "${CONFIG}"
  --checkpoint "${CHECKPOINT}"
  --num_batches "${NUM_BATCHES:-all}"
  --visual_mode "${VISUAL_MODE:-correct}"
  --num_steps "${NUM_STEPS:-1}"
  --out_csv "${OUT_CSV}"
)
[[ "${USE_EMA:-1}" == "1" ]] && ARGS+=(--use_ema)
[[ "${NO_PROGRESS:-0}" == "1" ]] && ARGS+=(--no_progress)
[[ -n "${SAVE_DIR:-}" ]] && ARGS+=(--save_dir "${SAVE_DIR}")

python scripts/evaluate.py "${ARGS[@]}"
