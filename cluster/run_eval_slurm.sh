#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-vist}"
: "${CHECKPOINT:?Set CHECKPOINT=/path/to/best.pt or last.pt before submitting evaluation.}"
CONFIG="${CONFIG:-}"
VISUAL_MODE="${VISUAL_MODE:-all}"
NUM_BATCHES="${NUM_BATCHES:-all}"
NUM_STEPS="${NUM_STEPS:-1}"
USE_EMA="${USE_EMA:-0}"
NO_PROGRESS="${NO_PROGRESS:-0}"

cd "${PROJECT_DIR}"
if command -v module >/dev/null 2>&1; then
  module load conda >/dev/null 2>&1 || module load anaconda >/dev/null 2>&1 || module load miniconda >/dev/null 2>&1 || true
fi
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  echo "ERROR: conda was not found." >&2
  exit 1
fi
conda activate "${CONDA_ENV}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

CKPT_DIR="$(cd "$(dirname "${CHECKPOINT}")" && pwd)"
if [[ -z "${CONFIG}" ]]; then
  if [[ -f "${CKPT_DIR}/config.resolved.yaml" ]]; then
    CONFIG="${CKPT_DIR}/config.resolved.yaml"
  elif [[ -f "${CKPT_DIR}/../config.resolved.yaml" ]]; then
    CONFIG="${CKPT_DIR}/../config.resolved.yaml"
  else
    echo "ERROR: CONFIG was not set and config.resolved.yaml was not found next to the checkpoint." >&2
    exit 1
  fi
fi

OUT_CSV="${OUT_CSV:-${CKPT_DIR}/eval_${VISUAL_MODE}_$(date +%Y%m%d_%H%M%S).csv}"
EVAL_ARGS=(
  --config "${CONFIG}"
  --checkpoint "${CHECKPOINT}"
  --num_batches "${NUM_BATCHES}"
  --visual_mode "${VISUAL_MODE}"
  --num_steps "${NUM_STEPS}"
  --out_csv "${OUT_CSV}"
)
[[ "${USE_EMA}" == "1" ]] && EVAL_ARGS+=(--use_ema)
[[ "${NO_PROGRESS}" == "1" ]] && EVAL_ARGS+=(--no_progress)

python scripts/evaluate.py "${EVAL_ARGS[@]}"
