#!/usr/bin/env bash
set -euo pipefail

# This file is executed inside the Slurm allocation. Prefer submitting it with
# cluster/submit_train_slurm.sh so resources are requested consistently.

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BASE_CONFIG="${BASE_CONFIG:-configs/visual_floss_mrstft.yaml}"
CONDA_ENV="${CONDA_ENV:-mambaflow}"
RUN_NAME="${RUN_NAME:-visual_floss_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs_cluster}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/${RUN_NAME}}"
DATA_ROOT="${DATA_ROOT:-}"

BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-0}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-3e-5}"
VAL_EVERY="${VAL_EVERY:-8000}"
VAL_BATCHES="${VAL_BATCHES:-}"
MAX_STEPS="${MAX_STEPS:-}"
RESUME_FROM="${RESUME_FROM:-}"
INIT_FROM="${INIT_FROM:-}"
DISABLE_PROGRESS="${DISABLE_PROGRESS:-0}"

mkdir -p "${RUN_DIR}"
cd "${PROJECT_DIR}"

# Load Conda if the site provides an environment module. The exact module name
# varies by cluster image, so failures here are non-fatal as long as conda exists.
if command -v module >/dev/null 2>&1; then
  module load conda >/dev/null 2>&1 || module load anaconda >/dev/null 2>&1 || module load miniconda >/dev/null 2>&1 || true
fi

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  echo "ERROR: conda was not found. Load the correct module or set up the environment before submitting." >&2
  exit 1
fi

conda activate "${CONDA_ENV}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

python - <<'PY'
import os, torch
print("Python CUDA available:", torch.cuda.is_available())
print("CUDA visible devices:", os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>"))
if torch.cuda.is_available():
    print("CUDA device 0:", torch.cuda.get_device_name(0))
PY

CONFIG_OUT="${RUN_DIR}/config.resolved.yaml"
RENDER_ARGS=(
  --base-config "${BASE_CONFIG}"
  --out "${CONFIG_OUT}"
  --run-dir "${RUN_DIR}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --val-batch-size "${VAL_BATCH_SIZE}"
  --val-num-workers "${VAL_NUM_WORKERS}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --val-every "${VAL_EVERY}"
  --device cuda
)

if [[ -n "${VAL_BATCHES}" ]]; then
  RENDER_ARGS+=(--val-batches "${VAL_BATCHES}")
fi

if [[ -n "${DATA_ROOT}" ]]; then
  RENDER_ARGS+=(--data-root "${DATA_ROOT}")
fi
[[ -n "${ACAPELLA_TRAIN:-}" ]] && RENDER_ARGS+=(--acapella-train "${ACAPELLA_TRAIN}")
[[ -n "${ACAPELLA_VAL:-}" ]] && RENDER_ARGS+=(--acapella-val "${ACAPELLA_VAL}")
[[ -n "${MUSDB_TRAIN:-}" ]] && RENDER_ARGS+=(--musdb-train "${MUSDB_TRAIN}")
[[ -n "${MUSDB_VAL:-}" ]] && RENDER_ARGS+=(--musdb-val "${MUSDB_VAL}")
[[ -n "${AUDIOSET_TRAIN:-}" ]] && RENDER_ARGS+=(--audioset-train "${AUDIOSET_TRAIN}")
[[ -n "${AUDIOSET_VAL:-}" ]] && RENDER_ARGS+=(--audioset-val "${AUDIOSET_VAL}")
[[ "${DISABLE_PROGRESS}" == "1" ]] && RENDER_ARGS+=(--disable-progress)

python scripts/render_config.py "${RENDER_ARGS[@]}"
cp "${BASE_CONFIG}" "${RUN_DIR}/base_config.yaml"
cp "$0" "${RUN_DIR}/run_train_slurm.sh"

TRAIN_ARGS=(--config "${CONFIG_OUT}")
[[ -n "${MAX_STEPS}" ]] && TRAIN_ARGS+=(--max_steps "${MAX_STEPS}")
[[ -n "${RESUME_FROM}" ]] && TRAIN_ARGS+=(--resume_from "${RESUME_FROM}")
[[ -n "${INIT_FROM}" ]] && TRAIN_ARGS+=(--init_from "${INIT_FROM}")

python scripts/train.py "${TRAIN_ARGS[@]}"
