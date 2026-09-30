#!/usr/bin/env bash
# Submit the VIST training run. New runs start from random initialization;
# set RESUME_FROM only to continue the same run.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs}"

if [[ -n "${RESUME_FROM:-}" && ! -f "${RESUME_FROM}" ]]; then
  echo "ERROR: resume checkpoint not found: ${RESUME_FROM}" >&2
  exit 1
fi

RUN_NAME="${RUN_NAME:-vist_$(date +%Y%m%d_%H%M%S)}"
if [[ -z "${RESUME_FROM:-}" && -e "${RUN_ROOT}/${RUN_NAME}" ]]; then
  echo "ERROR: refusing to reuse existing run directory: ${RUN_ROOT}/${RUN_NAME}" >&2
  exit 1
fi

export PROJECT_DIR RUN_ROOT RUN_NAME
export BASE_CONFIG="${BASE_CONFIG:-${PROJECT_DIR}/configs/vist.yaml}"
export JOB_NAME="${JOB_NAME:-vist}"
export CONDA_ENV="${CONDA_ENV:-vist}"
export TIME="${TIME:-08:00:00}"
export CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
export MEM="${MEM:-48G}"
export GPUS="${GPUS:-1}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-2}"
export VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-0}"
export EPOCHS="${EPOCHS:-100}"
export LR="${LR:-3e-5}"
export VAL_EVERY="${VAL_EVERY:-4000}"
export DISABLE_PROGRESS="${DISABLE_PROGRESS:-1}"
export USE_TORCHRUN="${USE_TORCHRUN:-0}"

exec bash "${PROJECT_DIR}/cluster/submit_train_slurm.sh"
