#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BASE_CONFIG="${BASE_CONFIG:-${PROJECT_DIR}/configs/visual_floss_mrstft.yaml}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs}"
RUN_NAME="${RUN_NAME:-visual_floss_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/${RUN_NAME}}"

mkdir -p "${RUN_DIR}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

RENDER_ARGS=(
  --base-config "${BASE_CONFIG}"
  --out "${RUN_DIR}/config.resolved.yaml"
  --run-dir "${RUN_DIR}"
  --batch-size "${BATCH_SIZE:-2}"
  --num-workers "${NUM_WORKERS:-4}"
  --val-batch-size "${VAL_BATCH_SIZE:-2}"
  --val-num-workers "${VAL_NUM_WORKERS:-0}"
  --epochs "${EPOCHS:-100}"
  --lr "${LR:-3e-5}"
  --val-every "${VAL_EVERY:-4000}"
  --device "${DEVICE:-cuda}"
)

[[ -n "${DATA_ROOT:-}" ]] && RENDER_ARGS+=(--data-root "${DATA_ROOT}")
[[ -n "${ACAPELLA_TRAIN:-}" ]] && RENDER_ARGS+=(--acapella-train "${ACAPELLA_TRAIN}")
[[ -n "${ACAPELLA_VAL:-}" ]] && RENDER_ARGS+=(--acapella-val "${ACAPELLA_VAL}")
[[ -n "${MUSDB_TRAIN:-}" ]] && RENDER_ARGS+=(--musdb-train "${MUSDB_TRAIN}")
[[ -n "${MUSDB_VAL:-}" ]] && RENDER_ARGS+=(--musdb-val "${MUSDB_VAL}")
[[ -n "${AUDIOSET_TRAIN:-}" ]] && RENDER_ARGS+=(--audioset-train "${AUDIOSET_TRAIN}")
[[ -n "${AUDIOSET_VAL:-}" ]] && RENDER_ARGS+=(--audioset-val "${AUDIOSET_VAL}")
[[ "${DISABLE_PROGRESS:-0}" == "1" ]] && RENDER_ARGS+=(--disable-progress)

python scripts/render_config.py "${RENDER_ARGS[@]}"

TRAIN_ARGS=(--config "${RUN_DIR}/config.resolved.yaml")
[[ -n "${MAX_STEPS:-}" ]] && TRAIN_ARGS+=(--max_steps "${MAX_STEPS}")
[[ -n "${RESUME_FROM:-}" ]] && TRAIN_ARGS+=(--resume_from "${RESUME_FROM}")

python scripts/train.py "${TRAIN_ARGS[@]}"
