#!/usr/bin/env bash
set -euo pipefail

# Required: export SLURM_ACCOUNT=<your account> and CHECKPOINT=/path/to/best.pt.
# SNOW_ACCOUNT is accepted as a backwards-compatible alias.
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-${SNOW_ACCOUNT:-}}"
: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT to your Slurm account.}"
: "${CHECKPOINT:?Set CHECKPOINT=/path/to/best.pt or /path/to/last.pt.}"

PARTITION="${PARTITION:-gpu}"
QOS="${QOS:-normal}"
TIME="${TIME:-04:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-2}"
MEM="${MEM:-24G}"
GPUS="${GPUS:-1}"
JOB_NAME="${JOB_NAME:-visual_floss_eval}"
CONDA_ENV="${CONDA_ENV:-mambaflow}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs}"
mkdir -p "${RUN_ROOT}/slurm_logs"


SBATCH_EXTRA_ARGS=()
if [[ -n "${NODELIST:-}" ]]; then
  SBATCH_EXTRA_ARGS+=(--nodelist="${NODELIST}")
fi

# ${arr[@]+"${arr[@]}"} so an EMPTY array does not trip `set -u` on bash < 4.4.
sbatch ${SBATCH_EXTRA_ARGS[@]+"${SBATCH_EXTRA_ARGS[@]}"} \
  --job-name="${JOB_NAME}" \
  --partition="${PARTITION}" \
  --account="${SLURM_ACCOUNT}" \
  --qos="${QOS}" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --mem="${MEM}" \
  --gres="gpu:${GPUS}" \
  --time="${TIME}" \
  --output="${RUN_ROOT}/slurm_logs/%x_%j.out" \
  --error="${RUN_ROOT}/slurm_logs/%x_%j.err" \
  --export=ALL,PROJECT_DIR="${PROJECT_DIR}",CONDA_ENV="${CONDA_ENV}",CHECKPOINT="${CHECKPOINT}",CONFIG="${CONFIG:-}",VISUAL_MODE="${VISUAL_MODE:-all}",NUM_BATCHES="${NUM_BATCHES:-all}",BATCH_SIZE="${BATCH_SIZE:-1}",NUM_STEPS="${NUM_STEPS:-1}",USE_EMA="${USE_EMA:-0}",NO_PROGRESS="${NO_PROGRESS:-0}",OUT_CSV="${OUT_CSV:-}" \
  "${PROJECT_DIR}/cluster/run_eval_slurm.sh"
