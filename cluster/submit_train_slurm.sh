#!/usr/bin/env bash
set -euo pipefail

# Generic Slurm submission wrapper for Visual-FLOSS training.
# Required: export SLURM_ACCOUNT=<your-account> (SNOW_ACCOUNT is accepted as
# a backwards-compatible alias).

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-${SNOW_ACCOUNT:-}}"
: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT to your Slurm account.}"

PARTITION="${PARTITION:-gpu}"
QOS="${QOS:-normal}"
TIME="${TIME:-48:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEM="${MEM:-32G}"
GPUS="${GPUS:-1}"
JOB_NAME="${JOB_NAME:-visual_floss}"
CONDA_ENV="${CONDA_ENV:-mambaflow}"
RUN_NAME="${RUN_NAME:-visual_floss_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs}"

mkdir -p "${RUN_ROOT}/slurm_logs"

# Optional sbatch scheduling controls.
#
#   DEPENDENCY  chain this job behind another, e.g. DEPENDENCY=afterany:2350948
#               Use afterany, NOT afterok: a job killed by the wall clock ends
#               in TIMEOUT, not COMPLETED, so an afterok dependency would never
#               be satisfied and SLURM would eventually cancel this job.
#   NODELIST    pin to specific node(s), e.g. NODELIST=node028
#   BEGIN       earliest start time, e.g. BEGIN=04:30, BEGIN=now+2hours,
#               BEGIN=2026-09-12T04:30:00, BEGIN=tomorrow.  This is a floor,
#               not a guarantee: the job still has to wait for a free GPU.
SBATCH_EXTRA_ARGS=()
if [[ -n "${DEPENDENCY:-}" ]]; then
  SBATCH_EXTRA_ARGS+=(--dependency="${DEPENDENCY}")
  echo "submit: dependency ${DEPENDENCY}"
fi
if [[ -n "${NODELIST:-}" ]]; then
  SBATCH_EXTRA_ARGS+=(--nodelist="${NODELIST}")
  echo "submit: nodelist ${NODELIST}"
fi
if [[ -n "${BEGIN:-}" ]]; then
  SBATCH_EXTRA_ARGS+=(--begin="${BEGIN}")
  echo "submit: begin ${BEGIN}"
fi
echo "submit: job=${JOB_NAME} run=${RUN_NAME}"
echo "submit: config=${BASE_CONFIG:-<wrapper default>}"
echo "submit: resume=${RESUME_FROM:-<none>} init=${INIT_FROM:-<none>}"

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
  --gres="${GRES:-gpu:${GPUS}}" \
  --time="${TIME}" \
  --output="${RUN_ROOT}/slurm_logs/%x_%j.out" \
  --error="${RUN_ROOT}/slurm_logs/%x_%j.err" \
  --export=ALL,GRES="${GRES:-gpu:${GPUS}}",GPUS="${GPUS}",USE_TORCHRUN="${USE_TORCHRUN:-0}",PROJECT_DIR="${PROJECT_DIR}",CONDA_ENV="${CONDA_ENV}",RUN_NAME="${RUN_NAME}",RUN_ROOT="${RUN_ROOT}",DATA_ROOT="${DATA_ROOT:-}",BASE_CONFIG="${BASE_CONFIG:-}",BATCH_SIZE="${BATCH_SIZE:-2}",NUM_WORKERS="${NUM_WORKERS:-4}",VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}",VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-0}",EPOCHS="${EPOCHS:-100}",LR="${LR:-3e-5}",VAL_EVERY="${VAL_EVERY:-8000}",VAL_BATCHES="${VAL_BATCHES:-}",MAX_STEPS="${MAX_STEPS:-}",RESUME_FROM="${RESUME_FROM:-}",INIT_FROM="${INIT_FROM:-}",DISABLE_PROGRESS="${DISABLE_PROGRESS:-0}",ACAPELLA_TRAIN="${ACAPELLA_TRAIN:-}",ACAPELLA_VAL="${ACAPELLA_VAL:-}",MUSDB_TRAIN="${MUSDB_TRAIN:-}",MUSDB_VAL="${MUSDB_VAL:-}",AUDIOSET_TRAIN="${AUDIOSET_TRAIN:-}",AUDIOSET_VAL="${AUDIOSET_VAL:-}" \
  "${PROJECT_DIR}/cluster/run_train_slurm.sh"
