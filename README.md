# Visual-FLOSS for Audio-Visual Singing Voice Separation

This repository contains a compact research implementation of **Visual-FLOSS**, a mixture-initialized flow objective for audio-visual singing voice separation. The public release intentionally provides one canonical experiment so the training objective and reproduction path are unambiguous.

## Objective

The released configuration optimizes exactly

\[
\mathcal{L}=0.05\,\mathcal{L}_{\mathrm{FLOSS}}+0.10\,\mathcal{L}_{\mathrm{MR\text{-}STFT}}.
\]

There is no waveform L1 deployment loss, endpoint L1 loss, drift loss, flow-matching loss, MeanFlow loss, or auxiliary reliability loss. Some of these quantities remain in the training logs as diagnostics, but their coefficients are zero.

Visual-FLOSS supervises velocities at both the deployment state and sampled intermediate states. The scale-normalized velocity error is expressed in decibels, while the multi-resolution STFT term encourages spectrally faithful separated audio.

## Repository layout

```text
configs/visual_floss_mrstft.yaml   canonical experiment
mambaflow/                        model, objective, and training code
patches/                          Acappella/MUSDB/AudioSet data loader
scripts/train_local.sh            portable local training wrapper
scripts/evaluate_local.sh         portable local evaluation wrapper
scripts/train.py                  Python training entry point
scripts/evaluate.py               Python evaluation entry point
scripts/infer_one.py              single-example inference
cluster/                          generic Slurm training/evaluation scripts
tests/                            focused model and Visual-FLOSS tests
```

## Installation

Python 3.10 and a CUDA-enabled PyTorch installation are recommended.

```bash
git clone <YOUR-GITHUB-URL>
cd visual-floss-avss

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Install a PyTorch build appropriate for your CUDA driver if the default package is unsuitable.

## Data

The experiment expects:

- Acappella vocal/video segments for the target source;
- MUSDB accompaniment audio;
- AudioSet interference audio.

You may edit the six placeholder paths in `configs/visual_floss_mrstft.yaml`. The recommended approach is to leave the tracked config unchanged and export machine-specific paths:

```bash
export ACAPELLA_TRAIN=/path/to/acappella/splits/train
export ACAPELLA_VAL=/path/to/acappella/splits/val_seen
export MUSDB_TRAIN=/path/to/musdb_accomp/train
export MUSDB_VAL=/path/to/musdb_accomp/valid
export AUDIOSET_TRAIN=/path/to/audioset_split/train
export AUDIOSET_VAL=/path/to/audioset_split/val
```

The repository does not redistribute datasets or trained checkpoints.

## Train locally

```bash
export DATA_ROOT=/path/to/data
export RUN_ROOT="$PWD/runs"
bash scripts/train_local.sh
```

`DATA_ROOT` is sufficient when it contains the directory structure shown above. Explicit `ACAPELLA_*`, `MUSDB_*`, and `AUDIOSET_*` variables override individual paths. The wrapper writes the fully resolved configuration into the run directory before training.

This is a from-scratch experiment. Do not pass an initialization checkpoint. With the default configuration, physical batch size is 2 and gradient accumulation is 4, giving an effective batch size of 8 on one GPU.

## Run with Slurm

The included wrapper converts environment variables into a run-specific configuration and submits the job:

```bash
export PROJECT_DIR="$PWD"
export BASE_CONFIG="$PWD/configs/visual_floss_mrstft.yaml"
export RUN_ROOT=/path/to/runs
export RUN_NAME="visual_floss_mrstft_$(date +%Y%m%d_%H%M%S)"

export SLURM_ACCOUNT=<slurm-account>
export PARTITION=medium
export QOS=normal
export TIME=08:00:00
export GRES=gpu:l40s:1
export GPUS=1
export CONDA_ENV=mambaflow

export DATA_ROOT=/path/to/data
export ACAPELLA_TRAIN=/path/to/acappella/splits/train
export ACAPELLA_VAL=/path/to/acappella/splits/val_seen
export MUSDB_TRAIN=/path/to/musdb_accomp/train
export MUSDB_VAL=/path/to/musdb_accomp/valid
export AUDIOSET_TRAIN=/path/to/audioset_split/train
export AUDIOSET_VAL=/path/to/audioset_split/val

unset INIT_FROM RESUME_FROM MAX_STEPS
bash cluster/submit_visual_floss_slurm.sh
```

To resume the same run, set `RESUME_FROM=/path/to/last.pt`, reuse its `RUN_NAME`, and run the same wrapper again. Keep `INIT_FROM` unset.

For cluster evaluation:

```bash
export SLURM_ACCOUNT=<slurm-account>
export CHECKPOINT=/path/to/best.pt
export CONFIG=/path/to/config.resolved.yaml  # optional when beside checkpoint
export NUM_STEPS=1
export VISUAL_MODE=correct
bash cluster/submit_eval_slurm.sh
```

## Evaluate

```bash
export CHECKPOINT=/path/to/best.pt
export CONFIG=/path/to/config.resolved.yaml  # optional when beside checkpoint
bash scripts/evaluate_local.sh
```

Run `python scripts/evaluate.py --help` for the complete set of dataset and inference options supported by your checkout.

## Tests

```bash
pip install -e '.[dev]'
python -m pytest tests/test_visual_floss.py tests/test_shapes.py -v
```

## Reproducibility notes

- Random seed: `1234`
- Sample rate: `16,384 Hz`
- One-step deployment inference by default
- Effective training batch size: `8`
- EMA decay: `0.999`; validation and best-checkpoint selection use EMA weights
- Best checkpoint metric: validation SI-SDR

Results depend on the exact dataset construction and filtering. Report the number of valid evaluation examples and avoid silently replacing missing or silent interferers with zero audio.

## Citation

A formal citation will be added when the accompanying paper is public. Until then, please cite the repository URL and commit hash.

## License

No license is granted by default. Add an explicit license before accepting external contributions or permitting reuse.
