# VIST: Visually Indexed, Mixture-Anchored Source Transport for One-Step Audiovisual Singing Voice Separation

Adithi Shankar, Gloria Haro, Xavier Serra, Martín Rocamora
Universitat Pompeu Fabra, Barcelona

Code and audio examples for our ICASSP 2027 submission.

VIST separates a target singing voice from a music mixture that also contains accompaniment and a second, similar singer. The video of the target singer indicates which voice to return. A single velocity network is trained on straight paths from a perturbed copy of the mixture to the target voice. At test time, the target is estimated with **one network evaluation** starting from the mixture:

$$
\hat{\mathbf{S}} = \mathbf{M} + u_\theta(\mathbf{M}, 0 \mid \mathbf{M}, \mathbf{V}), \qquad \hat{\mathbf{B}} = \mathbf{M} - \hat{\mathbf{S}}
$$

There is no noise sampling and no step that assigns outputs to sources. The accompaniment and the interfering singer are recovered as the remainder of the mixture.

## Results

We evaluate on the unseen–unheard test set of Acappella under a 100% interference stress test: 1,500 fixed mixtures, each containing an interfering singer. The mixtures follow the MambaVoice protocol. Scores are in dB.

| Model | Params (M) | SDR mean | SDR median | SI-SDR | SIR |
|---|---:|---:|---:|---:|---:|
| Audio-only | 14.8 | 0.64 | 0.45 | 0.30 | 1.42 |
| VoViT | 39.0 | 11.82 | 12.63 | 10.08 | 19.37 |
| VoViT<sup>100</sup> | 39.0 | 8.83 | 9.94 | 8.53 | 13.69 |
| MambaVoice | 16.2 | 11.11 | 11.55 | 9.71 | 17.14 |
| **VIST** | 23.2 | **12.18** | **13.32** | **11.12** | **20.22** |

- Integrating the learned field with more Euler steps does not help: K = 1, 2, 4 steps give 12.18, 12.08, and 12.06 dB SDR.
- With the correct face, the model separates the wrong singer (target swapping) in 3.8% of mixtures. When the face is zeroed, time-shifted, or taken from another singer, this rises to 48–55%, so target selection depends on the time alignment of face and voice.

## Audio examples

[`samples/`](samples) contains five test mixtures from the unseen–unheard set (100% interference, one-step inference, 16,384 Hz, 4 s each). Each folder contains:

| File | Content |
|---|---|
| `mixture.wav` | input: target voice + interfering singer + accompaniment |
| `target_estimate.wav` | VIST estimate of the target voice, $\hat{S}$ |
| `target_reference.wav` | ground-truth target voice |
| `residual_estimate.wav` | estimated remainder, $M - \hat{S}$ |
| `residual_reference.wav` | ground-truth remainder |
| `metadata.json` | per-example SI-SDR scores |

| Example | Input SI-SDR | Output SI-SDR | SI-SDRi |
|---|---:|---:|---:|
| [`example_1352`](samples/example_1352) | −4.94 | 16.17 | 21.11 |
| [`example_0775`](samples/example_0775) | −2.30 | 17.22 | 19.52 |
| [`example_0351`](samples/example_0351) | −2.45 | 12.00 | 14.45 |
| [`example_1039`](samples/example_1039) | −0.31 | 12.39 | 12.70 |
| [`example_0296`](samples/example_0296) | 3.06 | 5.85 | 2.78 |

The examples range from strong to weak separations. `example_0296` shows a harder case. [`samples/metadata_test_unseen_1500.csv`](samples/metadata_test_unseen_1500.csv) lists the per-mixture SI-SDR scores for all 1,500 test mixtures.

## Method

**Training paths.** For a mixture STFT $\mathbf{M}$ and target $\mathbf{S}$, the path state is $\mathbf{z}_t = (1-t)(\mathbf{M}+\mathbf{n}) + t\mathbf{S}$ and the velocity target is $\mathbf{v}^\star = \mathbf{S}-\mathbf{M}-\mathbf{n}$. Here $\mathbf{n}$ is the STFT of white noise whose standard deviation is 0.1 × the mixture RMS. Half of each batch is placed at the inference state $(\mathbf{n}, t) = (\mathbf{0}, 0)$. The remaining states draw $t$ from a logit-normal distribution ($\mu=-0.4$, $\sigma=1$, $t \le 0.95$). Both the noise level and $t$ are ramped up over the first 16k updates.

**Objective.** The network is trained with

$$
\mathcal{L} = 0.05\,\mathcal{L}_{\mathrm{vel}} + 0.10\,\mathcal{L}_{\mathrm{MR}} + 0.05\,\mathcal{L}_{\mathrm{rel}},
$$

where $\mathcal{L}_{\mathrm{vel}} = \mathrm{clip}\big(10\log_{10}\tfrac{\lVert\hat{\mathbf{v}}-\mathbf{v}^\star\rVert^2+\epsilon}{\lVert\mathbf{v}^\star\rVert^2+\epsilon},\,-20,\,30\big)$ is the scale-normalized decibel velocity loss from FLOSS. $\mathcal{L}_{\mathrm{MR}}$ is a multi-resolution STFT loss on the one-step estimate, computed in a second pass at the inference state. $\mathcal{L}_{\mathrm{rel}}$ is a binary cross-entropy that trains the visual reliability gate $r$ to detect the 10% of visual frames that are zeroed at random during training.

**Network.** The network is a four-level TFC-TDF U-Net with 60/120/180/240 channels and two blocks per level. Its input is $[\mathbf{z}_t; \mathbf{M}]$ as 513-bin complex STFTs (window 1024, hop 256). The mixture is encoded with the band-split attention encoder from MambaVoice. The target singer's face is encoded with an ST-GCN over 68 facial landmarks. The audio and visual tokens are fused with FiLM and condition the U-Net at every scale. At the bottleneck, a cross-attention layer over the visual tokens is gated by a learned reliability $r \in [0.25, 1]$. The model has 23.2M parameters.

## Repository layout

```text
configs/vist.yaml          the VIST experiment (all hyperparameters of the paper)
vist/
  model.py                 training paths, loss (Eqs. 1-6) and one-step inference (Eq. 3)
  unet.py                  TFC-TDF U-Net velocity network
  conditioner.py           FiLM fusion of audio and visual tokens, reliability r
  encoders/                band-split audio encoder and ST-GCN over facial landmarks
  data/acappella.py        Acappella + MUSDB18 + AudioSet mixtures
  trainer.py               AdamW, warm-up + cosine, EMA, checkpoint selection
scripts/
  train.py, train_local.sh           training
  evaluate.py, evaluate_local.sh     SDR / SIR / SI-SDR, visual interventions, target swapping
  separate.py                        separate one mixture
  render_config.py                   fill in machine-specific paths
cluster/                   Slurm wrappers
tests/                     unit tests
samples/                   audio examples and per-mixture test scores
```

## Installation

Python 3.10 and a CUDA build of PyTorch are recommended.

```bash
git clone https://github.com/adithishankar19/visual-flow-avss.git
cd visual-flow-avss
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

If the default wheel does not match your CUDA driver, install a PyTorch build that does.

## Data

The experiment uses:

- [Acappella](https://ipcv.github.io/Acappella/): target singing videos, as audio + 68 facial landmarks per frame
- [MUSDB18](https://sigsep.github.io/datasets/musdb.html): accompaniment stems
- [AudioSet](https://research.google.com/audioset/): vocal-like interference clips

The simplest setup is a single `DATA_ROOT` with this structure:

```text
$DATA_ROOT/
  acappella/splits/{train,val_seen,test_unseen}
  musdb_accomp/{train,valid,test}
  audioset_split/{train,val,test}
```

To use a different layout, set individual paths. These override `DATA_ROOT`:

```bash
export ACAPELLA_TRAIN=/path/to/acappella/splits/train
export ACAPELLA_VAL=/path/to/acappella/splits/val_seen
export MUSDB_TRAIN=/path/to/musdb_accomp/train
export MUSDB_VAL=/path/to/musdb_accomp/valid
export AUDIOSET_TRAIN=/path/to/audioset_split/train
export AUDIOSET_VAL=/path/to/audioset_split/val
```

This repository does not redistribute datasets or trained checkpoints.

## Training

```bash
export DATA_ROOT=/path/to/data
export RUN_ROOT="$PWD/runs"
bash scripts/train_local.sh
```

The model is trained from random initialization. With the default configuration, the batch size is 2 with 4 gradient-accumulation steps, which gives an effective batch size of 8 on one GPU. The wrapper writes the fully resolved configuration to the run directory before training starts.

<details>
<summary>Slurm</summary>

```bash
export PROJECT_DIR="$PWD"
export BASE_CONFIG="$PWD/configs/vist.yaml"
export RUN_ROOT=/path/to/runs
export RUN_NAME="vist_$(date +%Y%m%d_%H%M%S)"

export SLURM_ACCOUNT=<account>
export PARTITION=medium QOS=normal TIME=08:00:00
export GRES=gpu:l40s:1 GPUS=1
export CONDA_ENV=vist
export DATA_ROOT=/path/to/data

unset RESUME_FROM MAX_STEPS
bash cluster/submit_vist_slurm.sh
```

To resume a run, reuse its `RUN_NAME`, set `RESUME_FROM=/path/to/last.pt`, and submit again.

For evaluation, see `cluster/submit_eval_slurm.sh`. It takes the same `CHECKPOINT`, `CONFIG`, `NUM_STEPS`, and `VISUAL_MODE` variables as the local script below.

</details>

## Evaluation

```bash
export CHECKPOINT=/path/to/best.pt
export CONFIG=/path/to/config.resolved.yaml   # optional if it sits next to the checkpoint
export NUM_STEPS=1                            # Euler steps (paper: 1)
export VISUAL_MODE=correct                    # correct | zero | shift | wrong | all
bash scripts/evaluate_local.sh
```

For every mixture, the script reports BSS-Eval SDR and SIR (512-tap distortion filters, with the target and the rest of the mixture as references, after resampling to 16 kHz), SI-SDR, and whether the target was swapped, i.e. whether the remainder $\mathbf{M}-\hat{\mathbf{S}}$ is closer to the target than $\hat{\mathbf{S}}$ in SI-SDR. It prints means with 95% bootstrap confidence intervals, and `OUT_CSV` keeps the per-mixture scores. Set `SAVE_DIR` to also write the audio in the layout of [`samples/`](samples).

`VISUAL_MODE` runs the visual interventions of Table 3: the correct face, zeroed landmarks, landmarks shifted in time by half the clip, and the landmarks of a singer from another test clip. `all` also prints the paired SDR differences to the correct face. EMA weights are used by default. See `python scripts/evaluate.py --help` for all options.

To separate a single mixture:

```bash
python scripts/separate.py --checkpoint best.pt --mixture mix.wav --landmarks face.npy --out_dir out/
```

The mixture must be at 16,384 Hz, and `face.npy` holds the target singer's 68 facial landmarks at 25 fps, with shape `[T, 2, 68]`.

By default, the validation slot is the seen-singer validation split, so that checkpoint selection never sees the test set. To evaluate on the unseen–unheard test set, render the config with `python scripts/render_config.py --use-test-splits ...`.

## Tests

```bash
pip install -e '.[dev]'
python -m pytest tests -v
```

## Reproducibility notes

| | |
|---|---|
| Seed | 1234 |
| Sample rate | 16,384 Hz |
| STFT | 1024 window, 256 hop |
| Optimizer | AdamW, lr 3e-5 (12k warm-up, cosine to 1e-6), wd 1e-4, grad clip 1 |
| Effective batch | 8 (2 × 4 accumulation), bf16 |
| EMA | 0.999, used for validation and checkpoint selection |
| Checkpoint selection | best validation SI-SDR on the seen-singer split |
| Inference | 1 network evaluation |

Results depend on how the dataset is built and filtered. When reporting numbers, give the number of valid evaluation mixtures, and do not silently replace missing or silent interferers with zeros.

## Citation

```bibtex
@misc{shankar2026vist,
  title  = {{VIST}: Visually Indexed, Mixture-Anchored Source Transport for One-Step Audiovisual Singing Voice Separation},
  author = {Shankar, Adithi and Haro, Gloria and Serra, Xavier and Rocamora, Mart{\'i}n},
  note   = {Submitted to ICASSP 2027},
  year   = {2026}
}
```

## Acknowledgements

This work builds on the [Acappella / VoViT](https://github.com/JuanFMontesinos/VoViT)


