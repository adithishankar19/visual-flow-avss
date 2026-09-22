from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import contextlib

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from mambaflow.data import SyntheticAcapellaDataset, build_vovit_dataset
from mambaflow.losses import si_sdr
from mambaflow.models import MambaVoiceMCFlow


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    """Seed python/numpy/torch RNGs.

    Note this does NOT by itself make the dataset deterministic: the Acapella
    dataset draws its interferer, accompaniment and SNRs from the global
    `random` module inside __getitem__, which DataLoader workers reseed every
    epoch.  Pass `deterministic: true` in the dataset init_kwargs for that.
    """
    import random as _random

    import numpy as _np

    _random.seed(seed)
    _np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(cfg: Dict[str, Any], split: str = "train"):
    data_cfg = cfg.get("data", {})
    kind = data_cfg.get("kind", "synthetic")
    if kind == "synthetic":
        return SyntheticAcapellaDataset(**data_cfg.get("synthetic", {}))
    if kind == "vovit":
        return build_vovit_dataset(data_cfg.get(split, data_cfg.get("train", {})))
    raise ValueError(f"Unknown data.kind={kind!r}")


def collate_dict(batch):
    keys = set().union(*(b.keys() for b in batch))
    out = {}
    for k in keys:
        vals = [b.get(k) for b in batch]
        if vals[0] is None:
            continue
        if torch.is_tensor(vals[0]):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    return out


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _make_loader(
    cfg: Dict[str, Any],
    split: str,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    drop_last: bool,
) -> DataLoader:
    ds = build_dataset(cfg, split)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_dict,
        drop_last=drop_last,
    )


def _parse_optional_int(value: Any, *, none_tokens: tuple[str, ...] = ("all", "full", "none", "null")) -> Optional[int]:
    """Parse config values where null/all means no limit."""
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().lower() in none_tokens:
            return None
        return int(value)
    value = int(value)
    return None if value <= 0 else value


def _safe_len(obj: Any) -> Optional[int]:
    try:
        return len(obj)
    except Exception:
        return None


def _batch_indices(batch: Dict[str, Any], batch_size: int) -> list[Any]:
    """Best-effort sample ids for debug printing."""
    for key in ("idx", "index", "indices", "sample_idx", "sample_id", "id"):
        value = batch.get(key)
        if value is None:
            continue
        if torch.is_tensor(value):
            value = value.detach().cpu().tolist()
        if isinstance(value, (list, tuple)):
            return list(value)[:batch_size]
        return [value for _ in range(batch_size)]
    return list(range(batch_size))


def _is_frozen_prior_param(name: str) -> bool:
    """True for the parameters of a frozen first-stage prior (flow.prior).

    Those carry the same `conditioner.bundle.model.*` names as the model's own
    feature extractor, so substring matching would count them into the
    trainability groups and report the live extractor as PARTIAL.
    """
    parts = name.split(".")
    if parts and parts[0] == "module":
        parts = parts[1:]
    return bool(parts) and parts[0] == "prior"


def _count_named_params(model: torch.nn.Module, needle: str) -> tuple[int, int]:
    """Return (total, trainable) parameter counts whose names contain `needle`."""
    total = 0
    trainable = 0
    for name, param in model.named_parameters():
        if _is_frozen_prior_param(name):
            continue
        if needle in name:
            n = param.numel()
            total += n
            if param.requires_grad:
                trainable += n
    return total, trainable


def _log_feature_extractor_trainability(
    model: torch.nn.Module,
    *,
    assert_trainable: bool = False,
) -> None:
    """Print whether the local ST-GCN and BandSplit/audio encoder are trainable."""
    groups = {
        "local ST-GCN / graph_net": "conditioner.bundle.model.graph_net",
        "local BandSplit / audio_net": "conditioner.bundle.model.audio_net",
        "all local feature extractor": "conditioner.bundle.model",
    }
    missing_or_frozen: list[str] = []
    for label, needle in groups.items():
        total, trainable = _count_named_params(model, needle)
        if total == 0:
            status = "MISSING"
            missing_or_frozen.append(label)
        elif trainable == 0:
            status = "FROZEN"
            missing_or_frozen.append(label)
        elif trainable == total:
            status = "TRAINABLE"
        else:
            status = "PARTIAL"
        tqdm.write(
            f"feature trainability: {label}: {status} "
            f"trainable={trainable:,} / total={total:,}"
        )
    if assert_trainable and missing_or_frozen:
        raise RuntimeError(
            "Feature extractor was expected to be trainable, but these groups are missing/frozen: "
            + ", ".join(missing_or_frozen)
            + ". Check backbone.use_local_feature_extractor=true and backbone.freeze=false."
        )


class TrainableEMA:
    """Exponential moving average for trainable parameters only."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        if not (0.0 < float(decay) < 1.0):
            raise ValueError(f"ema_decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()
        if not self.shadow:
            raise RuntimeError("EMA requested but no trainable parameters were found")

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        decay = self.decay
        params = dict(model.named_parameters())
        for name, avg in self.shadow.items():
            param = params.get(name)
            if param is None:
                continue
            avg.mul_(decay).add_(param.detach(), alpha=1.0 - decay)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "decay": self.decay,
            "shadow": {k: v.detach().cpu() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state: Dict[str, Any], device: torch.device | str | None = None) -> None:
        self.decay = float(state.get("decay", self.decay))
        shadow = state.get("shadow", state)
        self.shadow = {k: v.detach().clone().to(device) if device is not None else v.detach().clone() for k, v in shadow.items()}

    @contextlib.contextmanager
    def apply_to(self, model: torch.nn.Module) -> Iterator[None]:
        """Temporarily swap EMA weights into the model."""
        params = dict(model.named_parameters())
        backup: Dict[str, torch.Tensor] = {}
        try:
            with torch.no_grad():
                for name, avg in self.shadow.items():
                    param = params.get(name)
                    if param is None:
                        continue
                    backup[name] = param.detach().clone()
                    param.copy_(avg.to(device=param.device, dtype=param.dtype))
            yield
        finally:
            with torch.no_grad():
                for name, old in backup.items():
                    params[name].copy_(old.to(device=params[name].device, dtype=params[name].dtype))


def _load_ema_weights_into_model(model: torch.nn.Module, ema_state: Dict[str, Any]) -> None:
    """Permanently copy EMA trainable weights into a model for evaluation."""
    shadow = ema_state.get("shadow", ema_state)
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, avg in shadow.items():
            param = params.get(name)
            if param is not None:
                param.copy_(avg.to(device=param.device, dtype=param.dtype))


@torch.no_grad()
def validate(
    model: MambaVoiceMCFlow,
    loader: DataLoader,
    device: torch.device,
    *,
    num_batches: Optional[int] = None,
    num_steps: Optional[int] = None,
    show_progress: bool = True,
    desc: str = "val",
    diagnostic_batches: int = 0,
    diagnostic_examples: int = 4,
) -> Dict[str, float]:
    # Save exact training/eval state of all modules
    module_training_states = [
        (module, module.training)
        for module in model.modules()
    ]
    model.eval()

    totals: Dict[str, float] = {}
    count = 0
    sdr_values = []
    base_sdr_values = []  # For dual-head base-only SI-SDR
    correction_gain_values = []  # For dual-head correction gain
    residual_to_target_values = []
    mixture_to_target_values = []
    halfmix_to_target_values = []
    target_left_in_residual_flags = []
    flow_hurts_halfmix_flags = []
    target_rms_values = []
    pred_rms_values = []
    target_gain_values = []
    residual_target_gain_values = []
    interferer_gain_values = []
    pred_target_rms_ratio_values = []
    mix_err_values = []

    total = _safe_len(loader)
    if num_batches is not None and total is not None:
        total = min(total, num_batches)
    iterator = tqdm(loader, total=total, desc=desc, leave=False, disable=not show_progress)

    for i, batch in enumerate(iterator):
        if num_batches is not None and i >= num_batches:
            break
        batch = _move_batch_to_device(batch, device)

        losses = model.training_loss(batch)
        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + float(v.detach())

        if "mixture" in batch and "target" in batch:
            out = model.separate(
                batch["mixture"],
                face=batch.get("face"),
                body=batch.get("body"),
                num_steps=num_steps,
            )
            pred_target = out["target"]
            pred_residual = out.get("residual", batch["mixture"] - pred_target)
            target = batch["target"]
            mixture = batch["mixture"]
            l = min(pred_target.shape[-1], pred_residual.shape[-1], target.shape[-1], mixture.shape[-1])

            pred_target_l = pred_target[..., :l]
            pred_residual_l = pred_residual[..., :l]
            target_l = target[..., :l]
            mixture_l = mixture[..., :l]
            halfmix_l = 0.5 * mixture_l

            sisdr_target = si_sdr(target_l, pred_target_l)
            sdr_values.append(sisdr_target.mean().item())

            # Base-only validation for either the legacy dual head or the
            # frozen-drift flow-map adapter.
            has_correction_head = (
                (
                    getattr(model.head, "dual_head", False)
                    and model.head.delta_conv is not None
                )
                or getattr(model.head, "has_flowmap_adapter", False)
            )
            if has_correction_head:
                # Compute base-only SI-SDR for drift anchor preservation
                from mambaflow.audio import stft_waveform, complex_to_ri, ri_to_sources_complex, istft_waveform

                mix_spec = stft_waveform(batch["mixture"], model.stft_cfg)
                mixture_ri = complex_to_ri(mix_spec)

                # Get conditioning
                captured = model.condition(
                    batch["mixture"],
                    batch.get("face"),
                    batch.get("body"),
                )
                cond = captured["conditioning"]
                t_zero = torch.zeros(pred_target.shape[0], device=pred_target.device)

                # Base-only prediction at one-step: u_base is residual velocity (S-M)
                with torch.no_grad():
                    u_base, _, _ = model.head.forward_dual_head(
                        mixture_ri, mixture_ri, cond, t_zero,
                        temporal_tokens=captured.get("temporal_tokens"),
                        visual_activity=captured.get("visual_activity"),
                        cross_attention_tokens=captured.get("video_tokens"),
                        interval_end=torch.ones_like(t_zero),
                    )

                # Base target estimate: S_base = M + u_base (endpoint of residual path)
                base_endpoint_ri = mixture_ri + u_base
                base_complex = ri_to_sources_complex(base_endpoint_ri, num_sources=1)[:, 0]
                pred_target_base = istft_waveform(
                    base_complex,
                    model.stft_cfg,
                    length=batch["mixture"].shape[-1],
                )
                pred_target_base_l = pred_target_base[..., :l]
                sisdr_target_base = si_sdr(target_l, pred_target_base_l)

                # Accumulate base SI-SDR and correction gain
                base_sdr_values.append(sisdr_target_base.mean().item())
                correction_gain_values.append(sisdr_target.mean().item() - sisdr_target_base.mean().item())

            sisdr_residual = si_sdr(target_l, pred_residual_l)
            sisdr_mix = si_sdr(target_l, mixture_l)
            sisdr_halfmix = si_sdr(target_l, halfmix_l)

            target_rms = target_l.pow(2).mean(dim=-1).sqrt()
            pred_rms = pred_target_l.pow(2).mean(dim=-1).sqrt()
            target_energy = target_l.pow(2).sum(dim=-1) + 1e-8
            target_gain = (pred_target_l * target_l).sum(dim=-1) / target_energy
            residual_target_gain = (pred_residual_l * target_l).sum(dim=-1) / target_energy
            ref_residual_l = mixture_l - target_l
            ref_residual_energy = ref_residual_l.pow(2).sum(dim=-1) + 1e-8
            interferer_gain = (pred_target_l * ref_residual_l).sum(dim=-1) / ref_residual_energy
            pred_target_rms_ratio = pred_rms / (target_rms + 1e-8)

            residual_to_target_values.append(sisdr_residual.mean().item())
            mixture_to_target_values.append(sisdr_mix.mean().item())
            halfmix_to_target_values.append(sisdr_halfmix.mean().item())
            target_left_in_residual_flags.append((sisdr_residual > sisdr_target).float().mean().item())
            flow_hurts_halfmix_flags.append((sisdr_target < sisdr_halfmix).float().mean().item())
            target_rms_values.append(target_rms.mean().item())
            pred_rms_values.append(pred_rms.mean().item())
            target_gain_values.append(target_gain.mean().item())
            residual_target_gain_values.append(residual_target_gain.mean().item())
            interferer_gain_values.append(interferer_gain.abs().mean().item())
            pred_target_rms_ratio_values.append(pred_target_rms_ratio.mean().item())
            mix_err_values.append(out["mixture_error"].abs().mean().item())

            if diagnostic_batches > 0 and i < diagnostic_batches:
                ids = _batch_indices(batch, target_l.shape[0])
                max_examples = min(int(diagnostic_examples), target_l.shape[0])
                for j in range(max_examples):
                    tqdm.write(
                        "val_diag "
                        f"batch={i} "
                        f"idx={ids[j]} "
                        f"sisdr_pred_target={float(sisdr_target[j].detach().cpu()):.4f} "
                        f"sisdr_pred_residual_to_target={float(sisdr_residual[j].detach().cpu()):.4f} "
                        f"sisdr_mixture_to_target={float(sisdr_mix[j].detach().cpu()):.4f} "
                        f"sisdr_halfmix_to_target={float(sisdr_halfmix[j].detach().cpu()):.4f} "
                        f"target_left_in_residual={bool((sisdr_residual[j] > sisdr_target[j]).detach().cpu())} "
                        f"flow_hurts_halfmix={bool((sisdr_target[j] < sisdr_halfmix[j]).detach().cpu())} "
                        f"target_rms={float(target_rms[j].detach().cpu()):.6f} "
                        f"pred_rms={float(pred_rms[j].detach().cpu()):.6f} "
                        f"target_gain={float(target_gain[j].detach().cpu()):.4f} "
                        f"residual_target_gain={float(residual_target_gain[j].detach().cpu()):.4f} "
                        f"interferer_gain={float(interferer_gain[j].abs().detach().cpu()):.4f} "
                        f"pred_target_rms_ratio={float(pred_target_rms_ratio[j].detach().cpu()):.4f}"
                    )

        count += 1

    # Restore the exact mixed train/eval state that existed before validation.
    # This preserves staged freezing without globally reactivating BatchNorm in frozen modules.
    for module, training_state in module_training_states:
        module.training = training_state

    if count == 0:
        return {}

    metrics = {f"val_{k}": v / count for k, v in totals.items()}
    if sdr_values:
        metrics["val_si_sdr"] = sum(sdr_values) / len(sdr_values)
    # Dual-head metrics: use separate lists to avoid double val_ prefix
    if base_sdr_values:
        metrics["val_si_sdr_base"] = sum(base_sdr_values) / len(base_sdr_values)
    if correction_gain_values:
        metrics["val_si_sdr_correction_gain"] = sum(correction_gain_values) / len(correction_gain_values)
    if residual_to_target_values:
        metrics["val_si_sdr_residual_to_target"] = sum(residual_to_target_values) / len(residual_to_target_values)
    if mixture_to_target_values:
        metrics["val_si_sdr_mixture_to_target"] = sum(mixture_to_target_values) / len(mixture_to_target_values)
    if halfmix_to_target_values:
        metrics["val_si_sdr_halfmix_to_target"] = sum(halfmix_to_target_values) / len(halfmix_to_target_values)
    if target_left_in_residual_flags:
        metrics["val_target_left_in_residual_rate"] = 100.0 * sum(target_left_in_residual_flags) / len(target_left_in_residual_flags)
    if flow_hurts_halfmix_flags:
        metrics["val_flow_hurts_halfmix_rate"] = 100.0 * sum(flow_hurts_halfmix_flags) / len(flow_hurts_halfmix_flags)
    if target_rms_values:
        metrics["val_target_rms"] = sum(target_rms_values) / len(target_rms_values)
    if pred_rms_values:
        metrics["val_pred_rms"] = sum(pred_rms_values) / len(pred_rms_values)
    if target_gain_values:
        metrics["val_target_gain"] = sum(target_gain_values) / len(target_gain_values)
    if residual_target_gain_values:
        metrics["val_residual_target_gain"] = sum(residual_target_gain_values) / len(residual_target_gain_values)
    if interferer_gain_values:
        metrics["val_pred_interferer_gain_abs"] = sum(interferer_gain_values) / len(interferer_gain_values)
    if pred_target_rms_ratio_values:
        metrics["val_pred_target_rms_ratio"] = sum(pred_target_rms_ratio_values) / len(pred_target_rms_ratio_values)
    if mix_err_values:
        metrics["val_mixture_abs_error"] = sum(mix_err_values) / len(mix_err_values)
    return metrics


def _save_checkpoint(
    path: Path,
    model: MambaVoiceMCFlow,
    cfg: Dict[str, Any],
    *,
    step: int,
    epoch: int,
    opt: Optional[torch.optim.Optimizer] = None,
    sched: Optional[Any] = None,
    best_metric: Optional[float] = None,
    ema: Optional[TrainableEMA] = None,
    is_ema_model: bool = False,
) -> None:
    payload: Dict[str, Any] = {"model": model.state_dict(), "cfg": cfg, "step": step, "epoch": epoch}
    if opt is not None:
        payload["optimizer"] = opt.state_dict()
    if sched is not None:
        payload["scheduler"] = sched.state_dict()
    if best_metric is not None:
        payload["best_metric"] = best_metric
    if ema is not None:
        payload["ema"] = ema.state_dict()
        payload["ema_decay"] = ema.decay
    if is_ema_model:
        payload["is_ema_model"] = True
    # Write-then-rename so a kill mid-save cannot leave a truncated checkpoint
    # in place of a good one.  os.replace is atomic within a filesystem, which
    # matters most for the frequently-overwritten best.pt and for periodic
    # last.pt writes on a wall-clock-limited cluster job.
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def train(
    cfg: Dict[str, Any],
    max_steps: int | None = None,
    resume_from: Optional[str | Path] = None,
    init_from: Optional[str | Path] = None,
) -> Path:
    train_cfg = cfg.get("training", {})
    if (
        bool(train_cfg.get("require_init_from", False))
        and init_from is None
        and resume_from is None
    ):
        raise ValueError(
            "This config requires --init_from DRIFT_BEST_PT for a new run "
            "(or --resume_from for continuation)."
        )
    device = torch.device(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(train_cfg.get("out_dir", "runs/mambavoice_mcflow"))
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = train_cfg.get("seed", None)
    if seed is not None:
        seed_everything(int(seed))
        tqdm.write(f"seeded run with seed={int(seed)}")

    # Mixed precision. bf16 needs no loss scaling and is the safe default on
    # Ampere/Ada; fp16 keeps a GradScaler.
    amp_dtype_name = str(train_cfg.get("amp", "none")).lower()
    amp_dtype = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
                 "fp16": torch.float16, "float16": torch.float16,
                 "none": None, "off": None, "false": None}.get(amp_dtype_name)
    if amp_dtype_name not in {"none", "off", "false"} and amp_dtype is None:
        raise ValueError(f"training.amp must be one of bf16/fp16/none, got {amp_dtype_name!r}")
    use_amp = amp_dtype is not None and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype is torch.float16)

    # Gradient accumulation: raises the effective batch without more memory.
    accum_steps = max(1, int(train_cfg.get("grad_accum_steps", 1)))

    loader = _make_loader(
        cfg,
        "train",
        batch_size=int(train_cfg.get("batch_size", 2)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        drop_last=True,
    )

    val_every = int(train_cfg.get("val_every", 0) or 0)
    val_batches = _parse_optional_int(train_cfg.get("val_batches", None))
    val_num_steps = train_cfg.get("val_num_steps", cfg.get("flow", {}).get("num_steps", None))
    val_num_steps = None if val_num_steps is None else int(val_num_steps)
    val_diagnostic_batches = int(train_cfg.get("val_diagnostic_batches", 0) or 0)
    val_diagnostic_examples = int(train_cfg.get("val_diagnostic_examples", 4) or 4)
    save_best = bool(train_cfg.get("save_best", True))
    best_metric_name = str(train_cfg.get("best_metric", "val_si_sdr"))
    best_mode = str(train_cfg.get("best_mode", "max"))
    show_progress = bool(train_cfg.get("progress", True))
    show_val_progress = bool(train_cfg.get("val_progress", True))
    ema_decay = train_cfg.get("ema_decay", None)
    ema_enabled = ema_decay is not None and float(ema_decay) > 0.0
    ema_eval = bool(train_cfg.get("ema_eval", ema_enabled))
    save_ema_best = bool(train_cfg.get("save_ema_best", ema_enabled))
    save_ema_last = bool(train_cfg.get("save_ema_last", ema_enabled))
    # Periodic last.pt writes.  0 (default) keeps the historical behaviour of
    # only writing last.pt when the loop finishes or hits max_steps -- which
    # means a SLURM wall-clock kill leaves no last.pt at all.  Setting this to
    # the val_every cadence bounds the loss from a kill to that many steps.
    save_every = int(train_cfg.get("save_every", 0) or 0)
    if save_every < 0:
        raise ValueError("training.save_every must be >= 0")
    ema: Optional[TrainableEMA] = None

    val_loader = None
    if val_every > 0:
        val_split = str(train_cfg.get("val_split", "val"))
        val_loader = _make_loader(
            cfg,
            val_split,
            batch_size=int(train_cfg.get("val_batch_size", 1)),
            shuffle=False,
            num_workers=int(train_cfg.get("val_num_workers", 0)),
            drop_last=False,
        )
        val_len = _safe_len(val_loader)
        batch_text = "all" if val_batches is None else str(val_batches)
        tqdm.write(
            f"validation enabled: split={val_split} every_steps={val_every} batches={batch_text} "
            f"loader_len={val_len if val_len is not None else 'unknown'}"
        )

    model = MambaVoiceMCFlow(cfg).to(device)

    # --- CRITICAL: Initialize Lazy Layers Before Optimizer Setup ---
    init_batch = next(iter(loader))
    init_batch = _move_batch_to_device(init_batch, device)
    with torch.no_grad():
        _ = model.training_loss(init_batch)
    del init_batch

    # The flow-map objective freezes a warm-started drift model.  Its
    # conditioner contains LazyLinear parameters, which only become freezable
    # after the initialization batch above has run.
    if model.spec_objective == "flowmap_adapter":
        model._freeze_for_flowmap_adapter()
        if getattr(model, "_flowmap_freeze_pending", False):
            raise RuntimeError(
                "flowmap adapter initialization left lazy base parameters "
                "unmaterialized; check that the initialization batch contains "
                "both audio and visual inputs"
            )

    if bool(train_cfg.get("log_feature_trainability", True)):
        _log_feature_extractor_trainability(
            model,
            assert_trainable=bool(train_cfg.get("assert_feature_extractor_trainable", False)),
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("Model has no trainable parameters")
    opt = torch.optim.AdamW(
        trainable_params,
        lr=float(train_cfg.get("lr", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    epochs = int(train_cfg.get("epochs", 1))
    batches_per_epoch = _safe_len(loader)
    # One optimizer step consumes accum_steps batches.
    steps_per_epoch = None if batches_per_epoch is None else batches_per_epoch // accum_steps
    if max_steps is not None:
        total_steps = max_steps
    elif steps_per_epoch is not None:
        total_steps = steps_per_epoch * epochs
    else:
        total_steps = None
    if accum_steps > 1:
        tqdm.write(
            f"grad accumulation: {accum_steps} x batch_size={int(train_cfg.get('batch_size', 2))} "
            f"-> effective batch {accum_steps * int(train_cfg.get('batch_size', 2))}"
        )
    if use_amp:
        tqdm.write(f"mixed precision enabled: {amp_dtype}")

    # Linear warmup followed by cosine decay.  Warmup is optional so legacy
    # configs retain their previous behavior.
    scheduler_total_steps = total_steps if total_steps is not None else 1_000_000
    warmup_steps_cfg = train_cfg.get("warmup_steps", None)
    if warmup_steps_cfg is not None:
        warmup_steps = max(0, int(warmup_steps_cfg))
    else:
        warmup_frac = max(0.0, float(train_cfg.get("warmup_frac", 0.0)))
        warmup_steps = int(round(scheduler_total_steps * warmup_frac))
    warmup_steps = min(warmup_steps, max(0, scheduler_total_steps - 1))
    eta_min = float(train_cfg.get("min_lr", 1e-6))
    if warmup_steps > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt,
            start_factor=float(train_cfg.get("warmup_start_factor", 1e-3)),
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=max(1, scheduler_total_steps - warmup_steps),
            eta_min=eta_min,
        )
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )
        tqdm.write(
            f"lr schedule: linear warmup {warmup_steps} steps -> cosine, eta_min={eta_min:.2e}"
        )
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=scheduler_total_steps,
            eta_min=eta_min,
        )

    start_epoch = 0
    step = 0
    best_metric: Optional[float] = None

    # Weight-only initialization is intentionally distinct from resume.  This is
    # the correct way to warm-start a new objective (e.g. residual flow ->
    # AlphaFlow): network parameters are reused but AdamW moments, scheduler,
    # epoch, step and best metric all restart cleanly.
    if init_from is not None:
        init_path = Path(init_from)
        if not init_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found to initialize from: {init_path}")
        tqdm.write(f"Initializing model weights only from: {init_path}")
        init_blob = torch.load(init_path, map_location=device)
        incompatible = model.load_state_dict(init_blob["model"], strict=False)
        # A frozen flow.prior is loaded from its own checkpoint when the model is
        # built, and an INIT_FROM checkpoint is not expected to carry it.
        # Filtering its keys keeps this warning about the keys that matter.
        missing = [k for k in incompatible.missing_keys if not _is_frozen_prior_param(k)]
        unexpected = list(incompatible.unexpected_keys)
        if model.spec_objective == "flowmap_adapter":
            source_cfg = init_blob.get("cfg", {})
            source_flow = source_cfg.get("flow", {}) or {}
            source_objective = str(source_flow.get("objective", "")).lower()
            if source_objective not in {
                "drift", "drift_only", "one_step", "one_step_drift"
            }:
                raise ValueError(
                    "flowmap_adapter must initialize from a drift-only "
                    f"checkpoint, got objective={source_objective!r}"
                )
            for section in ("backbone", "model", "stft"):
                if source_cfg.get(section) != cfg.get(section):
                    raise RuntimeError(
                        f"Drift checkpoint {section!r} config does not match the "
                        "flow-map config; exact step-zero parity is impossible"
                    )
            current_head = dict(cfg.get("head", {}))
            for adapter_key in (
                "interval_embedding_reference",
                "flowmap_adapter_blocks",
                "flowmap_adapter_channels",
                "flowmap_adapter_dropout",
            ):
                current_head.pop(adapter_key, None)
            if source_cfg.get("head") != current_head:
                raise RuntimeError(
                    "Drift checkpoint head config differs from the frozen base "
                    "defined by the flow-map config"
                )
            current_flow = cfg.get("flow", {}) or {}
            for flow_key in (
                "target_mode", "init_mode", "visual_cross_attention",
                "adaptive_drift", "consistency",
            ):
                if source_flow.get(flow_key) != current_flow.get(flow_key):
                    raise RuntimeError(
                        f"Drift checkpoint flow.{flow_key} differs from the "
                        "flow-map config; exact initialization is not guaranteed"
                    )
            bad_missing = [
                key for key in missing
                if not key.startswith("head.flowmap_adapter.")
            ]
            if bad_missing or unexpected:
                raise RuntimeError(
                    "Drift checkpoint is not architecture-compatible with the "
                    "flow-map config. Only head.flowmap_adapter.* may be missing; "
                    f"bad_missing={bad_missing[:10]} unexpected={unexpected[:10]}"
                )
            adapter_output = model.head.flowmap_adapter.output
            if (
                torch.count_nonzero(adapter_output.weight).item() != 0
                or torch.count_nonzero(adapter_output.bias).item() != 0
            ):
                raise RuntimeError(
                    "Flow-map adapter lost its zero-output initialization while "
                    "loading the drift checkpoint"
                )
            model._freeze_for_flowmap_adapter()
            tqdm.write(
                "flowmap init verified: drift weights loaded; only the "
                "zero-output correction adapter is trainable"
            )
        if model.spec_objective == "visual_floss":
            source_cfg = init_blob.get("cfg", {})
            source_flow = source_cfg.get("flow", {}) or {}
            source_objective = str(source_flow.get("objective", "")).lower()
            if source_objective not in {
                "drift", "drift_only", "one_step", "one_step_drift"
            }:
                raise ValueError(
                    "visual_floss must initialize from a drift-only checkpoint, "
                    f"got objective={source_objective!r}"
                )
            for section in ("backbone", "model", "stft", "head"):
                if source_cfg.get(section) != cfg.get(section):
                    raise RuntimeError(
                        f"Drift checkpoint {section!r} config does not match the "
                        "Visual-FLOSS config; exact step-zero parity is impossible"
                    )
            current_flow = cfg.get("flow", {}) or {}
            for flow_key in (
                "target_mode", "init_mode", "visual_cross_attention",
                "adaptive_drift", "consistency",
            ):
                if source_flow.get(flow_key) != current_flow.get(flow_key):
                    raise RuntimeError(
                        f"Drift checkpoint flow.{flow_key} differs from the "
                        "Visual-FLOSS config; exact initialization is not guaranteed"
                    )
            if missing or unexpected:
                raise RuntimeError(
                    "Visual-FLOSS uses the exact drift architecture, so INIT_FROM "
                    "must load with no missing or unexpected model keys; "
                    f"missing={missing[:10]} unexpected={unexpected[:10]}"
                )
            tqdm.write(
                "visual-floss init verified: exact drift architecture and weights "
                "loaded; deployment starts at the original one-step predictor"
            )
        if missing or unexpected:
            tqdm.write(
                f"INIT_FROM non-strict load: missing={len(missing)} unexpected={len(unexpected)} "
                f"missing_head={missing[:5]} unexpected_head={unexpected[:5]}"
            )
        tqdm.write("--> Weight-only initialization complete; optimizer/scheduler start fresh.")

    # --- Handle Resume Payload Parsing ---
    if resume_from is not None and init_from is not None:
        raise ValueError("Use only one of resume_from or init_from")
    if resume_from is not None:
        resume_path = Path(resume_from)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found to resume: {resume_path}")
        
        tqdm.write(f"Resuming execution from checkpoint payload: {resume_path}")
        checkpoint_data = torch.load(resume_path, map_location=device)

        # `best.pt` is written with EMA weights swapped in and opt=None.  Resuming
        # from it restarts SGD from an averaged point with zeroed AdamW moments,
        # which reliably produces a large transient dip (and has been observed as
        # negative validation SI-SDR after a resume).  `best_raw.pt` / `last.pt`
        # carry the raw weights and the optimizer state and are the correct
        # checkpoints to continue from.
        if checkpoint_data.get("is_ema_model", False):
            tqdm.write(
                "WARNING: this checkpoint holds EMA weights (is_ema_model=True). "
                "Resuming training from it discards the raw weight trajectory. "
                "Prefer best_raw.pt or last.pt for continuation; use this file "
                "for evaluation only."
            )
        if "optimizer" not in checkpoint_data:
            tqdm.write(
                "WARNING: no optimizer state in checkpoint. AdamW moments restart "
                "at zero, which usually costs a few thousand steps of quality."
            )

        model.load_state_dict(checkpoint_data["model"])
        if "optimizer" in checkpoint_data and opt is not None:
            opt.load_state_dict(checkpoint_data["optimizer"])

        step = checkpoint_data.get("step", 0)
        start_epoch = checkpoint_data.get("epoch", 0)
        best_metric = checkpoint_data.get("best_metric", None)
        
        # Restore scheduler history
        if "scheduler" in checkpoint_data:
            sched.load_state_dict(checkpoint_data["scheduler"])
            tqdm.write("--> Scheduler state successfully restored from checkpoint payload.")
        else:
            tqdm.write(f"--> Warning: No scheduler state in checkpoint. Fast-forwarding scheduler to step {step}.")
            for _ in range(step):
                sched.step()
        
        if ema_enabled and "ema" in checkpoint_data:
            ema = TrainableEMA(model, decay=float(ema_decay))
            ema.load_state_dict(checkpoint_data["ema"], device=device)
        
        tqdm.write(f"Resumed successfully at Epoch={start_epoch}, Global Step={step}")

    if ema_enabled and ema is None:
        tqdm.write(f"EMA enabled: decay={float(ema_decay):.6f} eval_with_ema={ema_eval}")
        ema = TrainableEMA(model, decay=float(ema_decay))

    log_every = int(train_cfg.get("log_every", 10))
    grad_clip = float(train_cfg.get("grad_clip", 5.0))
    interference_curriculum = train_cfg.get("interference_curriculum", None)

    def apply_interference_curriculum(epoch_idx: int) -> None:
        if not interference_curriculum:
            return
        # loader.dataset is VovitTupleAdapter, which holds the real dataset one
        # level down in .dataset and defines no __getattr__ forwarding.  Checking
        # the wrapper directly made this function a silent no-op: every configured
        # curriculum was ignored for the whole run.  Walk the wrapper chain, with
        # a bound so a self-referential .dataset cannot spin forever.
        dataset = getattr(loader, "dataset", None)
        for _ in range(8):
            if dataset is None or hasattr(dataset, "interference_prob"):
                break
            nxt = getattr(dataset, "dataset", None)
            if nxt is dataset:
                break
            dataset = nxt
        if dataset is None or not hasattr(dataset, "interference_prob"):
            tqdm.write(
                "interference curriculum: configured but no dataset in the loader chain "
                "exposes interference_prob -- curriculum IGNORED"
            )
            return
        chosen = None
        for stage in interference_curriculum:
            if isinstance(stage, dict):
                stage_epoch = int(stage.get("epoch", 0))
                stage_prob = float(stage.get("prob", stage.get("interference_prob", 1.0)))
            else:
                stage_epoch = int(stage[0])
                stage_prob = float(stage[1])
            if epoch_idx >= stage_epoch:
                chosen = stage_prob
        if chosen is not None:
            chosen = max(0.0, min(1.0, chosen))
            old = float(dataset.interference_prob)
            dataset.interference_prob = chosen
            if abs(old - chosen) > 1e-12:
                tqdm.write(
                    f"interference curriculum: epoch={epoch_idx + 1} probability {old:.3f}->{chosen:.3f}"
                )

    pbar = tqdm(total=total_steps, desc="train", disable=not show_progress)
    if step > 0 and pbar is not None:
        pbar.update(step)

    def write_last(epoch_value: int) -> Path:
        """Write last.pt (and last_ema.pt), overwriting any previous pair.

        Shared by the periodic save, the max_steps exit and the normal end of
        training so the three cannot drift apart.  Overwriting keeps disk use
        flat at two files regardless of save_every.
        """
        ckpt_path = out_dir / "last.pt"
        _save_checkpoint(
            ckpt_path, model, cfg, step=step, epoch=epoch_value,
            opt=opt, sched=sched, best_metric=best_metric, ema=ema,
        )
        if ema is not None and save_ema_last:
            _ema_path = out_dir / "last_ema.pt"
            with ema.apply_to(model):
                _save_checkpoint(
                    _ema_path, model, cfg, step=step, epoch=epoch_value,
                    opt=None, sched=sched, best_metric=best_metric, ema=ema,
                    is_ema_model=True,
                )
        return ckpt_path

    def is_better(value: float, best: Optional[float]) -> bool:
        if best is None:
            return True
        if best_mode == "min":
            return value < best
        return value > best

    def maybe_validate(epoch_idx: int, *, label: Optional[str] = None, update_best: bool = True) -> None:
        nonlocal best_metric
        if val_loader is None:
            return
        eval_context = ema.apply_to(model) if (ema is not None and ema_eval) else contextlib.nullcontext()
        with eval_context:
            metrics = validate(
                model,
                val_loader,
                device,
                num_batches=val_batches,
                num_steps=val_num_steps,
                show_progress=show_val_progress,
                desc=f"val@{step}{'_ema' if (ema is not None and ema_eval) else ''}",
                diagnostic_batches=val_diagnostic_batches,
                diagnostic_examples=val_diagnostic_examples,
            )
        prefix = f"{label} " if label else ""
        if not metrics:
            tqdm.write(f"{prefix}epoch={epoch_idx + 1}/{epochs} step={step} val_empty=true")
            return
        msg = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        tqdm.write(f"{prefix}epoch={epoch_idx + 1}/{epochs} step={step} {msg}")

        if not update_best:
            return
        metric = metrics.get(best_metric_name)
        if save_best and metric is not None and is_better(metric, best_metric):
            best_metric = metric
            best_path = out_dir / "best.pt"
            if ema is not None and ema_eval and save_ema_best:
                with ema.apply_to(model):
                    _save_checkpoint(best_path, model, cfg, step=step, epoch=epoch_idx + 1, opt=None, sched=sched, best_metric=best_metric, ema=ema, is_ema_model=True)
                raw_best_path = out_dir / "best_raw.pt"
                _save_checkpoint(raw_best_path, model, cfg, step=step, epoch=epoch_idx + 1, opt=opt, sched=sched, best_metric=best_metric, ema=ema)
                tqdm.write(f"saved_best={best_path} raw={raw_best_path} {best_metric_name}={best_metric:.4f}")
            else:
                _save_checkpoint(best_path, model, cfg, step=step, epoch=epoch_idx + 1, opt=opt, sched=sched, best_metric=best_metric, ema=ema)
                tqdm.write(f"saved_best={best_path} {best_metric_name}={best_metric:.4f}")

    # Optional: score the model once before any update.  For an INIT_FROM warm
    # start this puts the starting point in the log, so a drop caused by
    # converting a checkpoint to a new objective is never mistaken for a
    # training effect.  Fresh runs only (a resume already has its history), and
    # logged only: best.pt and best_metric are left alone.
    if bool(train_cfg.get("validate_at_start", False)) and val_loader is not None and resume_from is None:
        maybe_validate(start_epoch - 1, label="val_at_start", update_best=True)

    try:
        # Start training loop exactly from the checkpoint epoch index
        for epoch_idx in range(start_epoch, epochs):
            model.train()
            # Apply staged unfreezing for dual-head training if configured.
            stage_unfreezes = train_cfg.get("stage_unfreezes")
            if stage_unfreezes:
                # Store for later use in validation (restore frozen eval mode)
                model._stage_config = stage_unfreezes
                model._current_epoch = epoch_idx
                model.set_training_stage(epoch_idx, stage_unfreezes)
            apply_interference_curriculum(epoch_idx)
            opt.zero_grad(set_to_none=True)
            for micro, batch in enumerate(loader):
                batch = _move_batch_to_device(batch, device)
                model.set_training_progress(step, total_steps)
                if use_amp:
                    with torch.autocast(device_type=device.type, dtype=amp_dtype):
                        losses = model.training_loss(batch)
                else:
                    losses = model.training_loss(batch)

                # Scale so the accumulated gradient matches a single large batch.
                scaler.scale(losses["loss"] / accum_steps).backward()

                if (micro + 1) % accum_steps != 0:
                    continue

                scaler.unscale_(opt)
                # Clip only the parameters the optimizer actually updates.
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

                # Step learning rate scheduler each optimizer step
                sched.step()

                if ema is not None:
                    ema.update(model)

                if step % log_every == 0:
                    current_lr = opt.param_groups[0]['lr']
                    msg = " ".join(f"{k}={float(v.detach()):.4f}" for k, v in losses.items())
                    tqdm.write(
                        f"epoch={epoch_idx + 1}/{epochs} step={step} lr={current_lr:.2e} "
                        f"grad_norm={float(grad_norm):.4f} {msg}"
                    )

                step += 1
                if pbar is not None:
                    pbar.update(1)

                if val_loader is not None and step % val_every == 0:
                    maybe_validate(epoch_idx)

                if save_every and step % save_every == 0:
                    saved = write_last(epoch_idx + 1)
                    tqdm.write(f"epoch={epoch_idx + 1}/{epochs} step={step} saved_last={saved}")

                if max_steps is not None and step >= max_steps:
                    return write_last(epoch_idx + 1)

            if bool(train_cfg.get("val_at_epoch_end", False)) and val_loader is not None:
                maybe_validate(epoch_idx)
    finally:
        if pbar is not None:
            pbar.close()

    ckpt = write_last(epochs)
    return ckpt


def load_model_from_checkpoint(
    checkpoint: str,
    map_location: str | torch.device = "cpu",
    *,
    use_ema: bool = False,
) -> MambaVoiceMCFlow:
    blob = torch.load(checkpoint, map_location=map_location)
    model = MambaVoiceMCFlow(blob["cfg"])
    # strict=False is needed because heads gained parameters over time, but a
    # silent partial load means evaluating a partly randomly-initialised model
    # and reporting the number as if it were the trained one. Always say what
    # did not match.
    incompatible = model.load_state_dict(blob["model"], strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        print(
            f"WARNING: checkpoint {checkpoint!r} did not match the model exactly.\n"
            f"  {len(missing)} missing key(s) (left at initialisation): {missing[:10]}"
            f"{' ...' if len(missing) > 10 else ''}\n"
            f"  {len(unexpected)} unexpected key(s) (ignored): {unexpected[:10]}"
            f"{' ...' if len(unexpected) > 10 else ''}"
        )
    if use_ema:
        if "ema" not in blob:
            raise KeyError(f"Checkpoint {checkpoint!r} does not contain EMA weights")
        _load_ema_weights_into_model(model, blob["ema"])
    return model
