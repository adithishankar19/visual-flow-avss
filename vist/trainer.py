from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from vist.data import build_dataset, collate_dict
from vist.losses import si_sdr
from vist.model import VIST


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    """Seed python/numpy/torch RNGs.

    This does not by itself make the dataset deterministic: the Acappella
    dataset draws its interferer, accompaniment and gains from the global
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


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
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
    return DataLoader(
        build_dataset(cfg, split),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_dict,
        drop_last=drop_last,
    )


def parse_optional_int(value: Any) -> Optional[int]:
    """Parse config values where null/all means no limit."""
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().lower() in {"all", "full", "none", "null"}:
            return None
        return int(value)
    value = int(value)
    return None if value <= 0 else value


def _safe_len(obj: Any) -> Optional[int]:
    try:
        return len(obj)
    except Exception:
        return None


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
    model: VIST,
    loader: DataLoader,
    device: torch.device,
    *,
    num_batches: Optional[int] = None,
    num_steps: Optional[int] = None,
    show_progress: bool = True,
    desc: str = "val",
) -> Dict[str, float]:
    was_training = model.training
    model.eval()

    totals: Dict[str, float] = {}
    count = 0
    si_sdr_values = []
    remainder_si_sdr_values = []
    swap_flags = []

    total = _safe_len(loader)
    if num_batches is not None and total is not None:
        total = min(total, num_batches)
    for i, batch in enumerate(tqdm(loader, total=total, desc=desc, leave=False, disable=not show_progress)):
        if num_batches is not None and i >= num_batches:
            break
        batch = move_batch_to_device(batch, device)

        for k, v in model.training_loss(batch).items():
            totals[k] = totals.get(k, 0.0) + float(v.detach())

        out = model.separate(batch["mixture"], face=batch.get("face"), num_steps=num_steps)
        length = min(out["target"].shape[-1], batch["target"].shape[-1])
        target = batch["target"][..., :length]
        estimate = si_sdr(target, out["target"][..., :length])
        remainder = si_sdr(target, out["residual"][..., :length])
        si_sdr_values.append(estimate.mean().item())
        remainder_si_sdr_values.append(remainder.mean().item())
        swap_flags.append((remainder > estimate).float().mean().item())
        count += 1

    model.train(was_training)
    if count == 0:
        return {}
    metrics = {f"val_{k}": v / count for k, v in totals.items()}
    metrics["val_si_sdr"] = sum(si_sdr_values) / count
    metrics["val_si_sdr_remainder_to_target"] = sum(remainder_si_sdr_values) / count
    metrics["val_swap_rate"] = 100.0 * sum(swap_flags) / count
    return metrics


def _save_checkpoint(
    path: Path,
    model: VIST,
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
    # Write-then-rename so a kill mid-save cannot leave a truncated checkpoint.
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def train(
    cfg: Dict[str, Any],
    max_steps: int | None = None,
    resume_from: Optional[str | Path] = None,
) -> Path:
    train_cfg = cfg.get("training", {})
    device = torch.device(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(train_cfg.get("out_dir", "runs/vist"))
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = train_cfg.get("seed", None)
    if seed is not None:
        seed_everything(int(seed))
        tqdm.write(f"seeded run with seed={int(seed)}")

    # Mixed precision. bf16 needs no loss scaling; fp16 keeps a GradScaler.
    amp_dtype_name = str(train_cfg.get("amp", "none")).lower()
    amp_dtype = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
                 "fp16": torch.float16, "float16": torch.float16,
                 "none": None, "off": None, "false": None}.get(amp_dtype_name)
    if amp_dtype_name not in {"none", "off", "false"} and amp_dtype is None:
        raise ValueError(f"training.amp must be one of bf16/fp16/none, got {amp_dtype_name!r}")
    use_amp = amp_dtype is not None and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype is torch.float16)

    accum_steps = max(1, int(train_cfg.get("grad_accum_steps", 1)))
    batch_size = int(train_cfg.get("batch_size", 2))
    loader = _make_loader(
        cfg,
        "train",
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        drop_last=True,
    )

    val_every = int(train_cfg.get("val_every", 0) or 0)
    val_batches = parse_optional_int(train_cfg.get("val_batches", None))
    val_num_steps = int(train_cfg.get("val_num_steps", 1))
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
    # Periodic last.pt writes bound what a wall-clock kill can lose.
    save_every = int(train_cfg.get("save_every", 0) or 0)
    if save_every < 0:
        raise ValueError("training.save_every must be >= 0")
    ema: Optional[TrainableEMA] = None

    val_loader = None
    if val_every > 0:
        val_loader = _make_loader(
            cfg,
            "val",
            batch_size=int(train_cfg.get("val_batch_size", 1)),
            shuffle=False,
            num_workers=int(train_cfg.get("val_num_workers", 0)),
            drop_last=False,
        )
        tqdm.write(
            f"validation enabled: every_steps={val_every} "
            f"batches={'all' if val_batches is None else val_batches} "
            f"loader_len={_safe_len(val_loader)}"
        )

    model = VIST(cfg).to(device)

    # The conditioner uses LazyLinear projections; materialise them before the
    # optimizer collects parameters.
    init_batch = move_batch_to_device(next(iter(loader)), device)
    with torch.no_grad():
        model.training_loss(init_batch)
    del init_batch

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    tqdm.write(f"trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    opt = torch.optim.AdamW(
        trainable_params,
        lr=float(train_cfg.get("lr", 3e-5)),
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
    tqdm.write(f"effective batch size: {accum_steps * batch_size} ({batch_size} x {accum_steps} accumulation)")
    if use_amp:
        tqdm.write(f"mixed precision enabled: {amp_dtype}")

    # Linear warm-up followed by cosine decay.
    scheduler_total_steps = total_steps if total_steps is not None else 1_000_000
    warmup_steps = min(max(0, int(train_cfg.get("warmup_steps", 0))), max(0, scheduler_total_steps - 1))
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
        tqdm.write(f"lr schedule: linear warmup {warmup_steps} steps -> cosine, eta_min={eta_min:.2e}")
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=scheduler_total_steps, eta_min=eta_min)

    start_epoch = 0
    step = 0
    best_metric: Optional[float] = None

    if resume_from is not None:
        resume_path = Path(resume_from)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found to resume: {resume_path}")
        tqdm.write(f"Resuming from checkpoint: {resume_path}")
        checkpoint_data = torch.load(resume_path, map_location=device)

        # best.pt holds EMA weights and no optimizer state; resuming from it
        # restarts from an averaged point with zeroed AdamW moments.
        if checkpoint_data.get("is_ema_model", False):
            tqdm.write(
                "WARNING: this checkpoint holds EMA weights (is_ema_model=True). "
                "Prefer best_raw.pt or last.pt for continuation; use this file "
                "for evaluation only."
            )
        if "optimizer" not in checkpoint_data:
            tqdm.write("WARNING: no optimizer state in checkpoint; AdamW moments restart at zero.")

        model.load_state_dict(checkpoint_data["model"])
        if "optimizer" in checkpoint_data:
            opt.load_state_dict(checkpoint_data["optimizer"])

        step = checkpoint_data.get("step", 0)
        start_epoch = checkpoint_data.get("epoch", 0)
        best_metric = checkpoint_data.get("best_metric", None)
        if "scheduler" in checkpoint_data:
            sched.load_state_dict(checkpoint_data["scheduler"])
        else:
            tqdm.write(f"WARNING: no scheduler state in checkpoint; fast-forwarding scheduler to step {step}.")
            for _ in range(step):
                sched.step()
        if ema_enabled and "ema" in checkpoint_data:
            ema = TrainableEMA(model, decay=float(ema_decay))
            ema.load_state_dict(checkpoint_data["ema"], device=device)
        tqdm.write(f"Resumed at epoch={start_epoch}, step={step}")

    if ema_enabled and ema is None:
        tqdm.write(f"EMA enabled: decay={float(ema_decay):.6f} eval_with_ema={ema_eval}")
        ema = TrainableEMA(model, decay=float(ema_decay))

    log_every = int(train_cfg.get("log_every", 10))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    pbar = tqdm(total=total_steps, desc="train", disable=not show_progress)
    if step > 0:
        pbar.update(step)

    def write_last(epoch_value: int) -> Path:
        """Write last.pt (and last_ema.pt), overwriting any previous pair."""
        ckpt_path = out_dir / "last.pt"
        _save_checkpoint(
            ckpt_path, model, cfg, step=step, epoch=epoch_value,
            opt=opt, sched=sched, best_metric=best_metric, ema=ema,
        )
        if ema is not None and save_ema_last:
            with ema.apply_to(model):
                _save_checkpoint(
                    out_dir / "last_ema.pt", model, cfg, step=step, epoch=epoch_value,
                    opt=None, sched=sched, best_metric=best_metric, ema=ema,
                    is_ema_model=True,
                )
        return ckpt_path

    def is_better(value: float, best: Optional[float]) -> bool:
        if best is None:
            return True
        return value < best if best_mode == "min" else value > best

    def maybe_validate(epoch_idx: int) -> None:
        nonlocal best_metric
        if val_loader is None:
            return
        use_ema = ema is not None and ema_eval
        with ema.apply_to(model) if use_ema else contextlib.nullcontext():
            metrics = validate(
                model,
                val_loader,
                device,
                num_batches=val_batches,
                num_steps=val_num_steps,
                show_progress=show_val_progress,
                desc=f"val@{step}{'_ema' if use_ema else ''}",
            )
        if not metrics:
            tqdm.write(f"epoch={epoch_idx + 1}/{epochs} step={step} val_empty=true")
            return
        tqdm.write(f"epoch={epoch_idx + 1}/{epochs} step={step} " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        metric = metrics.get(best_metric_name)
        if save_best and metric is not None and is_better(metric, best_metric):
            best_metric = metric
            best_path = out_dir / "best.pt"
            if use_ema and save_ema_best:
                with ema.apply_to(model):
                    _save_checkpoint(best_path, model, cfg, step=step, epoch=epoch_idx + 1, opt=None, sched=sched, best_metric=best_metric, ema=ema, is_ema_model=True)
                raw_best_path = out_dir / "best_raw.pt"
                _save_checkpoint(raw_best_path, model, cfg, step=step, epoch=epoch_idx + 1, opt=opt, sched=sched, best_metric=best_metric, ema=ema)
                tqdm.write(f"saved_best={best_path} raw={raw_best_path} {best_metric_name}={best_metric:.4f}")
            else:
                _save_checkpoint(best_path, model, cfg, step=step, epoch=epoch_idx + 1, opt=opt, sched=sched, best_metric=best_metric, ema=ema)
                tqdm.write(f"saved_best={best_path} {best_metric_name}={best_metric:.4f}")

    try:
        for epoch_idx in range(start_epoch, epochs):
            model.train()
            opt.zero_grad(set_to_none=True)
            for micro, batch in enumerate(loader):
                batch = move_batch_to_device(batch, device)
                model.set_training_progress(step, total_steps)
                with torch.autocast(device_type=device.type, dtype=amp_dtype) if use_amp else contextlib.nullcontext():
                    losses = model.training_loss(batch)

                # Scale so the accumulated gradient matches a single large batch.
                scaler.scale(losses["loss"] / accum_steps).backward()
                if (micro + 1) % accum_steps != 0:
                    continue

                scaler.unscale_(opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                if ema is not None:
                    ema.update(model)

                if step % log_every == 0:
                    msg = " ".join(f"{k}={float(v.detach()):.4f}" for k, v in losses.items())
                    tqdm.write(
                        f"epoch={epoch_idx + 1}/{epochs} step={step} lr={opt.param_groups[0]['lr']:.2e} "
                        f"grad_norm={float(grad_norm):.4f} {msg}"
                    )

                step += 1
                pbar.update(1)

                if val_loader is not None and step % val_every == 0:
                    maybe_validate(epoch_idx)
                if save_every and step % save_every == 0:
                    tqdm.write(f"epoch={epoch_idx + 1}/{epochs} step={step} saved_last={write_last(epoch_idx + 1)}")
                if max_steps is not None and step >= max_steps:
                    return write_last(epoch_idx + 1)
    finally:
        pbar.close()

    return write_last(epochs)


def load_model_from_checkpoint(
    checkpoint: str,
    map_location: str | torch.device = "cpu",
    *,
    use_ema: bool = False,
) -> VIST:
    """Build VIST from a checkpoint's stored config and load its weights strictly."""
    blob = torch.load(checkpoint, map_location=map_location)
    model = VIST(blob["cfg"])
    model.load_state_dict(blob["model"], strict=True)
    if use_ema:
        if "ema" not in blob:
            raise KeyError(f"Checkpoint {checkpoint!r} does not contain EMA weights")
        _load_ema_weights_into_model(model, blob["ema"])
    return model
