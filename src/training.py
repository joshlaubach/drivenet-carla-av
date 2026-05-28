"""Supervised training loop with early stopping and mixed-precision support."""

from __future__ import annotations

import copy
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    augmenter: Any | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    """Train for one epoch with optional mixed-precision. Returns mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    use_amp = scaler is not None

    for images, states, actions, meta in loader:
        images = images.to(device, non_blocking=True)
        states = states.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        meta = meta.to(device, non_blocking=True)

        if augmenter is not None:
            images, actions = augmenter(images, actions)

        with torch.amp.autocast("cuda", enabled=use_amp):
            preds = model(images, states, meta)
            loss = criterion(preds, actions)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    """Evaluate model and return (combined_loss, per_target_mse)."""
    model.eval()
    total_loss = 0.0
    per_target_se = np.zeros(3)
    n_samples = 0
    n_batches = 0
    for images, states, actions, meta in loader:
        images = images.to(device, non_blocking=True)
        states = states.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        meta = meta.to(device, non_blocking=True)

        preds = model(images, states, meta)
        loss = criterion(preds, actions)
        total_loss += loss.item()
        n_batches += 1

        diff = (preds - actions).cpu().numpy()
        per_target_se += (diff ** 2).sum(axis=0)
        n_samples += len(actions)

    per_target_mse = per_target_se / n_samples
    return total_loss / n_batches, per_target_mse


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str,
    save_dir: str,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    augmenter: Any | None = None,
    max_epochs: int = 200,
    early_stop_patience: int = 7,
    eval_criterion: nn.Module | None = None,
) -> dict[str, Any]:
    """Full training loop with early stopping and mixed-precision. Returns history dict."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_path / f"{model_name}_best.pt"

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    _eval_criterion = eval_criterion if eval_criterion is not None else criterion

    best_val_loss = float("inf")
    best_state_dict = None
    patience_counter = 0
    history: dict[str, Any] = {
        "train_losses": [], "val_losses": [], "lr_history": [],
        "val_steer_mse": [], "val_throttle_mse": [], "val_brake_mse": [],
        "best_epoch": 0, "best_val_loss": float("inf"), "epochs_trained": 0,
    }

    epoch_width = len(str(max_epochs))
    t0 = time.time()
    for epoch in range(max_epochs):
        lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, augmenter,
            scaler=scaler,
        )
        val_loss, val_per_target = evaluate(
            model, val_loader, _eval_criterion, device,
        )
        scheduler.step(val_loss)

        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)
        history["lr_history"].append(lr)
        history["val_steer_mse"].append(float(val_per_target[0]))
        history["val_throttle_mse"].append(float(val_per_target[1]))
        history["val_brake_mse"].append(float(val_per_target[2]))

        is_best = not (val_loss != val_loss) and val_loss < best_val_loss  # False if NaN
        marker = " *" if is_best else ""
        print(
            f"Epoch {epoch + 1:>{epoch_width}}/{max_epochs}  "
            f"train={train_loss:.6f}  val={val_loss:.6f}  "
            f"[s={val_per_target[0]:.4f} t={val_per_target[1]:.4f} "
            f"b={val_per_target[2]:.4f}]  lr={lr:.2e}{marker}",
            flush=True,
        )

        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            best_state_dict = copy.deepcopy(model.state_dict())
            torch.save(best_state_dict, checkpoint_path)
            history["best_epoch"] = epoch + 1
            history["best_val_loss"] = best_val_loss
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch + 1}", flush=True)
            break

    elapsed = time.time() - t0
    history["epochs_trained"] = epoch + 1
    print(
        f"Training complete: {history['epochs_trained']} epochs in "
        f"{elapsed / 60:.1f} min, best val={history['best_val_loss']:.6f} "
        f"at epoch {history['best_epoch']}",
        flush=True,
    )
    print(f"Checkpoint saved: {checkpoint_path}", flush=True)

    if history["best_epoch"] == history["epochs_trained"]:
        log.warning(
            "best_epoch == epochs_trained (%d). Model may not have converged "
            "-- consider increasing max_epochs.", history["epochs_trained"],
        )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    else:
        log.warning("No improvement was recorded during training — model weights unchanged.")
    return history


@torch.no_grad()
def predict_all(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Run inference on all samples. Returns dict of predictions, targets, meta."""
    model.eval()
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_meta: list[np.ndarray] = []
    for images, states, actions, meta in loader:
        images = images.to(device, non_blocking=True)
        states = states.to(device, non_blocking=True)
        meta = meta.to(device, non_blocking=True)
        preds = model(images, states, meta)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(actions.numpy())
        all_meta.append(meta.cpu().numpy())
    return {
        "predictions": np.concatenate(all_preds),
        "targets": np.concatenate(all_targets),
        "meta": np.concatenate(all_meta),
    }
