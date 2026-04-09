"""
BehaviorCloningAgent -- WAT Framework / Workflow 02

Reads: workflows/02_behavior_cloning.md
Sequences: data loading, preprocessing, DrivingDataset, DriveNet, train_model

Trains a single BC_model (all 6 towns, GPU augmentation, full metadata embeddings).
Checkpoint saved to models/BC_model_best.pt to match notebook 02 naming.

Usage:
    from src.agents.training_agent import BehaviorCloningAgent
    agent = BehaviorCloningAgent()
    agent.run()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import load_config, require_keys
from src.dataset import DrivingDataset, GPUAugmenter
from src.drivenet import DriveNet
from src.preprocessing import crop_and_resize, encode_metadata
from src.training import evaluate, train_model

log = logging.getLogger(__name__)


class BehaviorCloningAgent:
    """Trains a single BC_model on all collected conditions.

    Does not implement the training loop -- sequences DrivingDataset,
    GPUAugmenter, DriveNet, and training.train_model per
    workflows/02_behavior_cloning.md.
    """

    def __init__(
        self,
        data_dir: str = "data",
        save_dir: str = "models",
        results_dir: str = "results",
        seed: int | None = None,
    ) -> None:
        self.cfg = load_config("bc")
        require_keys(
            self.cfg,
            ["seed", "batch_size", "lr", "weight_decay", "max_epochs",
             "early_stop_patience", "lr_patience", "lr_factor", "dropout",
             "loss_weights", "test_fraction", "val_fraction", "meta_dims",
             "model_name", "weather_codes", "town_codes", "road_type_codes",
             "tod_codes", "traffic_codes"],
            "bc",
        )

        self.data_dir = Path(data_dir)
        self.save_dir = Path(save_dir)
        self.results_dir = Path(results_dir)
        self.seed = seed if seed is not None else self.cfg["seed"]

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("BehaviorCloningAgent using device: %s", self.device)

    # -- Public entry point ----------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Train BC_model. Returns test metrics dict."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        cfg = self.cfg
        model_name: str = cfg["model_name"]

        log.info("Loading data from %s ...", self.data_dir)
        raw = self._load_all_chunks()

        log.info("Loaded %d frames. Preprocessing images ...", raw["images"].shape[0])
        images = crop_and_resize(raw["images"])
        meta = encode_metadata(
            raw,
            cfg["weather_codes"],
            cfg["town_codes"],
            cfg["road_type_codes"],
            cfg["tod_codes"],
            cfg["traffic_codes"],
        )

        train_idx, val_idx, test_idx = self._split_indices(len(images))
        np.savez_compressed(
            self.results_dir / "bc_split_indices.npz",
            train=train_idx, val=val_idx, test=test_idx,
        )

        criterion = self._weighted_mse(cfg["loss_weights"])

        train_ds = DrivingDataset(images, raw["states"], raw["actions"], meta, train_idx)
        val_ds   = DrivingDataset(images, raw["states"], raw["actions"], meta, val_idx)
        test_ds  = DrivingDataset(images, raw["states"], raw["actions"], meta, test_idx)

        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                                  shuffle=True,  num_workers=0, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                                  shuffle=False, num_workers=0, pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"],
                                  shuffle=False, num_workers=0, pin_memory=True)

        model = DriveNet(
            dropout=cfg["dropout"],
            state_dim=3,
            action_dim=3,
            meta_dims=cfg["meta_dims"],
        ).to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=cfg["lr_patience"],
            factor=cfg["lr_factor"],
        )
        augmenter = GPUAugmenter(device=self.device)

        log.info("Training %s ...", model_name)
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            model_name=model_name,
            save_dir=str(self.save_dir),
            device=self.device,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            augmenter=augmenter,
            max_epochs=cfg["max_epochs"],
            early_stop_patience=cfg["early_stop_patience"],
        )

        test_loss, test_per_target = evaluate(model, test_loader, criterion, self.device)
        metrics: dict[str, Any] = {
            "test_loss": test_loss,
            "test_steer_mse": float(test_per_target[0]),
            "test_throttle_mse": float(test_per_target[1]),
            "test_brake_mse": float(test_per_target[2]),
        }
        log.info(
            "%s -- test_loss=%.6f  steer=%.4f  throttle=%.4f  brake=%.4f",
            model_name, test_loss,
            test_per_target[0], test_per_target[1], test_per_target[2],
        )

        with open(self.results_dir / "bc_training_history.json", "w") as f:
            json.dump(history, f, indent=2)
        with open(self.results_dir / "bc_test_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        log.info("Results saved to %s.", self.results_dir)

        return metrics

    # -- Data loading ----------------------------------------------------------

    def _load_all_chunks(self) -> dict[str, np.ndarray]:
        """Load and concatenate all chunk .npz files from the data directory."""
        chunk_files = sorted(self.data_dir.rglob("chunk_*.npz"))
        if not chunk_files:
            raise RuntimeError(
                f"No chunk files found in {self.data_dir}. "
                "Run DataCollectionAgent first."
            )
        parts: dict[str, list[np.ndarray]] = {}
        for path in chunk_files:
            chunk = np.load(path, allow_pickle=True)
            for key in chunk.files:
                parts.setdefault(key, []).append(chunk[key])
            log.debug("Loaded %s (%d frames).", path.name, len(chunk["images"]))
        return {k: np.concatenate(v, axis=0) for k, v in parts.items()}

    # -- Utilities -------------------------------------------------------------

    def _split_indices(
        self, n: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create stratified train/val/test index split."""
        cfg = self.cfg
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(n)
        n_test = int(n * cfg["test_fraction"])
        n_val  = int((n - n_test) * cfg["val_fraction"])
        test_idx  = idx[:n_test]
        val_idx   = idx[n_test: n_test + n_val]
        train_idx = idx[n_test + n_val:]
        log.info("Split: train=%d  val=%d  test=%d", len(train_idx), len(val_idx), len(test_idx))
        return train_idx, val_idx, test_idx

    def _weighted_mse(self, weights: list[float]) -> nn.Module:
        """Create a weighted MSE loss module."""
        w = torch.tensor(weights, dtype=torch.float32)

        class WeightedMSE(nn.Module):
            def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                return ((pred - target) ** 2 * w.to(pred.device)).mean()

        return WeightedMSE()
