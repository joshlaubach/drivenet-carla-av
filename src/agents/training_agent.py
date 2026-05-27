"""BehaviorCloningAgent -- WAT Framework / Workflow 02

Reads: workflows/02_behavior_cloning.md
Sequences: data loading, preprocessing, DrivingDataset, DriveNet, train_model

Trains one BC model per sensor suite. The sensor suite is specified at init
time and determines the model architecture and checkpoint filename:
  - single_cam: DriveNet      -> BC_model_single_cam_best.pt
  - multi_cam:  MultiCamDriveNet -> BC_model_multi_cam_best.pt
  - lidar:      LidarDriveNet -> BC_model_lidar_best.pt

Usage:
    from src.agents.training_agent import BehaviorCloningAgent
    agent = BehaviorCloningAgent(sensor_suite="single_cam")
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

from src.carla_env import CarlaEnv
from src.config import load_config, require_keys
from src.dataset import DrivingDataset, GPUAugmenter
from src.drivenet import DriveNet
from src.drivenet_lidar import LidarDriveNet
from src.drivenet_multicam import MultiCamDriveNet
from src.preprocessing import RESIZE_H, RESIZE_W, crop_and_resize, encode_metadata
from src.training import evaluate, train_model

log = logging.getLogger(__name__)


class BehaviorCloningAgent:
    """Trains a BC model on all collected conditions for one sensor suite.

    Does not implement the training loop -- sequences DrivingDataset,
    GPUAugmenter, a DriveNet-family model, and training.train_model per
    workflows/02_behavior_cloning.md.
    """

    def __init__(
        self,
        sensor_suite: str = "single_cam",
        data_dir: str = "data",
        save_dir: str = "models",
        results_dir: str = "results",
        seed: int | None = None,
    ) -> None:
        if sensor_suite not in CarlaEnv.VALID_SUITES:
            raise ValueError(
                f"sensor_suite must be one of {CarlaEnv.VALID_SUITES}, "
                f"got '{sensor_suite}'."
            )
        self.cfg = load_config("bc")
        require_keys(
            self.cfg,
            ["seed", "batch_size", "lr", "weight_decay", "max_epochs",
             "early_stop_patience", "lr_patience", "lr_factor", "dropout",
             "loss_weights", "test_fraction", "val_fraction", "meta_dims",
             "model_name", "weather_codes", "road_type_codes",
             "tod_codes", "traffic_codes", "style_codes"],
            "bc",
        )

        self.sensor_suite = sensor_suite
        self.data_dir = Path(data_dir)
        self.save_dir = Path(save_dir)
        self.results_dir = Path(results_dir)
        self.seed = seed if seed is not None else self.cfg["seed"]

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(
            "BehaviorCloningAgent using device: %s  sensor_suite: %s",
            self.device, self.sensor_suite,
        )

    # -- Public entry point ---------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Train a BC model for self.sensor_suite. Returns test metrics dict."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        cfg = self.cfg
        model_name = f"BC_model_{self.sensor_suite}"

        log.info("Loading data from %s ...", self.data_dir)
        raw = self._load_all_chunks()  # images already cropped/resized inside

        log.info("Loaded %d frames.", raw["images"].shape[0])
        images = raw["images"]  # shape varies by suite; always stored under "images"
        meta = encode_metadata(
            raw,
            cfg["weather_codes"],
            cfg["road_type_codes"],
            cfg["tod_codes"],
            cfg["traffic_codes"],
            cfg["style_codes"],
        )

        train_idx, val_idx, test_idx = self._split_indices(len(images))
        np.savez_compressed(
            self.results_dir / f"bc_split_indices_{self.sensor_suite}.npz",
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

        model = self._build_model().to(self.device)
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
            "sensor_suite": self.sensor_suite,
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

        hist_path = self.results_dir / f"bc_training_history_{self.sensor_suite}.json"
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
        metrics_path = self.results_dir / f"bc_test_metrics_{self.sensor_suite}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        log.info("Results saved to %s.", self.results_dir)

        return metrics

    # -- Model construction ---------------------------------------------------

    def _build_model(self) -> nn.Module:
        """Instantiate the appropriate DriveNet model for the sensor suite."""
        cfg = self.cfg
        kwargs = dict(
            dropout=cfg["dropout"],
            state_dim=6,
            action_dim=3,
            meta_dims=cfg["meta_dims"],
        )
        if self.sensor_suite == "single_cam":
            return DriveNet(**kwargs)
        elif self.sensor_suite == "multi_cam":
            return MultiCamDriveNet(**kwargs)
        else:  # lidar
            return LidarDriveNet(**kwargs)

    # -- Data loading ---------------------------------------------------------

    def _load_all_chunks(self) -> dict[str, np.ndarray]:
        """Load, preprocess, and concatenate all chunk NPZ files.

        Payload key and preprocessing are selected by sensor suite:
          single_cam  — key "images"       (N,H,W,3 uint8);  crop/resize each frame
          multi_cam   — key "images_multi" (N,3,H,W,3 uint8); crop/resize each of 3 views
          lidar       — key "bev"          (N,100,200,3 float32); already at model res, no-op
        Result is always stored under "images" for uniform downstream access.
        """
        chunk_files = sorted(self.data_dir.rglob("chunk_*.npz"))
        if not chunk_files:
            raise RuntimeError(
                f"No chunk files found in {self.data_dir}. "
                "Run DataCollectionAgent first."
            )

        if self.sensor_suite == "single_cam":
            payload_key = "images"
        elif self.sensor_suite == "multi_cam":
            payload_key = "images_multi"
        else:  # lidar
            payload_key = "bev"

        image_parts: list[np.ndarray] = []
        other_parts: dict[str, list[np.ndarray]] = {}
        for path in chunk_files:
            with np.load(path, allow_pickle=True) as chunk:
                raw_imgs = chunk[payload_key]
                if self.sensor_suite == "single_cam":
                    processed = crop_and_resize(raw_imgs)
                elif self.sensor_suite == "multi_cam":
                    # raw_imgs: (N, 3, H, W, 3) — crop/resize each view independently
                    n_frames, n_cams = raw_imgs.shape[0], raw_imgs.shape[1]
                    processed = np.empty(
                        (n_frames, n_cams, RESIZE_H, RESIZE_W, 3), dtype=np.uint8
                    )
                    for cam in range(n_cams):
                        processed[:, cam] = crop_and_resize(raw_imgs[:, cam])
                else:  # lidar — BEV is already at model resolution, pass through
                    processed = raw_imgs

                image_parts.append(processed)
                for key in chunk.files:
                    if key != payload_key:
                        other_parts.setdefault(key, []).append(chunk[key])
            log.debug("Preprocessed %s (%d frames).", path.name, len(image_parts[-1]))

        result = {"images": np.concatenate(image_parts, axis=0)}
        result.update({k: np.concatenate(v, axis=0) for k, v in other_parts.items()})
        return result

    # -- Utilities ------------------------------------------------------------

    def _split_indices(
        self, n: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.cfg
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(n)
        n_test = int(n * cfg["test_fraction"])
        n_val  = int((n - n_test) * cfg["val_fraction"])
        test_idx  = idx[:n_test]
        val_idx   = idx[n_test: n_test + n_val]
        train_idx = idx[n_test + n_val:]
        log.info(
            "Split: train=%d  val=%d  test=%d",
            len(train_idx), len(val_idx), len(test_idx),
        )
        return train_idx, val_idx, test_idx

    def _weighted_mse(self, weights: list[float]) -> nn.Module:
        w = torch.tensor(weights, dtype=torch.float32)

        class WeightedMSE(nn.Module):
            def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                return ((pred - target) ** 2 * w.to(pred.device)).mean()

        return WeightedMSE()
