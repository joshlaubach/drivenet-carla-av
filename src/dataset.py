"""PyTorch dataset and GPU-based augmentation for the DriveNet pipeline.

Raw state layout stored in .npz chunks (5 values, un-normalized):
    [speed_kmh, heading_deg, speed_limit_kmh, lane_count, is_junction]

Processed state fed to the model (6 values, normalized):
    [speed_norm, sin_heading, cos_heading,
     speed_limit_norm, lane_count_norm, is_junction]
"""

from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from src.preprocessing import SPEED_NORM


class DrivingDataset(Dataset):
    """PyTorch dataset that stores images as uint8 and converts on access.

    Parameters
    ----------
    images : ndarray
        Full image array (N, H, W, 3) uint8.
    states : ndarray
        Raw states (N, 5) float32 --
        [speed_kmh, heading_deg, speed_limit_kmh, lane_count, is_junction].
    actions : ndarray
        Expert actions (N, 3) float32 -- [steer, throttle, brake].
    meta : ndarray
        Encoded metadata (N, 5) int8.
    indices : ndarray
        Subset indices into the arrays for this split.
    """

    def __init__(
        self,
        images: np.ndarray,
        states: np.ndarray,
        actions: np.ndarray,
        meta: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        self.images = images
        self.indices = indices

        raw = states[indices].copy()
        speed_norm = raw[:, 0:1] / SPEED_NORM
        heading_rad = np.deg2rad(raw[:, 1:2])
        speed_limit_norm = np.clip(raw[:, 2:3] / 130.0, 0.0, 1.0)
        lane_count_norm = raw[:, 3:4] / 4.0
        is_junction = raw[:, 4:5]
        self.states = torch.from_numpy(
            np.hstack(
                [speed_norm, np.sin(heading_rad), np.cos(heading_rad),
                 speed_limit_norm, lane_count_norm, is_junction]
            ).astype(np.float32)
        )

        self.actions = torch.from_numpy(actions[indices].copy())
        self.meta = torch.from_numpy(meta[indices].copy().astype(np.int64))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        img = self.images[self.indices[idx]]
        if isinstance(img, torch.Tensor):
            is_uint8 = img.dtype == torch.uint8
        else:
            is_uint8 = img.dtype == np.uint8
            img = torch.from_numpy(img.copy())
        if img.ndim == 3:
            # single_cam / lidar: (H, W, C) -> (C, H, W)
            img = img.permute(2, 0, 1).float()
        else:
            # multi_cam: (n_cam, H, W, C) -> (n_cam, C, H, W)
            img = img.permute(0, 3, 1, 2).float()
        if is_uint8:
            img = img / 255.0
        return img, self.states[idx], self.actions[idx], self.meta[idx]


class GPUAugmenter:
    """Batch-level augmentation on GPU -- replaces per-sample CPU transforms.

    Parameters
    ----------
    device : torch.device
        GPU device for augmentation tensors.
    flip_prob : float
        Probability of horizontal flip per sample.
    jitter_brightness, jitter_contrast, jitter_saturation, jitter_hue : float
        ColorJitter parameters.
    noise_std : float
        Standard deviation of additive Gaussian noise.
    erase_prob : float
        Probability of random erasing per sample.
    """

    def __init__(
        self,
        device: torch.device,
        flip_prob: float = 0.5,
        jitter_brightness: float = 0.2,
        jitter_contrast: float = 0.2,
        jitter_saturation: float = 0.2,
        jitter_hue: float = 0.05,
        noise_std: float = 0.02,
        erase_prob: float = 0.1,
    ) -> None:
        self.device = device
        self.flip_prob = flip_prob
        self.noise_std = noise_std
        self.erase_prob = erase_prob
        self.jitter = T.ColorJitter(
            brightness=jitter_brightness, contrast=jitter_contrast,
            saturation=jitter_saturation, hue=jitter_hue,
        )

    @torch.no_grad()
    def __call__(
        self, images: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentation to a batch of images and actions.

        Handles both 4-D single_cam/lidar input (N, C, H, W) and 5-D
        multi_cam input (N, n_cam, C, H, W).  For multi_cam, all cameras
        of the same frame share one flip decision so the steer sign flip
        remains consistent.  ColorJitter, noise, and erasing operate on
        the (N*n_cam, C, H, W) reshape and are applied per-camera-frame.
        """
        is_multicam = images.ndim == 5
        if is_multicam:
            n, n_cam, c, h, w = images.shape
        else:
            n, c, h, w = images.shape
            n_cam = 1

        # Per-sample flip decision (one coin flip per frame, not per camera)
        flip_per_sample = (
            torch.rand(n, device=self.device) < self.flip_prob
        )  # (N,)

        if is_multicam:
            # Broadcast (N,) -> (N, 1, 1, 1, 1) to flip all cameras together
            flip_mask = flip_per_sample.view(n, 1, 1, 1, 1)
            images = torch.where(flip_mask, images.flip(-1), images)
            # Reshape to 4-D for ColorJitter / noise / erasing
            images = images.reshape(n * n_cam, c, h, w)
        else:
            flip_mask = flip_per_sample.view(n, 1, 1, 1)
            images = torch.where(flip_mask, images.flip(-1), images)

        steer_mask = flip_per_sample.float() * -2 + 1  # +1 no-flip, -1 flipped
        actions = actions.clone()
        actions[:, 0] *= steer_mask

        # ColorJitter (runs on GPU tensors; operates on 4-D)
        images = self.jitter(images)
        images = images.clamp(0.0, 1.0)

        # Gaussian noise
        images = images + torch.randn_like(images) * self.noise_std

        # Random erasing (independent per camera frame)
        n_flat = images.shape[0]  # N*n_cam (multicam) or N (single/lidar)
        erase_mask = torch.rand(n_flat, device=self.device) < self.erase_prob
        if erase_mask.any():
            n_erase = int(erase_mask.sum().item())
            eh = max(1, int(h * 0.1))
            ew = max(1, int(w * 0.1))
            tops = torch.randint(0, h - eh, (n_erase,), device=self.device)
            lefts = torch.randint(0, w - ew, (n_erase,), device=self.device)
            erase_indices = torch.where(erase_mask)[0]
            for i, (t, left) in enumerate(zip(tops, lefts)):
                idx = erase_indices[i]
                images[idx, :, t:t + eh, left:left + ew] = torch.rand(
                    c, eh, ew, device=self.device,
                )

        images = images.clamp(0.0, 1.0)

        if is_multicam:
            images = images.reshape(n, n_cam, c, h, w)

        return images, actions
