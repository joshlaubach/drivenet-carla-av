"""Image preprocessing and metadata encoding for the DriveNet pipeline.

Architectural constants (CROP_TOP, RESIZE_H, etc.) define the model's input
shape and must stay consistent across all notebooks and agents.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

# ---- Architectural constants (must stay consistent across all notebooks) ----
CROP_TOP: int = 130
CROP_BOTTOM: int = 530
RESIZE_H: int = 100
RESIZE_W: int = 200
SPEED_NORM: float = 60.0


def crop_and_resize(images: np.ndarray) -> np.ndarray:
    """Crop sky/hood from a batch of images and resize to model input dims.

    Parameters
    ----------
    images : ndarray, shape (N, H, W, 3), dtype uint8

    Returns
    -------
    ndarray, shape (N, RESIZE_H, RESIZE_W, 3), dtype uint8
    """
    cropped = images[:, CROP_TOP:CROP_BOTTOM, :, :]
    n = cropped.shape[0]
    out = np.empty((n, RESIZE_H, RESIZE_W, 3), dtype=np.uint8)
    for i in range(n):
        out[i] = cv2.resize(
            cropped[i], (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_AREA,
        )
    return out


def preprocess_obs(obs: dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a single CARLA observation dict into model-ready CPU tensors.

    Parameters
    ----------
    obs : dict
        Keys ``"camera"`` (H, W, 3 uint8) and ``"state"`` (3,) float32.

    Returns
    -------
    img_t : Tensor, shape (1, 3, RESIZE_H, RESIZE_W)
    state_t : Tensor, shape (1, 3)
    """
    img = obs["camera"][CROP_TOP:CROP_BOTTOM, :, :]
    img = cv2.resize(img, (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_AREA)
    img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img_t = img_t.unsqueeze(0)
    state_t = torch.from_numpy(obs["state"]).unsqueeze(0)
    return img_t, state_t


def encode_metadata(
    data: dict[str, np.ndarray],
    weather_codes: dict[str, int],
    town_codes: dict[str, int],
    road_type_codes: dict[str, int],
    tod_codes: dict[str, int],
    traffic_codes: dict[str, int],
) -> np.ndarray:
    """Map string metadata arrays to integer codes.

    Parameters
    ----------
    data : dict-like
        Must contain string arrays for ``weather_preset``, ``town``,
        ``road_type``, ``time_of_day``, and ``traffic_density``.
    *_codes : dict
        Mapping from string names to integer codes.

    Returns
    -------
    ndarray, shape (N, 5), dtype int8
    """
    n = data["images"].shape[0]
    meta = np.zeros((n, 5), dtype=np.int8)
    meta[:, 0] = np.array(
        [weather_codes[w] for w in data["weather_preset"].astype(str)],
        dtype=np.int8,
    )
    meta[:, 1] = np.array(
        [town_codes[t] for t in data["town"].astype(str)],
        dtype=np.int8,
    )
    meta[:, 2] = np.array(
        [road_type_codes[r] for r in data["road_type"].astype(str)],
        dtype=np.int8,
    )
    meta[:, 3] = np.array(
        [tod_codes[t] for t in data["time_of_day"].astype(str)],
        dtype=np.int8,
    )
    meta[:, 4] = np.array(
        [traffic_codes[t] for t in data["traffic_density"].astype(str)],
        dtype=np.int8,
    )
    return meta
