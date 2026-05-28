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


def normalize_states(states_raw: np.ndarray) -> np.ndarray:
    """Convert raw 5-D state array (from collection_agent) to normalized 6-D state.

    Raw layout stored in chunk NPZs:
        [speed_kmh, heading_deg, speed_limit_kmh, lane_count, is_junction]

    Normalized layout expected by DriveNet (matches CLAUDE.md state vector):
        [speed/60, sin(heading), cos(heading), speed_limit/130, lane_count/4, is_junction]

    Parameters
    ----------
    states_raw : ndarray, shape (N, 5), dtype float32

    Returns
    -------
    ndarray, shape (N, 6), dtype float32
    """
    h_rad = np.deg2rad(states_raw[:, 1])
    return np.column_stack([
        states_raw[:, 0] / 60.0,
        np.sin(h_rad),
        np.cos(h_rad),
        states_raw[:, 2] / 130.0,
        states_raw[:, 3] / 4.0,
        states_raw[:, 4],
    ]).astype(np.float32)


def encode_metadata(
    data: dict[str, np.ndarray],
    weather_codes: dict[str, int],
    road_type_codes: dict[str, int],
    tod_codes: dict[str, int],
    traffic_codes: dict[str, int],
    style_codes: dict[str, int],
) -> np.ndarray:
    """Map string metadata arrays to integer codes.

    Column layout (matches meta_dims in bc.yaml):
        0: weather_preset
        1: road_type
        2: time_of_day
        3: traffic_density
        4: style

    Parameters
    ----------
    data : dict-like
        Must contain string arrays for each metadata field.
    *_codes : dict
        Mapping from string names to integer codes for each field.

    Returns
    -------
    ndarray, shape (N, 5), dtype int8
    """
    n = data["actions"].shape[0]
    meta = np.zeros((n, 5), dtype=np.int8)
    meta[:, 0] = np.array(
        [weather_codes[w] for w in data["weather_preset"].astype(str)],
        dtype=np.int8,
    )
    meta[:, 1] = np.array(
        [road_type_codes[r] for r in data["road_type"].astype(str)],
        dtype=np.int8,
    )
    meta[:, 2] = np.array(
        [tod_codes[t] for t in data["time_of_day"].astype(str)],
        dtype=np.int8,
    )
    meta[:, 3] = np.array(
        [traffic_codes[t] for t in data["traffic_density"].astype(str)],
        dtype=np.int8,
    )
    meta[:, 4] = np.array(
        [style_codes[s] for s in data["style"].astype(str)],
        dtype=np.int8,
    )
    return meta
