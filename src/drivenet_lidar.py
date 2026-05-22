"""LidarDriveNet: BEV-projection lidar policy for autonomous driving.

Raw lidar point clouds from CARLA are projected into a Bird's Eye View
(BEV) grid -- a top-down 2D image where each pixel represents a small
patch of ground and its value encodes the maximum point height and point
density in that patch. This BEV image is then fed through the same CNN
backbone as the single-camera and multi-camera models, making it a fair
architectural comparison for the causal sensor experiment.

BEV projection is the standard representation used by Waymo, Uber ATG,
and most production AV stacks that rely on lidar.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.drivenet import DriveNet
from src.preprocessing import RESIZE_H, RESIZE_W


# BEV grid parameters. The grid covers a 40 m x 40 m patch centered on
# the ego vehicle and is downsampled to the same spatial resolution as
# the camera input so the CNN backbone can be reused without changes.
BEV_RANGE_M: float = 40.0          # half-range in both x and y (metres)
BEV_HEIGHT_MIN_M: float = -2.0     # points below this are ground clutter
BEV_HEIGHT_MAX_M: float = 4.0      # points above this are noise


def points_to_bev(
    points: np.ndarray,
    grid_h: int = RESIZE_H,
    grid_w: int = RESIZE_W,
    range_m: float = BEV_RANGE_M,
    height_min: float = BEV_HEIGHT_MIN_M,
    height_max: float = BEV_HEIGHT_MAX_M,
) -> np.ndarray:
    """Project a lidar point cloud to a 3-channel BEV image.

    The three channels encode:
        0 -- height map: maximum point height (z) in each cell, normalized
             to [0, 1] over [height_min, height_max].
        1 -- density map: log-scaled point count per cell, normalized to
             [0, 1]. Log scaling compresses the wide dynamic range.
        2 -- intensity map: mean lidar intensity per cell (if available),
             otherwise zeros.

    Parameters
    ----------
    points : ndarray, shape (N, 4) or (N, 3)
        Lidar points in ego-frame coordinates. Columns are [x, y, z] and
        optionally [intensity]. x points forward, y points left, z points up.
    grid_h, grid_w : int
        Output grid resolution in pixels.
    range_m : float
        Half-range of the BEV grid in metres. Points outside [-range_m,
        range_m] in both x and y are discarded.
    height_min, height_max : float
        Height clipping range in metres.

    Returns
    -------
    ndarray, shape (3, grid_h, grid_w), dtype float32
    """
    height_map = np.full((grid_h, grid_w), -1.0, dtype=np.float32)
    density_map = np.zeros((grid_h, grid_w), dtype=np.float32)
    intensity_map = np.zeros((grid_h, grid_w), dtype=np.float32)
    intensity_count = np.zeros((grid_h, grid_w), dtype=np.float32)

    if points.shape[0] == 0:
        return np.zeros((3, grid_h, grid_w), dtype=np.float32)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    has_intensity = points.shape[1] >= 4

    # Discard points outside the BEV range and height bounds
    mask = (
        (np.abs(x) <= range_m)
        & (np.abs(y) <= range_m)
        & (z >= height_min)
        & (z <= height_max)
    )
    x, y, z = x[mask], y[mask], z[mask]
    intensity = points[mask, 3] if has_intensity else np.zeros_like(x)

    if x.shape[0] == 0:
        return np.stack([height_map, density_map, intensity_map], axis=0)

    # Map ego-frame coordinates to grid pixel indices.
    # x (forward) maps to rows: row 0 = farthest forward.
    # y (left)    maps to cols: col 0 = farthest left.
    row = np.clip(
        ((range_m - x) / (2.0 * range_m) * grid_h).astype(np.int32),
        0, grid_h - 1,
    )
    col = np.clip(
        ((range_m - y) / (2.0 * range_m) * grid_w).astype(np.int32),
        0, grid_w - 1,
    )

    # Fill height, density, and intensity maps
    for r, c, zi, ii in zip(row, col, z, intensity):
        if zi > height_map[r, c]:
            height_map[r, c] = zi
        density_map[r, c] += 1.0
        intensity_map[r, c] += ii
        intensity_count[r, c] += 1.0

    # Track which cells received at least one point before normalizing
    occupied = height_map > -1.0

    # Normalize height: empty cells map to 0.0
    height_map = np.where(
        occupied,
        np.clip(
            (height_map - height_min) / (height_max - height_min), 0.0, 1.0
        ),
        0.0,
    )

    # Normalize density with log scaling
    max_density = density_map.max()
    if max_density > 0:
        density_map = np.log1p(density_map) / np.log1p(max_density)

    # Normalize intensity to [0, 1] -- CARLA reports intensity in [0, 1],
    # but we clamp regardless to handle out-of-spec inputs.
    nonzero = intensity_count > 0
    intensity_map[nonzero] /= intensity_count[nonzero]
    intensity_map = np.clip(intensity_map, 0.0, 1.0)

    return np.stack([height_map, density_map, intensity_map], axis=0)


class LidarDriveNet(nn.Module):
    """Lidar BEV policy that reuses the DriveNet CNN backbone.

    A raw lidar point cloud is first projected to a 3-channel BEV image
    via points_to_bev(), then passed through the same 5-layer CNN used
    by the single-camera and multi-camera models. This keeps the
    architectural comparison fair for the causal sensor experiment.

    Parameters
    ----------
    dropout : float
        Dropout probability in the MLP head.
    state_dim : int
        Dimension of the continuous state vector (default 6).
    action_dim : int
        Number of action outputs (default 3: steer, throttle, brake).
    meta_dims : list[int] | None
        Category counts for each metadata embedding field.
    """

    def __init__(
        self,
        dropout: float = 0.3,
        state_dim: int = 6,
        action_dim: int = 3,
        meta_dims: list[int] | None = None,
    ) -> None:
        super().__init__()

        # Reuse DriveNet as the full model. The only difference at the
        # Python level is that the input to forward() is a BEV image
        # rather than a camera image. The tensor shapes are identical.
        self._model = DriveNet(
            dropout=dropout,
            state_dim=state_dim,
            action_dim=action_dim,
            meta_dims=meta_dims,
        )
        self.features = self._model.features
        self.head = self._model.head
        self.meta_dims = self._model.meta_dims
        if meta_dims is not None:
            self.meta_embeddings = self._model.meta_embeddings

    def forward(
        self,
        bev: torch.Tensor,
        state: torch.Tensor,
        meta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass on a BEV image tensor.

        Parameters
        ----------
        bev : Tensor, shape (B, 3, H, W)
            BEV image produced by points_to_bev(), already as a float
            tensor scaled to [0, 1].
        state : Tensor, shape (B, state_dim)
        meta : Tensor, shape (B, len(meta_dims)) or None

        Returns
        -------
        Tensor, shape (B, 3) -- [steer, throttle, brake]
        """
        return self._model(bev, state, meta)
