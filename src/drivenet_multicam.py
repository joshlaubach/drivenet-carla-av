"""MultiCamDriveNet: Late-fusion multi-camera policy for autonomous driving.

Three cameras are used: front, left, and right. Each camera is processed
by a shared CNN backbone (identical weights), and the resulting feature
vectors are concatenated before being passed to the MLP head. This is
the same late-fusion, shared-backbone design used in Tesla's HydraNet.

Sharing weights forces the backbone to learn view-invariant features and
keeps the parameter count identical to the single-camera baseline, making
the sensor comparison in the causal analysis a fair one.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.drivenet import DriveNet


class MultiCamDriveNet(nn.Module):
    """Condition-aware multi-camera policy with late fusion.

    Three cameras (front, left, right) each pass through a shared CNN
    backbone. Their feature vectors are concatenated, then combined with
    the state vector and metadata embeddings before the MLP head.

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
        Default fields: weather (6), road_type (3), time_of_day (3),
        traffic_density (3), driving_style (3).
    n_cameras : int
        Number of camera views (default 3: front, left, right).
    """

    def __init__(
        self,
        dropout: float = 0.3,
        state_dim: int = 6,
        action_dim: int = 3,
        meta_dims: list[int] | None = None,
        n_cameras: int = 3,
    ) -> None:
        super().__init__()
        self.n_cameras = n_cameras

        # Single backbone shared across all cameras
        _ref = DriveNet(dropout=dropout, state_dim=state_dim,
                        action_dim=action_dim)
        self.shared_backbone = _ref.features
        feat_size = _ref._feat_size

        self.meta_dims = meta_dims
        meta_total = 0
        if meta_dims is not None:
            self.meta_embeddings = nn.ModuleList()
            for n_categories in meta_dims:
                embed_dim = min(n_categories // 2 + 1, 4)
                self.meta_embeddings.append(
                    nn.Embedding(n_categories, embed_dim)
                )
                meta_total += embed_dim

        # Fused input: concatenated features from all cameras + state + meta
        fused_dim = feat_size * n_cameras + state_dim + meta_total

        self.head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, action_dim),
        )

    def forward(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        meta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through all cameras, then fuse and predict actions.

        Parameters
        ----------
        images : Tensor, shape (B, n_cameras, 3, H, W)
            Stacked camera images. Axis 1 indexes cameras in the order
            [front, left, right].
        state : Tensor, shape (B, state_dim)
        meta : Tensor, shape (B, len(meta_dims)) or None

        Returns
        -------
        Tensor, shape (B, 3) -- [steer, throttle, brake]
        """
        # Process each camera through the shared backbone
        cam_feats = []
        for i in range(self.n_cameras):
            cam_feats.append(self.shared_backbone(images[:, i]))

        parts = cam_feats + [state]

        if self.meta_dims is not None:
            if meta is None:
                raise ValueError(
                    "MultiCamDriveNet was built with meta_dims but received "
                    "meta=None."
                )
            parts += [emb(meta[:, i]) for i, emb in enumerate(self.meta_embeddings)]

        x = torch.cat(parts, dim=1)
        x = self.head(x)
        steer = torch.tanh(x[:, 0:1])
        throttle = torch.sigmoid(x[:, 1:2])
        brake = torch.sigmoid(x[:, 2:3])
        return torch.cat([steer, throttle, brake], dim=1)
