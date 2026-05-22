"""DriveNet: Condition-aware CNN policy for autonomous driving.

A 5-layer convolutional feature extractor with GroupNorm (which works
correctly at any batch size and does not shift behavior between training
and inference, making it better suited for sim-to-real transfer than
BatchNorm) and a 4-layer MLP head that outputs [steer, throttle, brake].
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DriveNet(nn.Module):
    """Condition-aware CNN policy mapping (image, state, metadata) to actions.

    Parameters
    ----------
    dropout : float
        Dropout probability applied in the MLP head.
    state_dim : int
        Dimension of the continuous state vector. Default is 6:
        (speed_norm, sin_heading, cos_heading, speed_limit_norm,
        lane_count_norm, is_junction).
    action_dim : int
        Number of action outputs (default 3: steer, throttle, brake).
    meta_dims : list[int] | None
        Number of categories per metadata field, used to build learned
        embedding layers. Default fields when not None:
        weather (6), road_type (3), time_of_day (3),
        traffic_density (3), driving_style (3).
        Pass None to disable all metadata embeddings.
    """

    def __init__(
        self,
        dropout: float = 0.3,
        state_dim: int = 6,
        action_dim: int = 3,
        meta_dims: list[int] | None = None,
    ) -> None:
        super().__init__()

        # GroupNorm groups are chosen so that channels / groups is an integer
        # in the range [3, 8], which is the stable operating range for GN.
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.GroupNorm(4, 24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.GroupNorm(6, 36),
            nn.ReLU(inplace=True),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.GroupNorm(8, 48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        with torch.no_grad():
            from src.preprocessing import RESIZE_H, RESIZE_W
            dummy = torch.zeros(1, 3, RESIZE_H, RESIZE_W)
            feat_size = self.features(dummy).shape[1]

        self.backbone_output_dim: int = feat_size + state_dim
        self._feat_size: int = feat_size

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

        self.head = nn.Sequential(
            nn.Linear(feat_size + state_dim + meta_total, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, action_dim),
        )

    def extract_features(
        self, image: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        """Return the pre-head feature vector: cat([CNN(image), state]).

        Shape: (B, backbone_output_dim). Metadata embeddings are not
        included. Used by ActorCritic to share the visual backbone
        between actor and critic heads.
        """
        return torch.cat([self.features(image), state], dim=1)

    def forward(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        meta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass producing [steer, throttle, brake] actions."""
        x = self.features(image)
        parts = [x, state]
        if self.meta_dims is not None:
            if meta is None:
                raise ValueError(
                    "DriveNet was built with meta_dims but received meta=None. "
                    "Either pass metadata tensors or rebuild the model with "
                    "meta_dims=None."
                )
            parts += [emb(meta[:, i]) for i, emb in enumerate(self.meta_embeddings)]
        x = torch.cat(parts, dim=1)
        x = self.head(x)
        steer = torch.tanh(x[:, 0:1])
        throttle = torch.sigmoid(x[:, 1:2])
        brake = torch.sigmoid(x[:, 2:3])
        return torch.cat([steer, throttle, brake], dim=1)
