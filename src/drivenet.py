"""DriveNet: Condition-aware CNN policy for autonomous driving.

5-layer convolutional feature extractor with optional metadata embeddings
and a 4-layer MLP head producing [steer, throttle, brake].
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DriveNet(nn.Module):
    """Condition-aware CNN policy that maps (image, state, metadata) to actions.

    Parameters
    ----------
    dropout : float
        Dropout probability in the MLP head.
    state_dim : int
        Dimension of the state vector (default 3: speed, sin_heading, cos_heading).
    action_dim : int
        Number of action outputs (default 3: steer, throttle, brake).
    meta_dims : list[int] | None
        Number of categories per metadata field for embedding layers.
        ``None`` disables metadata embeddings (backward-compatible).
    """

    def __init__(
        self,
        dropout: float = 0.3,
        state_dim: int = 3,
        action_dim: int = 3,
        meta_dims: list[int] | None = None,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.BatchNorm2d(36),
            nn.ReLU(inplace=True),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        # Compute flattened feature size from a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 100, 200)
            feat_size = self.features(dummy).shape[1]

        self.backbone_output_dim: int = feat_size + state_dim
        self._feat_size: int = feat_size

        # Optional condition-metadata embeddings
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

        Shape: (B, backbone_output_dim) = (B, feat_size + state_dim).
        Used by ActorCritic to share the visual backbone between actor
        and critic heads. Meta embeddings are NOT included.
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
            if meta is not None:
                parts += [
                    emb(meta[:, i])
                    for i, emb in enumerate(self.meta_embeddings)
                ]
            else:
                batch = x.shape[0]
                for emb in self.meta_embeddings:
                    parts.append(torch.zeros(batch, emb.embedding_dim,
                                             device=x.device))
        x = torch.cat(parts, dim=1)
        x = self.head(x)
        steer = torch.tanh(x[:, 0:1])
        throttle = torch.sigmoid(x[:, 1:2])
        brake = torch.sigmoid(x[:, 2:3])
        return torch.cat([steer, throttle, brake], dim=1)
