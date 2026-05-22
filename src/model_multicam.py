"""FSDNet: multi-camera + LiDAR fusion policy for CARLA.

Multi-camera + LiDAR fusion architecture. Implemented for future retraining;
current trained models use single front camera (see src/drivenet.py).

Architecture overview
---------------------
* Four ResNet-18 camera encoders (one per camera) via nn.ModuleDict.
  Each produces a 512-d feature vector after global average pooling.
* LiDAR BEV CNN encoder: 4 conv layers → AdaptiveAvgPool(4,4) → Linear(256).
* Fusion trunk: concat [4×512 cam + 256 lidar + 1 speed] → Linear(FUSED_DIM, 512)
  → LayerNorm → ReLU → Linear(512, 256) → LayerNorm → ReLU.
* Actor head: Linear(256, 3) → Tanh  →  [steer, throttle_raw, brake_raw] ∈ [-1, 1].
* Critic head: Linear(256, 1)  →  scalar value estimate for PPO.

Action remapping
----------------
Actor outputs are in [-1, 1].  Use FSDNet.to_carla(raw) to remap:
  steer        → [-1, 1]  (unchanged)
  throttle_raw → [0,  1]  ((raw + 1) / 2)
  brake_raw    → [0,  1]  ((raw + 1) / 2)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CAM_FEAT = 512      # ResNet-18 global average pool output dim
_LIDAR_FEAT = 256    # LiDAR encoder output dim
_FUSED_DIM = 4 * _CAM_FEAT + _LIDAR_FEAT + 1   # +1 speed scalar = 2305

_CAM_NAMES = ["front", "front_left", "front_right", "rear"]


# ---------------------------------------------------------------------------
# LiDAR BEV encoder
# ---------------------------------------------------------------------------

class _LidarEncoder(nn.Module):
    """4-layer conv encoder for 6-channel 256×256 BEV input."""

    def __init__(self) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.proj = nn.Linear(256 * 4 * 4, _LIDAR_FEAT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.cnn(x))


# ---------------------------------------------------------------------------
# FSDNet
# ---------------------------------------------------------------------------

class FSDNet(nn.Module):
    """Multi-camera + LiDAR fusion policy.

    Parameters
    ----------
    pretrained:
        Load ImageNet-pretrained weights for the ResNet-18 camera encoders.
        Set False for random-init or when restoring from a DriveNet checkpoint.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.cam_encoders = nn.ModuleDict({
            name: self._make_cam_encoder(weights) for name in _CAM_NAMES
        })
        self.lidar_encoder = _LidarEncoder()

        self.fusion = nn.Sequential(
            nn.Linear(_FUSED_DIM, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
        )

        self.actor = nn.Sequential(
            nn.Linear(256, 3),
            nn.Tanh(),
        )
        self.critic = nn.Linear(256, 1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        front: torch.Tensor,
        front_left: torch.Tensor,
        front_right: torch.Tensor,
        rear: torch.Tensor,
        lidar: torch.Tensor,
        speed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (action, value).

        Parameters
        ----------
        front, front_left, front_right, rear : (B, 3, H, W) float32
            Camera images normalised to [0, 1].
        lidar : (B, 6, 256, 256) float32
            BEV occupancy+intensity channels in [0, 1].
        speed : (B, 1) float32
            Ego speed in m/s (not normalised here; normalise before calling).

        Returns
        -------
        action : (B, 3) float32  — [steer, throttle_raw, brake_raw] ∈ [-1, 1]
        value  : (B, 1) float32
        """
        images = {
            "front": front,
            "front_left": front_left,
            "front_right": front_right,
            "rear": rear,
        }
        cam_feats = [self.cam_encoders[name](images[name]) for name in _CAM_NAMES]
        lidar_feat = self.lidar_encoder(lidar)

        fused = torch.cat([*cam_feats, lidar_feat, speed], dim=1)
        trunk = self.fusion(fused)

        action = self.actor(trunk)
        value = self.critic(trunk)
        return action, value

    # ------------------------------------------------------------------
    # Action remapping
    # ------------------------------------------------------------------

    @staticmethod
    def to_carla(raw: torch.Tensor) -> dict[str, torch.Tensor]:
        """Remap actor output from [-1, 1] to CARLA control ranges.

        Parameters
        ----------
        raw : (B, 3) or (3,) tensor — [steer, throttle_raw, brake_raw]

        Returns
        -------
        dict with keys 'steer', 'throttle', 'brake' all in [0, 1] except
        steer which remains in [-1, 1].
        """
        steer = raw[..., 0]
        throttle = (raw[..., 1] + 1.0) / 2.0
        brake = (raw[..., 2] + 1.0) / 2.0
        return {"steer": steer, "throttle": throttle, "brake": brake}

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _make_cam_encoder(weights: "ResNet18_Weights | None") -> nn.Module:
        backbone = resnet18(weights=weights)
        # Drop the classification head; keep up to global average pool
        return nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())
