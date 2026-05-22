"""PPO primitives: ActorCritic network, RolloutBuffer, ppo_update, and
style-aware reward shaping.

Training orchestration lives in PPOAgent, not here.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from src.drivenet import DriveNet
from src.preprocessing import RESIZE_H, RESIZE_W

log = logging.getLogger(__name__)

# Reference cruise speed used to normalize the speed bonus to roughly [0, 1]
# under typical urban driving. Matches CARLA's default urban speed limit.
SPEED_BONUS_REF_KMH: float = 40.0

# Empirical jerk scale: a moderate-jerk trajectory produces a penalty of
# roughly 1.0 at this normalization factor.
JERK_NORM_FACTOR: float = 1000.0


# ---------------------------------------------------------------------------
# Driving-style reward shaping
# ---------------------------------------------------------------------------

def compute_style_reward(
    base_reward: float,
    speed_kmh: float,
    prev_speed_kmh: float,
    prev_prev_speed_kmh: float,
    lane_id: int,
    prev_lane_id: int,
    style_weights: dict[str, float],
    dt: float = 0.05,
) -> float:
    """Apply style-dependent jerk, speed, and lane-change terms to base_reward.

    Parameters
    ----------
    base_reward : float
        Reward returned by CarlaEnv.step() (a flat +1.0 per step).
    speed_kmh, prev_speed_kmh, prev_prev_speed_kmh : float
        Three consecutive ego speed readings used to compute jerk.
    lane_id, prev_lane_id : int
        Current and previous CARLA waypoint lane IDs.
        Pass -1 as a sentinel for the first step of an episode.
    style_weights : dict
        Keys: jerk_penalty, speed_bonus, lane_change_penalty.
    dt : float
        Simulation timestep in seconds (default 0.05 = 20 FPS).

    Returns
    -------
    float
        Shaped reward.
    """
    reward = base_reward

    reward += (speed_kmh / SPEED_BONUS_REF_KMH) * style_weights["speed_bonus"]

    if prev_lane_id != -1 and prev_prev_speed_kmh >= 0.0:
        accel_now = (speed_kmh - prev_speed_kmh) / dt
        accel_prev = (prev_speed_kmh - prev_prev_speed_kmh) / dt
        jerk = abs(accel_now - accel_prev) / dt
        reward -= (jerk / JERK_NORM_FACTOR) * style_weights["jerk_penalty"]

    if prev_lane_id != -1 and lane_id != prev_lane_id:
        reward -= style_weights["lane_change_penalty"]

    return reward


# ---------------------------------------------------------------------------
# ActorCritic
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """Actor-critic policy initialized from a BC DriveNet checkpoint.

    Only the CNN feature extractor transfers from BC. Both the actor and
    critic heads are freshly initialized because the BC head has a
    different input size (it includes metadata embeddings). The visual
    backbone is what carries over the learned driving representation.

    A learned style embedding (chill / standard / hurry) is concatenated
    with the visual features and state before being passed to both heads,
    allowing one model to express multiple driving personalities.

    GroupNorm in the backbone requires no special handling during PPO
    updates -- unlike BatchNorm, it does not accumulate running statistics
    that could drift on small, correlated minibatches.
    """

    def __init__(
        self,
        bc_state_dict: dict[str, torch.Tensor],
        dropout: float = 0.3,
        state_dim: int = 6,
        action_dim: int = 3,
        n_styles: int = 3,
        style_embed_dim: int = 4,
        log_std_init: list[float] | None = None,
    ) -> None:
        super().__init__()

        _bc = DriveNet(dropout=dropout, state_dim=state_dim, action_dim=action_dim)
        result = _bc.load_state_dict(bc_state_dict, strict=False)
        if result.unexpected_keys:
            log.info(
                "Loaded BC backbone; skipped %d BC-only keys "
                "(metadata embeddings and head weights with different input size).",
                len(result.unexpected_keys),
            )

        self.features = _bc.features

        with torch.no_grad():
            feat_size = self.features(
                torch.zeros(1, 3, RESIZE_H, RESIZE_W)
            ).shape[1]

        self.style_embedding = nn.Embedding(n_styles, style_embed_dim)
        head_input_dim = feat_size + state_dim + style_embed_dim

        self.actor_head = nn.Sequential(
            nn.Linear(head_input_dim, 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, action_dim),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(head_input_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        if log_std_init is not None:
            self.log_std = nn.Parameter(
                torch.tensor(log_std_init, dtype=torch.float32)
            )
        else:
            self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def _feats(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        style: torch.Tensor,
    ) -> torch.Tensor:
        style_emb = self.style_embedding(style)
        return torch.cat([self.features(image), state, style_emb], dim=1)

    def get_value(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        style: torch.Tensor,
    ) -> torch.Tensor:
        return self.critic_head(self._feats(image, state, style)).squeeze(-1)

    def get_action_and_value(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        style: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action (or evaluate a given action) and return
        (action, log_prob, entropy, value)."""
        x = self._feats(image, state, style)
        raw = self.actor_head(x)
        std = torch.exp(self.log_std.clamp(-3.0, 1.0))
        dist = Normal(raw, std)

        if action is None:
            z = dist.rsample()
        else:
            eps = 1e-6
            z_steer = torch.atanh(action[:, 0:1].clamp(-1 + eps, 1 - eps))
            z_rest = torch.logit(action[:, 1:3].clamp(eps, 1 - eps))
            z = torch.cat([z_steer, z_rest], dim=1)

        steer = torch.tanh(z[:, 0:1])
        throttle = torch.sigmoid(z[:, 1:2])
        brake = torch.sigmoid(z[:, 2:3])
        act = torch.cat([steer, throttle, brake], dim=1)

        eps = 1e-6
        lp = dist.log_prob(z)
        lp[:, 0] = lp[:, 0] - torch.log(1.0 - steer[:, 0] ** 2 + eps)
        lp[:, 1] = lp[:, 1] - torch.log(
            throttle[:, 0] * (1.0 - throttle[:, 0]) + eps
        )
        lp[:, 2] = lp[:, 2] - torch.log(
            brake[:, 0] * (1.0 - brake[:, 0]) + eps
        )
        log_prob = lp.sum(dim=1)

        entropy = dist.entropy().sum(dim=1)
        value = self.critic_head(x).squeeze(-1)
        return act, log_prob, entropy, value


# ---------------------------------------------------------------------------
# MultiCamActorCritic
# ---------------------------------------------------------------------------

class MultiCamActorCritic(nn.Module):
    """Actor-critic policy for the multi-camera sensor suite.

    Uses the shared CNN backbone from MultiCamDriveNet. Actor and critic
    heads are freshly initialized (BC head weights cannot transfer because
    the multi-cam head input dimension differs from the BC single-cam head).
    A style embedding is concatenated with the fused visual features and
    state before both heads, identical to the single-camera ActorCritic.
    """

    def __init__(
        self,
        bc_state_dict: dict[str, torch.Tensor],
        dropout: float = 0.3,
        state_dim: int = 6,
        action_dim: int = 3,
        n_cameras: int = 3,
        n_styles: int = 3,
        style_embed_dim: int = 4,
        log_std_init: list[float] | None = None,
    ) -> None:
        super().__init__()
        from src.drivenet_multicam import MultiCamDriveNet

        _mc = MultiCamDriveNet(
            dropout=dropout, state_dim=state_dim,
            action_dim=action_dim, n_cameras=n_cameras,
        )
        # Load only the CNN backbone from the BC checkpoint. We filter to
        # 'features.*' keys to avoid a size-mismatch RuntimeError: the BC
        # head was built for a single camera, so its first Linear layer has a
        # different input dimension than the multi-camera head. strict=False
        # skips missing/unexpected keys but NOT size mismatches, so we must
        # pre-filter rather than rely on it to skip the head automatically.
        backbone_state = {
            k[len("features."):]: v
            for k, v in bc_state_dict.items()
            if k.startswith("features.")
        }
        _mc.shared_backbone.load_state_dict(backbone_state, strict=False)
        log.info(
            "MultiCamActorCritic: loaded %d backbone keys from BC checkpoint.",
            len(backbone_state),
        )

        self.shared_backbone = _mc.shared_backbone
        self.n_cameras = n_cameras

        with torch.no_grad():
            dummy = torch.zeros(1, 3, RESIZE_H, RESIZE_W)
            feat_size = self.shared_backbone(dummy).shape[1]

        self.style_embedding = nn.Embedding(n_styles, style_embed_dim)
        head_input_dim = feat_size * n_cameras + state_dim + style_embed_dim

        self.actor_head = nn.Sequential(
            nn.Linear(head_input_dim, 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, action_dim),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(head_input_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        if log_std_init is not None:
            self.log_std = nn.Parameter(
                torch.tensor(log_std_init, dtype=torch.float32)
            )
        else:
            self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def _feats(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        style: torch.Tensor,
    ) -> torch.Tensor:
        # images: (B, n_cameras, 3, H, W)
        cam_feats = [self.shared_backbone(images[:, i]) for i in range(self.n_cameras)]
        style_emb = self.style_embedding(style)
        return torch.cat(cam_feats + [state, style_emb], dim=1)

    def get_value(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        style: torch.Tensor,
    ) -> torch.Tensor:
        return self.critic_head(self._feats(images, state, style)).squeeze(-1)

    def get_action_and_value(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        style: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self._feats(images, state, style)
        raw = self.actor_head(x)
        std = torch.exp(self.log_std.clamp(-3.0, 1.0))
        dist = Normal(raw, std)

        if action is None:
            z = dist.rsample()
        else:
            eps = 1e-6
            z_steer = torch.atanh(action[:, 0:1].clamp(-1 + eps, 1 - eps))
            z_rest = torch.logit(action[:, 1:3].clamp(eps, 1 - eps))
            z = torch.cat([z_steer, z_rest], dim=1)

        steer = torch.tanh(z[:, 0:1])
        throttle = torch.sigmoid(z[:, 1:2])
        brake = torch.sigmoid(z[:, 2:3])
        act = torch.cat([steer, throttle, brake], dim=1)

        eps = 1e-6
        lp = dist.log_prob(z)
        lp[:, 0] = lp[:, 0] - torch.log(1.0 - steer[:, 0] ** 2 + eps)
        lp[:, 1] = lp[:, 1] - torch.log(
            throttle[:, 0] * (1.0 - throttle[:, 0]) + eps
        )
        lp[:, 2] = lp[:, 2] - torch.log(
            brake[:, 0] * (1.0 - brake[:, 0]) + eps
        )
        log_prob = lp.sum(dim=1)
        entropy = dist.entropy().sum(dim=1)
        value = self.critic_head(x).squeeze(-1)
        return act, log_prob, entropy, value


# ---------------------------------------------------------------------------
# RolloutBuffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Fixed-size pre-allocated buffer for on-policy rollout data."""

    def __init__(
        self,
        n_steps: int,
        obs_shape_img: tuple[int, ...],
        obs_shape_state: tuple[int, ...],
        action_dim: int,
        device: str = "cpu",
    ) -> None:
        self.n_steps = n_steps
        self.device = device

        self.obs_images = torch.zeros(n_steps, *obs_shape_img, dtype=torch.uint8)
        self.obs_states = torch.zeros(n_steps, *obs_shape_state)
        self.styles = torch.zeros(n_steps, dtype=torch.long)
        self.actions = torch.zeros(n_steps, action_dim)
        self.log_probs = torch.zeros(n_steps)
        self.rewards = torch.zeros(n_steps)
        self.values = torch.zeros(n_steps)
        self.dones = torch.zeros(n_steps)
        self.advantages = torch.zeros(n_steps)
        self.returns = torch.zeros(n_steps)

        self.ptr = 0
        self.full = False

    def add(
        self,
        obs_img: torch.Tensor,
        obs_state: torch.Tensor,
        style: int,
        action: torch.Tensor,
        log_prob: float,
        reward: float,
        value: float,
        done: float,
    ) -> None:
        self.obs_images[self.ptr] = (obs_img.squeeze(0) * 255).to(torch.uint8)
        self.obs_states[self.ptr] = obs_state.squeeze(0)
        self.styles[self.ptr] = style
        self.actions[self.ptr] = action.squeeze(0)
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = done
        self.ptr += 1
        if self.ptr == self.n_steps:
            self.full = True

    def compute_gae(
        self,
        last_value: float,
        last_done: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Compute generalized advantage estimates in-place."""
        last_gae = 0.0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1].item()
                next_value = self.values[t + 1].item()
            delta = (
                self.rewards[t].item()
                + gamma * next_value * next_non_terminal
                - self.values[t].item()
            )
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        self.returns = self.advantages + self.values

    def get_batches(self, batch_size: int):
        """Yield mini-batches of training data from the filled buffer."""
        assert self.full, "Buffer not full"
        adv = self.advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        indices = torch.randperm(self.n_steps)
        for start in range(0, self.n_steps, batch_size):
            idx = indices[start:start + batch_size]
            yield (
                self.obs_images[idx].float() / 255.0,
                self.obs_states[idx],
                self.styles[idx],
                self.actions[idx],
                self.log_probs[idx],
                adv[idx],
                self.returns[idx],
            )

    def reset(self) -> None:
        self.ptr = 0
        self.full = False


# ---------------------------------------------------------------------------
# PPO update step
# ---------------------------------------------------------------------------

def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    n_epochs: int,
    batch_size: int,
    clip_eps: float,
    entropy_coef: float,
    value_loss_coef: float,
    max_grad_norm: float,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, float, float, float]:
    """Run n_epochs passes over the rollout buffer with optional AMP.

    Returns
    -------
    tuple of (mean_policy_loss, mean_value_loss, mean_entropy, mean_approx_kl)
    """
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropy_bonuses: list[float] = []
    kl_divs: list[float] = []
    use_amp = scaler is not None

    for _ in range(n_epochs):
        for (img_b, state_b, style_b, act_b,
             old_lp_b, adv_b, ret_b) in buffer.get_batches(batch_size):
            img_b = img_b.to(device)
            state_b = state_b.to(device)
            style_b = style_b.to(device)
            act_b = act_b.to(device)
            old_lp_b = old_lp_b.to(device)
            adv_b = adv_b.to(device)
            ret_b = ret_b.to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                _, new_lp, entropy, new_value = model.get_action_and_value(
                    img_b, state_b, style_b, action=act_b,
                )
                ratio = torch.exp(new_lp - old_lp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * ((new_value - ret_b) ** 2).mean()
                neg_entropy = -entropy.mean()
                loss = (
                    policy_loss
                    + value_loss_coef * value_loss
                    + entropy_coef * neg_entropy
                )

            with torch.no_grad():
                approx_kl = (old_lp_b - new_lp).mean().item()
                kl_divs.append(approx_kl)

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_bonuses.append(entropy.mean().item())

    return (
        float(np.mean(policy_losses)),
        float(np.mean(value_losses)),
        float(np.mean(entropy_bonuses)),
        float(np.mean(kl_divs)),
    )
