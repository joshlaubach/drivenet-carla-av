"""
PPO primitives: ActorCritic network, RolloutBuffer, ppo_update, and
style-aware reward shaping.

Does not implement training orchestration -- see PPOAgent for that.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from src.drivenet import DriveNet
from src.preprocessing import RESIZE_H, RESIZE_W

log = logging.getLogger(__name__)


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
    """Modify *base_reward* with style-dependent jerk, speed, and lane-change terms.

    Parameters
    ----------
    base_reward : float
        Reward returned by ``CarlaEnv.step()``.
    speed_kmh, prev_speed_kmh, prev_prev_speed_kmh : float
        Three consecutive ego speed readings for jerk computation.
    lane_id, prev_lane_id : int
        Current and previous CARLA waypoint lane IDs.
        Use ``-1`` for the sentinel (first step of episode).
    style_weights : dict
        Keys: ``jerk_penalty``, ``speed_bonus``, ``lane_change_penalty``.
    dt : float
        Simulation timestep in seconds (default 0.05 = 20 FPS).

    Returns
    -------
    float
        Shaped reward.
    """
    reward = base_reward

    # -- Speed bonus (always active) --
    reward += (speed_kmh / 40.0) * style_weights["speed_bonus"]

    # -- Jerk penalty (needs ≥2 prior speed readings) --
    if prev_lane_id != -1 and prev_prev_speed_kmh >= 0.0:
        accel_now = (speed_kmh - prev_speed_kmh) / dt
        accel_prev = (prev_speed_kmh - prev_prev_speed_kmh) / dt
        jerk = abs(accel_now - accel_prev) / dt
        # Normalise jerk to keep penalty in a reasonable range
        jerk_norm = jerk / 1000.0
        reward -= jerk_norm * style_weights["jerk_penalty"]

    # -- Lane change penalty --
    if prev_lane_id != -1 and lane_id != prev_lane_id:
        reward -= style_weights["lane_change_penalty"]

    return reward


# ---------------------------------------------------------------------------
# ActorCritic
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """Actor-critic policy initialised from a BC DriveNet checkpoint.

    The actor head reuses the BC head weights; the critic head is a
    fresh 3-layer MLP over the same visual features.
    """

    def __init__(
        self,
        bc_state_dict: dict[str, torch.Tensor],
        dropout: float = 0.3,
        state_dim: int = 3,
        action_dim: int = 3,
        log_std_init: list[float] | None = None,
    ) -> None:
        super().__init__()
        _bc = DriveNet(dropout=dropout, state_dim=state_dim,
                       action_dim=action_dim)
        result = _bc.load_state_dict(bc_state_dict, strict=False)
        if result.unexpected_keys:
            log.info(
                "Skipped %d BC-only keys (e.g. metadata embeddings)",
                len(result.unexpected_keys),
            )

        self.features = _bc.features
        self.actor_head = _bc.head

        with torch.no_grad():
            feat_size = self.features(
                torch.zeros(1, 3, RESIZE_H, RESIZE_W)
            ).shape[1]

        self.critic_head = nn.Sequential(
            nn.Linear(feat_size + state_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        if log_std_init is not None:
            self.log_std = nn.Parameter(torch.tensor(log_std_init,
                                                     dtype=torch.float32))
        else:
            self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def _feats(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.features(image), state], dim=1)

    def get_value(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Return critic value estimate."""
        return self.critic_head(self._feats(image, state)).squeeze(-1)

    def get_action_and_value(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action (or evaluate given action) and return (act, log_prob, entropy, value)."""
        x = self._feats(image, state)
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
        lp[:, 0] = lp[:, 0] - torch.log(
            1.0 - steer[:, 0] ** 2 + eps
        )
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

        self.obs_images = torch.zeros(
            n_steps, *obs_shape_img, dtype=torch.uint8,
        )
        self.obs_states = torch.zeros(n_steps, *obs_shape_state)
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
        action: torch.Tensor,
        log_prob: float,
        reward: float,
        value: float,
        done: float,
    ) -> None:
        """Append one transition to the buffer."""
        self.obs_images[self.ptr] = (obs_img.squeeze(0) * 255).to(
            torch.uint8
        )
        self.obs_states[self.ptr] = obs_state.squeeze(0)
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
        """Compute generalised advantage estimates in-place."""
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
            last_gae = (
                delta + gamma * gae_lambda * next_non_terminal * last_gae
            )
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    def get_batches(
        self, batch_size: int
    ):
        """Yield mini-batches of (images, states, actions, log_probs, advantages, returns)."""
        assert self.full, "Buffer not full"
        adv = self.advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        indices = torch.randperm(self.n_steps)
        for start in range(0, self.n_steps, batch_size):
            idx = indices[start:start + batch_size]
            yield (
                self.obs_images[idx].float() / 255.0,
                self.obs_states[idx],
                self.actions[idx],
                self.log_probs[idx],
                adv[idx],
                self.returns[idx],
            )

    def reset(self) -> None:
        """Reset pointer so buffer can be refilled."""
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
) -> tuple[float, float, float]:
    """Run n_epochs passes over the rollout buffer with optional AMP. Returns mean losses."""
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropy_bonuses: list[float] = []
    use_amp = scaler is not None

    for _ in range(n_epochs):
        for (img_b, state_b, act_b,
             old_lp_b, adv_b, ret_b) in buffer.get_batches(batch_size):
            img_b = img_b.to(device)
            state_b = state_b.to(device)
            act_b = act_b.to(device)
            old_lp_b = old_lp_b.to(device)
            adv_b = adv_b.to(device)
            ret_b = ret_b.to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                _, new_lp, entropy, new_value = model.get_action_and_value(
                    img_b, state_b, action=act_b,
                )

                ratio = torch.exp(new_lp - old_lp_b)
                surr1 = ratio * adv_b
                surr2 = (
                    torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_b
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * ((new_value - ret_b) ** 2).mean()
                entropy_loss = -entropy.mean()
                loss = (
                    policy_loss
                    + value_loss_coef * value_loss
                    + entropy_coef * entropy_loss
                )

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
            entropy_bonuses.append(-entropy_loss.item())

    return (
        float(np.mean(policy_losses)),
        float(np.mean(value_losses)),
        float(np.mean(entropy_bonuses)),
    )
