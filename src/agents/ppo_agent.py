"""
PPOAgent -- WAT Framework / Workflow 03

Reads: workflows/03_ppo_finetuning.md
Sequences: ActorCritic, RolloutBuffer, ppo_update, CarlaEnv, preprocess_obs

Hardware constraint: no runtime map switching on RTX 5080 Blackwell.
Run once per town; curriculum switches weather only.

Supports three driving styles (chill/standard/hurry) via reward shaping.
Checkpoint naming: ppo_{town}_{style}_best.pt

Usage:
    from src.agents.ppo_agent import PPOAgent
    agent = PPOAgent(bc_checkpoint="models/BC_model_best.pt", town="Town03",
                     style="chill")
    agent.run()
"""

from __future__ import annotations

import json
import logging
import math
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.optim as optim

from src.carla_env import CarlaEnv
from src.carla_utils import make_weather
from src.config import load_config, require_keys
from src.ppo import ActorCritic, RolloutBuffer, compute_style_reward, ppo_update
from src.preprocessing import RESIZE_H, RESIZE_W

log = logging.getLogger(__name__)


class PPOAgent:
    """Fine-tunes a BC-initialized ActorCritic via PPO through live CARLA interaction.

    Does not implement PPO math -- sequences ActorCritic, RolloutBuffer,
    ppo_update, and CarlaEnv per workflows/03_ppo_finetuning.md.
    """

    def __init__(
        self,
        bc_checkpoint: str,
        town: str,
        style: str = "standard",
        save_dir: str = "models",
        results_dir: str = "results",
        host: str = "localhost",
        port: int = 2000,
        seed: int | None = None,
    ) -> None:
        self.cfg = load_config("ppo")
        require_keys(
            self.cfg,
            ["seed", "dropout", "cam_w", "cam_h", "lr", "clip_eps",
             "entropy_coef", "value_loss_coef", "n_steps", "batch_size",
             "n_epochs_ppo", "gamma", "gae_lambda", "total_timesteps",
             "curriculum_switch_step", "max_grad_norm", "crop_top",
             "crop_bottom", "weather_phase1", "weather_phase2",
             "reward_profiles"],
            "ppo",
        )

        if style not in self.cfg["reward_profiles"]:
            available = list(self.cfg["reward_profiles"].keys())
            raise KeyError(
                f"Unknown driving style '{style}'. "
                f"Available styles in configs/ppo.yaml: {available}"
            )

        self.bc_checkpoint = Path(bc_checkpoint)
        self.town = town
        self.style = style
        self.style_weights: dict[str, float] = self.cfg["reward_profiles"][style]
        self.save_dir = Path(save_dir)
        self.results_dir = Path(results_dir)
        self.host = host
        self.port = port
        self.seed = seed if seed is not None else self.cfg["seed"]

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(
            "PPOAgent [%s] using device: %s  style: %s",
            self.town, self.device, self.style,
        )

    # -- Public entry point ----------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run PPO fine-tuning for total_timesteps. Returns training history."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        cfg = self.cfg
        model = self._load_actor_critic()
        optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
        scaler = torch.amp.GradScaler("cuda") if self.device.type == "cuda" else None

        env = CarlaEnv(
            host=self.host,
            port=self.port,
            town=self.town,
            image_width=cfg["cam_w"],
            image_height=cfg["cam_h"],
        )

        history: dict[str, list[float]] = {
            "policy_losses": [], "value_losses": [], "entropy_bonuses": [],
            "kl_divs": [], "episode_rewards": [], "episode_lengths": [],
        }
        best_mean_reward = -math.inf
        checkpoint_path = self.save_dir / f"ppo_{self.town}_{self.style}_best.pt"

        try:
            history = self._training_loop(
                model, optimizer, scaler, env, history,
                best_mean_reward, checkpoint_path,
            )
        finally:
            env.close()

        self._save_results(history)
        return history

    # -- Training loop ---------------------------------------------------------

    def _training_loop(
        self,
        model: ActorCritic,
        optimizer: optim.Optimizer,
        scaler: torch.amp.GradScaler | None,
        env: CarlaEnv,
        history: dict[str, list[float]],
        best_mean_reward: float,
        checkpoint_path: Path,
    ) -> dict[str, list[float]]:
        """Core PPO collection-and-update loop with style reward shaping."""
        cfg = self.cfg
        crop_top: int = cfg["crop_top"]
        crop_bottom: int = cfg["crop_bottom"]

        buffer = RolloutBuffer(
            n_steps=cfg["n_steps"],
            obs_shape_img=(3, RESIZE_H, RESIZE_W),
            obs_shape_state=(3,),
            action_dim=3,
            device="cpu",
        )

        obs, _ = env.reset()
        current_weather = cfg["weather_phase1"][0]
        make_weather(env, current_weather)

        # Cache the CARLA map for lane-id queries (avoids repeated RPC)
        carla_map = env.world.get_map()

        total_steps = 0
        episode_reward = 0.0
        episode_length = 0
        episode_step = 0  # steps within current episode (for jerk warmup)
        recent_episode_rewards: list[float] = []

        # Style reward state
        prev_speed_kmh = 0.0
        prev_prev_speed_kmh = -1.0  # sentinel: no valid prev_prev yet
        prev_lane_id = -1

        done = False

        while total_steps < cfg["total_timesteps"]:
            # -- Collect n_steps of experience ---------------------------------
            buffer.reset()
            for _ in range(cfg["n_steps"]):
                img_t, state_t = self._preprocess(obs, crop_top, crop_bottom)
                img_t = img_t.to(self.device)
                state_t = state_t.to(self.device)

                with torch.no_grad():
                    action, log_prob, _, value = model.get_action_and_value(
                        img_t, state_t
                    )

                action_np = action.squeeze(0).cpu().numpy()
                next_obs, reward, terminated, truncated, info = env.step(action_np)
                done = terminated or truncated

                # -- Style reward shaping --
                vel = env.vehicle.get_velocity()
                speed_kmh = 3.6 * math.sqrt(
                    vel.x ** 2 + vel.y ** 2 + vel.z ** 2
                )
                try:
                    wp = carla_map.get_waypoint(env.vehicle.get_location())
                    lane_id = wp.lane_id
                except RuntimeError:
                    lane_id = prev_lane_id

                reward = compute_style_reward(
                    base_reward=reward,
                    speed_kmh=speed_kmh,
                    prev_speed_kmh=prev_speed_kmh,
                    prev_prev_speed_kmh=prev_prev_speed_kmh,
                    lane_id=lane_id,
                    prev_lane_id=prev_lane_id,
                    style_weights=self.style_weights,
                    dt=1.0 / 20.0,
                )

                prev_prev_speed_kmh = prev_speed_kmh
                prev_speed_kmh = speed_kmh
                prev_lane_id = lane_id
                episode_step += 1

                buffer.add(
                    img_t.cpu(), state_t.cpu(),
                    action.cpu(), log_prob.cpu().item(),
                    reward, value.cpu().item(), float(done),
                )

                episode_reward += reward
                episode_length += 1
                total_steps += 1

                if done:
                    recent_episode_rewards.append(episode_reward)
                    history["episode_rewards"].append(episode_reward)
                    history["episode_lengths"].append(episode_length)
                    episode_reward = 0.0
                    episode_length = 0
                    episode_step = 0

                    # Reset style state
                    prev_speed_kmh = 0.0
                    prev_prev_speed_kmh = -1.0
                    prev_lane_id = -1

                    current_weather = self._sample_weather(total_steps)
                    obs, _ = env.reset()
                    make_weather(env, current_weather)
                else:
                    obs = next_obs

                # Advance curriculum at switch step
                if total_steps == cfg["curriculum_switch_step"]:
                    log.info(
                        "[%s/%s] Curriculum switch at step %d -> Phase 2 (all weathers).",
                        self.town, self.style, total_steps,
                    )

            # -- Compute GAE and update ----------------------------------------
            img_t, state_t = self._preprocess(obs, crop_top, crop_bottom)
            img_t, state_t = img_t.to(self.device), state_t.to(self.device)
            with torch.no_grad():
                last_value = model.get_value(img_t, state_t).item()

            buffer.compute_gae(
                last_value=last_value,
                last_done=float(done),
                gamma=cfg["gamma"],
                gae_lambda=cfg["gae_lambda"],
            )

            p_loss, v_loss, ent, kl = ppo_update(
                model=model,
                optimizer=optimizer,
                buffer=buffer,
                n_epochs=cfg["n_epochs_ppo"],
                batch_size=cfg["batch_size"],
                clip_eps=cfg["clip_eps"],
                entropy_coef=cfg["entropy_coef"],
                value_loss_coef=cfg["value_loss_coef"],
                max_grad_norm=cfg["max_grad_norm"],
                device=self.device,
                scaler=scaler,
            )

            if math.isnan(p_loss) or math.isnan(v_loss):
                log.warning(
                    "[%s/%s] NaN loss at step %d -- skipping checkpoint.",
                    self.town, self.style, total_steps,
                )
            else:
                history["policy_losses"].append(p_loss)
                history["value_losses"].append(v_loss)
                history["entropy_bonuses"].append(ent)
                history["kl_divs"].append(kl)

            # -- Checkpoint on improved mean reward ----------------------------
            if len(recent_episode_rewards) >= 5:
                mean_reward = float(np.mean(recent_episode_rewards[-10:]))
                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    torch.save(model.state_dict(), checkpoint_path)
                    log.info(
                        "[%s/%s] Step %d -- new best mean_reward=%.2f -> %s",
                        self.town, self.style, total_steps,
                        mean_reward, checkpoint_path.name,
                    )

            if total_steps % 10_000 == 0:
                mean_r = (
                    float(np.mean(recent_episode_rewards[-10:]))
                    if recent_episode_rewards else 0.0
                )
                log.info(
                    "[%s/%s] Step %6d/%d  policy=%.4f  value=%.4f  "
                    "entropy=%.4f  kl=%.4f  mean_ep_r=%.2f",
                    self.town, self.style,
                    total_steps, cfg["total_timesteps"],
                    p_loss, v_loss, ent, kl, mean_r,
                )

        return history

    # -- Helpers ---------------------------------------------------------------

    def _load_actor_critic(self) -> ActorCritic:
        """Load BC checkpoint and build an ActorCritic model."""
        if not self.bc_checkpoint.exists():
            raise FileNotFoundError(
                f"BC checkpoint not found: {self.bc_checkpoint}. "
                "Run BehaviorCloningAgent first."
            )
        bc_state_dict = torch.load(self.bc_checkpoint, map_location="cpu")
        model = ActorCritic(
            bc_state_dict=bc_state_dict,
            dropout=self.cfg["dropout"],
        ).to(self.device)
        log.info(
            "[%s/%s] ActorCritic loaded from %s.",
            self.town, self.style, self.bc_checkpoint.name,
        )
        return model

    def _preprocess(
        self,
        obs: dict[str, Any],
        crop_top: int,
        crop_bottom: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Crop and tensorise a single CarlaEnv observation."""
        img = obs["camera"][crop_top:crop_bottom, :, :]
        img = cv2.resize(img, (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_AREA)
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_t = img_t.unsqueeze(0)
        state_t = torch.from_numpy(obs["state"]).unsqueeze(0)
        return img_t, state_t

    def _sample_weather(self, step: int) -> str:
        """Sample a weather preset from the current curriculum phase."""
        cfg = self.cfg
        pool = (
            cfg["weather_phase1"]
            if step < cfg["curriculum_switch_step"]
            else cfg["weather_phase2"]
        )
        return random.choice(pool)

    def _save_results(self, history: dict[str, list[float]]) -> None:
        """Persist training history and config snapshot."""
        hist_path = (
            self.results_dir
            / f"ppo_{self.town}_{self.style}_training_history.json"
        )
        serialisable = {
            k: [float(v) for v in vals] for k, vals in history.items()
        }
        with open(hist_path, "w") as f:
            json.dump(serialisable, f, indent=2)

        cfg_path = self.results_dir / "ppo_config.json"
        with open(cfg_path, "w") as f:
            json.dump(self.cfg, f, indent=2)

        log.info(
            "[%s/%s] PPO results saved to %s.",
            self.town, self.style, self.results_dir,
        )
