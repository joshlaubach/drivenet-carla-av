"""PPOAgent -- WAT Framework / Workflow 03

Reads: workflows/03_ppo_finetuning.md
Sequences: ActorCritic (or MultiCamActorCritic), RolloutBuffer, ppo_update,
           CarlaEnv, and CARLA process management.

Training cycles sequentially through four training towns (Town02, Town04,
Town06, Town10HD). Each cycle collects one rollout batch per town, then
updates the model. This lets one generalizing model learn across diverse
environments rather than specializing to a single town.

The weather curriculum expands from phase 1 (ClearNoon only) to phase 2
(all six presets) only when BOTH of the following are met:
  - Total steps >= curriculum_min_steps
  - Mean episode length over the last 20 episodes >= curriculum_perf_threshold

Supports three driving styles (chill / standard / hurry) via reward shaping.
All three styles share the same visual backbone; style is passed as an
integer token to a learned embedding layer in the policy.

One checkpoint is saved per style: models/ppo_{style}_best.pt

The agent manages the CARLA process lifecycle autonomously. You do not need
to start or stop CARLA manually. Each town gets its own fresh CARLA process
to avoid the in-place map-switch crash on RTX 5080 Blackwell hardware.

Usage:
    from src.agents.ppo_agent import PPOAgent
    agent = PPOAgent(
        bc_checkpoint="models/BC_model_best.pt",
        style="chill",
        sensor_suite="single_cam",
    )
    agent.run()
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.optim as optim

from src.carla_env import CarlaEnv
from src.carla_utils import kill_carla, launch_carla, make_weather, wait_for_carla
from src.config import load_config, require_keys
from src.ppo import (
    ActorCritic,
    MultiCamActorCritic,
    RolloutBuffer,
    compute_style_reward,
    ppo_update,
)
from src.preprocessing import RESIZE_H, RESIZE_W
from src.road_rule_monitor import RoadRuleMonitor

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CARLA_EXE = _PROJECT_ROOT / "CARLA_0.9.16" / "CarlaUE4.exe"


class PPOAgent:
    """Fine-tunes a BC-initialized policy via PPO across all training towns.

    Does not implement PPO math -- sequences ActorCritic, RolloutBuffer,
    ppo_update, and CarlaEnv per workflows/03_ppo_finetuning.md.
    """

    def __init__(
        self,
        bc_checkpoint: str,
        style: str = "standard",
        sensor_suite: str = "single_cam",
        save_dir: str = "models",
        results_dir: str = "results",
        host: str = "localhost",
        port: int = 2000,
        seed: int | None = None,
        carla_exe: str | Path | None = None,
    ) -> None:
        self.cfg = load_config("ppo")
        require_keys(
            self.cfg,
            ["seed", "dropout", "cam_w", "cam_h", "lr", "clip_eps",
             "entropy_coef", "value_loss_coef", "n_steps", "batch_size",
             "n_epochs_ppo", "gamma", "gae_lambda", "total_timesteps",
             "curriculum_min_steps", "curriculum_perf_threshold",
             "max_grad_norm", "crop_top", "crop_bottom",
             "weather_phase1", "weather_phase2", "reward_profiles",
             "training_towns", "style_codes"],
            "ppo",
        )

        if style not in self.cfg["reward_profiles"]:
            available = list(self.cfg["reward_profiles"].keys())
            raise KeyError(
                f"Unknown driving style '{style}'. "
                f"Available styles in configs/ppo.yaml: {available}"
            )
        if sensor_suite not in CarlaEnv.VALID_SUITES:
            raise ValueError(
                f"sensor_suite must be one of {CarlaEnv.VALID_SUITES}, "
                f"got '{sensor_suite}'."
            )

        self.bc_checkpoint = Path(bc_checkpoint)
        self.style = style
        self.style_token: int = self.cfg["style_codes"][style]
        self.style_weights: dict[str, float] = self.cfg["reward_profiles"][style]
        self.sensor_suite = sensor_suite
        self.save_dir = Path(save_dir)
        self.results_dir = Path(results_dir)
        self.host = host
        self.port = port
        self.seed = seed if seed is not None else self.cfg["seed"]
        self.carla_exe = Path(carla_exe) if carla_exe else _DEFAULT_CARLA_EXE

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Force CPU: simultaneous CUDA + DX12 CARLA rendering crashes the
        # RTX 5080 Blackwell GPU mid-rollout. CPU inference is 2.9 ms per
        # step -- not the bottleneck (CARLA simulation tick is).
        self.device = torch.device("cpu")
        log.info(
            "PPOAgent using device: %s  style: %s  sensor_suite: %s",
            self.device, self.style, self.sensor_suite,
        )

    # -- Public entry point ---------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run PPO fine-tuning for total_timesteps. Returns training history."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        model = self._load_actor_critic()
        optimizer = optim.Adam(model.parameters(), lr=self.cfg["lr"])
        scaler = (
            torch.amp.GradScaler("cuda") if self.device.type == "cuda" else None
        )

        history: dict[str, list[float]] = {
            "policy_losses": [], "value_losses": [], "entropy_bonuses": [],
            "kl_divs": [], "episode_rewards": [], "episode_lengths": [],
        }
        checkpoint_path = self.save_dir / f"ppo_{self.style}_{self.sensor_suite}_best.pt"
        best_mean_reward = -math.inf

        cfg = self.cfg
        total_steps = 0
        recent_ep_rewards: list[float] = []
        recent_ep_lengths: list[float] = []

        # Auto-resume: if a _resume.pt exists from a prior interrupted run,
        # restore all training state so the run continues from where it left off.
        resume = self._load_resume(model, optimizer, scaler)
        if resume:
            total_steps = resume["total_steps"]
            best_mean_reward = resume["best_mean_reward"]
            history = resume["history"]
            recent_ep_rewards = resume["recent_ep_rewards"]
            recent_ep_lengths = resume["recent_ep_lengths"]
            log.info(
                "[%s] Resumed from step %d (best reward %.2f).",
                self.style, total_steps, best_mean_reward,
            )

        while total_steps < cfg["total_timesteps"]:
            for town in cfg["training_towns"]:
                if total_steps >= cfg["total_timesteps"]:
                    break

                # Launch ONE CARLA per town and keep it running for ALL rollouts
                # in that town. Killing and relaunching between every 512-step
                # rollout causes DX12 memory conflicts on RTX 5080 Blackwell.
                proc = self._launch_carla()
                try:
                    self._wait_for_carla()
                    env = RoadRuleMonitor(CarlaEnv(
                        host=self.host,
                        port=self.port,
                        town=town,
                        image_width=cfg["cam_w"],
                        image_height=cfg["cam_h"],
                        sensor_suite=self.sensor_suite,
                    ))
                    try:
                        mean_ep_len = (
                            float(np.mean(recent_ep_lengths[-20:]))
                            if recent_ep_lengths else 0.0
                        )
                        weather = self._sample_weather(total_steps, mean_ep_len)
                        make_weather(env, weather)

                        # Collect all rollouts for this town in one CARLA session.
                        while total_steps < cfg["total_timesteps"]:
                            log.info(
                                "[%s] Collecting rollout in %s (step %d/%d).",
                                self.style, town, total_steps, cfg["total_timesteps"],
                            )
                            try:
                                steps_added, ep_rewards, ep_lengths = self._collect_rollout(
                                    model, env, town,
                                )
                            except RuntimeError as exc:
                                # CARLA crashed mid-rollout — break inner loop to
                                # relaunch CARLA and resume from current total_steps.
                                log.error(
                                    "[%s] Rollout failed (%s). Relaunching CARLA.", self.style, exc
                                )
                                break

                            total_steps += steps_added
                            recent_ep_rewards.extend(ep_rewards)
                            recent_ep_lengths.extend(ep_lengths)
                            history["episode_rewards"].extend(ep_rewards)
                            history["episode_lengths"].extend(ep_lengths)

                            # Update model immediately after each rollout.
                            p_loss, v_loss, ent, kl = self._update_model(
                                model, optimizer, scaler,
                            )
                            if not (math.isnan(p_loss) or math.isnan(v_loss)):
                                history["policy_losses"].append(p_loss)
                                history["value_losses"].append(v_loss)
                                history["entropy_bonuses"].append(ent)
                                history["kl_divs"].append(kl)

                            if len(recent_ep_rewards) >= 5:
                                mean_reward = float(np.mean(recent_ep_rewards[-10:]))
                                if mean_reward > best_mean_reward:
                                    best_mean_reward = mean_reward
                                    torch.save(model.state_dict(), checkpoint_path)
                                    log.info(
                                        "[%s] Step %d -- new best mean_reward=%.2f -> %s",
                                        self.style, total_steps,
                                        mean_reward, checkpoint_path.name,
                                    )

                            if total_steps % 10_000 < cfg["n_steps"]:
                                mean_r = (
                                    float(np.mean(recent_ep_rewards[-10:]))
                                    if recent_ep_rewards else 0.0
                                )
                                log.info(
                                    "[%s] Step %6d/%d  policy=%.4f  value=%.4f  "
                                    "entropy=%.4f  kl=%.4f  mean_ep_r=%.2f",
                                    self.style, total_steps, cfg["total_timesteps"],
                                    p_loss, v_loss, ent, kl, mean_r,
                                )
                                self._save_resume(
                                    model, optimizer, scaler,
                                    total_steps, best_mean_reward,
                                    history, recent_ep_rewards, recent_ep_lengths,
                                )

                    finally:
                        env.close()
                finally:
                    self._kill_carla(proc)
                    time.sleep(20.0)

        self._save_results(history)
        # Delete resume checkpoint — clean completion means no resume needed.
        self._delete_resume()
        return history

    # -- Rollout collection ----------------------------------------------------

    def _collect_rollout(
        self,
        model: ActorCritic | MultiCamActorCritic,
        env: CarlaEnv,
        town: str,
    ) -> tuple[int, list[float], list[float]]:
        """Collect one full rollout buffer of experience.

        Returns
        -------
        steps_added : int
        episode_rewards : list[float]
        episode_lengths : list[float]
        """
        cfg = self.cfg
        obs_shape_img = (
            (3, 3, RESIZE_H, RESIZE_W)
            if self.sensor_suite == "multi_cam"
            else (3, RESIZE_H, RESIZE_W)
        )

        self._buffer = RolloutBuffer(
            n_steps=cfg["n_steps"],
            obs_shape_img=obs_shape_img,
            obs_shape_state=(6,),
            action_dim=3,
            device="cpu",
        )

        style_t = torch.tensor(
            [self.style_token], dtype=torch.long, device=self.device
        )
        carla_map = env.world.get_map()

        obs, _ = env.reset()
        ep_reward = 0.0
        ep_length = 0
        ep_rewards: list[float] = []
        ep_lengths: list[float] = []

        prev_speed_kmh = 0.0
        prev_prev_speed_kmh = 0.0
        prev_lane_id = -1
        prev_steer = 0.0
        first_step = True

        for _ in range(cfg["n_steps"]):
            img_t, state_t = self._preprocess_obs(obs)
            img_t = img_t.to(self.device)
            state_t = state_t.to(self.device)

            with torch.no_grad():
                action, log_prob, _, value, z = model.get_action_and_value(
                    img_t, state_t, style_t
                )

            action_np = action.squeeze(0).cpu().numpy()
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            # Style reward shaping
            try:
                wp = carla_map.get_waypoint(env.vehicle.get_location())
                lane_id = wp.lane_id
            except RuntimeError:
                lane_id = prev_lane_id

            speed_kmh = info["speed_kmh"]
            current_steer = float(action_np[0])
            next_state = next_obs["state"]
            speed_limit_kmh = float(next_state[3]) * 130.0
            is_junction = bool(next_state[5])
            reward = compute_style_reward(
                base_reward=reward,
                speed_kmh=speed_kmh,
                prev_speed_kmh=prev_speed_kmh,
                prev_prev_speed_kmh=prev_prev_speed_kmh,
                lane_id=lane_id,
                prev_lane_id=prev_lane_id,
                style_weights=self.style_weights,
                dt=1.0 / 20.0,
                steer=current_steer,
                prev_steer=prev_steer,
                first_step=first_step,
                speed_limit_kmh=speed_limit_kmh,
                is_junction=is_junction,
            )

            prev_prev_speed_kmh = prev_speed_kmh
            prev_speed_kmh = speed_kmh
            prev_lane_id = lane_id
            prev_steer = current_steer
            first_step = False

            self._buffer.add(
                img_t.cpu(), state_t.cpu(),
                self.style_token,
                action.cpu(),
                z.cpu(),
                log_prob.cpu().item(),
                reward,
                value.cpu().item(),
                float(done),
            )

            ep_reward += reward
            ep_length += 1

            if done:
                ep_rewards.append(ep_reward)
                ep_lengths.append(float(ep_length))
                ep_reward = 0.0
                ep_length = 0
                prev_speed_kmh = 0.0
                prev_prev_speed_kmh = 0.0
                prev_lane_id = -1
                prev_steer = 0.0
                first_step = True
                obs, _ = env.reset()
            else:
                obs = next_obs

        # Bootstrap value for GAE
        img_t, state_t = self._preprocess_obs(obs)
        img_t, state_t = img_t.to(self.device), state_t.to(self.device)
        with torch.no_grad():
            last_value = model.get_value(img_t, state_t, style_t).item()

        self._buffer.compute_gae(
            last_value=last_value,
            last_done=float(done),
            gamma=cfg["gamma"],
            gae_lambda=cfg["gae_lambda"],
        )

        return cfg["n_steps"], ep_rewards, ep_lengths

    # -- Model update ----------------------------------------------------------

    def _update_model(
        self,
        model: ActorCritic | MultiCamActorCritic,
        optimizer: optim.Optimizer,
        scaler: torch.amp.GradScaler | None,
    ) -> tuple[float, float, float, float]:
        cfg = self.cfg
        return ppo_update(
            model=model,
            optimizer=optimizer,
            buffer=self._buffer,
            n_epochs=cfg["n_epochs_ppo"],
            batch_size=cfg["batch_size"],
            clip_eps=cfg["clip_eps"],
            entropy_coef=cfg["entropy_coef"],
            value_loss_coef=cfg["value_loss_coef"],
            max_grad_norm=cfg["max_grad_norm"],
            device=self.device,
            scaler=scaler,
        )

    # -- Observation preprocessing --------------------------------------------

    def _preprocess_obs(
        self, obs: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert a CarlaEnv observation to model-ready tensors.

        Returns
        -------
        img_t : Tensor
            (1, 3, H, W) for single_cam and lidar,
            (1, n_cameras, 3, H, W) for multi_cam.
        state_t : Tensor, shape (1, 6)
        """
        cfg = self.cfg
        crop_top: int = cfg["crop_top"]
        crop_bottom: int = cfg["crop_bottom"]
        state_t = torch.from_numpy(obs["state"]).unsqueeze(0)

        if self.sensor_suite == "single_cam":
            img = obs["camera"][crop_top:crop_bottom, :, :]
            img = cv2.resize(img, (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_AREA)
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            img_t = img_t.unsqueeze(0)

        elif self.sensor_suite == "multi_cam":
            imgs = []
            for i in range(3):
                cam = obs["cameras"][i, crop_top:crop_bottom, :, :]
                cam = cv2.resize(cam, (RESIZE_W, RESIZE_H),
                                 interpolation=cv2.INTER_AREA)
                imgs.append(
                    torch.from_numpy(cam).permute(2, 0, 1).float() / 255.0
                )
            img_t = torch.stack(imgs, dim=0).unsqueeze(0)  # (1, 3, 3, H, W)

        else:  # lidar -- BEV is already (H, W, 3) float32 in [0, 1]
            bev = obs["bev"]
            img_t = torch.from_numpy(bev).permute(2, 0, 1).float().unsqueeze(0)

        return img_t, state_t

    # -- Curriculum ------------------------------------------------------------

    def _sample_weather(self, total_steps: int, mean_ep_len: float) -> str:
        """Sample a weather preset from the active curriculum phase.

        Phase 2 (all weathers) activates only when the step count is above
        the minimum AND the policy has demonstrated sufficient competence
        (measured by mean episode length).
        """
        cfg = self.cfg
        min_steps_reached = total_steps >= cfg["curriculum_min_steps"]
        perf_reached = mean_ep_len >= cfg["curriculum_perf_threshold"]
        pool = (
            cfg["weather_phase2"]
            if (min_steps_reached and perf_reached)
            else cfg["weather_phase1"]
        )
        return random.choice(pool)

    # -- Model loading ---------------------------------------------------------

    def _load_actor_critic(self) -> ActorCritic | MultiCamActorCritic:
        """Load BC checkpoint and build the appropriate ActorCritic model."""
        if not self.bc_checkpoint.exists():
            raise FileNotFoundError(
                f"BC checkpoint not found: {self.bc_checkpoint}. "
                "Run BehaviorCloningAgent first."
            )
        bc_state = torch.load(self.bc_checkpoint, map_location="cpu")

        if self.sensor_suite == "multi_cam":
            model = MultiCamActorCritic(
                bc_state_dict=bc_state,
                dropout=self.cfg["dropout"],
            ).to(self.device)
        else:
            # single_cam and lidar both use the standard DriveNet CNN
            model = ActorCritic(
                bc_state_dict=bc_state,
                dropout=self.cfg["dropout"],
            ).to(self.device)

        log.info(
            "[%s] %s loaded from %s.",
            self.style,
            model.__class__.__name__,
            self.bc_checkpoint.name,
        )
        return model

    # -- CARLA lifecycle -------------------------------------------------------

    def _launch_carla(self):
        return launch_carla(self.carla_exe)

    def _wait_for_carla(self, max_wait: float = 60.0, poll_interval: float = 3.0) -> None:
        wait_for_carla(self.host, self.port, max_wait=max_wait, poll_interval=poll_interval)

    def _kill_carla(self, proc=None) -> None:
        kill_carla(proc)

    # -- Resume checkpoint -----------------------------------------------------

    def _resume_path(self) -> Path:
        return self.save_dir / f"ppo_{self.style}_{self.sensor_suite}_resume.pt"

    def _save_resume(
        self,
        model: ActorCritic | MultiCamActorCritic,
        optimizer: optim.Optimizer,
        scaler: torch.amp.GradScaler | None,
        total_steps: int,
        best_mean_reward: float,
        history: dict[str, list[float]],
        recent_ep_rewards: list[float],
        recent_ep_lengths: list[float],
    ) -> None:
        state = {
            "total_steps": total_steps,
            "best_mean_reward": best_mean_reward,
            "history": history,
            "recent_ep_rewards": recent_ep_rewards,
            "recent_ep_lengths": recent_ep_lengths,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        }
        torch.save(state, self._resume_path())
        log.debug("[%s] Resume checkpoint saved at step %d.", self.style, total_steps)

    def _load_resume(
        self,
        model: ActorCritic | MultiCamActorCritic,
        optimizer: optim.Optimizer,
        scaler: torch.amp.GradScaler | None,
    ) -> dict[str, Any] | None:
        path = self._resume_path()
        if not path.exists():
            return None
        state = torch.load(path, map_location=self.device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if scaler is not None and state.get("scaler_state_dict") is not None:
            scaler.load_state_dict(state["scaler_state_dict"])
        return {
            "total_steps": state["total_steps"],
            "best_mean_reward": state["best_mean_reward"],
            "history": state["history"],
            "recent_ep_rewards": state["recent_ep_rewards"],
            "recent_ep_lengths": state["recent_ep_lengths"],
        }

    def _delete_resume(self) -> None:
        path = self._resume_path()
        if path.exists():
            path.unlink()
            log.info("[%s] Resume checkpoint deleted (training complete).", self.style)

    # -- Persistence -----------------------------------------------------------

    def _save_results(self, history: dict[str, list[float]]) -> None:
        hist_path = self.results_dir / f"ppo_{self.style}_{self.sensor_suite}_training_history.json"
        serialisable = {k: [float(v) for v in vals] for k, vals in history.items()}
        with open(hist_path, "w") as f:
            json.dump(serialisable, f, indent=2)

        cfg_path = self.results_dir / "ppo_config.json"
        with open(cfg_path, "w") as f:
            json.dump(self.cfg, f, indent=2)

        log.info("[%s] PPO results saved to %s.", self.style, self.results_dir)
