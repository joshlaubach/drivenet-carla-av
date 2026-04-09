"""
EvaluationAgent -- WAT Framework / Workflow 04

Reads: workflows/04_evaluation.md
Sequences: model loading, CarlaEnv, make_weather, per-episode metric collection

Run once per town (hardware constraint). Appends results to eval_results.json
so partial runs across multiple towns merge into one file.

Loads all three driving-style PPO checkpoints per town when available.

Usage:
    from src.agents.eval_agent import EvaluationAgent
    agent = EvaluationAgent(town="Town03")
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
from scipy import stats

from src.carla_env import CarlaEnv
from src.carla_utils import make_weather
from src.config import load_config, require_keys
from src.drivenet import DriveNet
from src.ppo import ActorCritic
from src.preprocessing import RESIZE_H, RESIZE_W

log = logging.getLogger(__name__)

_DRIVING_STYLES = ["chill", "standard", "hurry"]


class EvaluationAgent:
    """Benchmarks all available models across the standardised evaluation grid
    for one town per run, per workflows/04_evaluation.md.

    Does not implement inference -- sequences model loading, CarlaEnv,
    and metric accumulation.
    """

    def __init__(
        self,
        town: str,
        models_dir: str = "models",
        results_dir: str = "results",
        host: str = "localhost",
        port: int = 2000,
        seed: int | None = None,
    ) -> None:
        self.cfg = load_config("eval")
        require_keys(
            self.cfg,
            ["eval_weathers", "episodes_per_condition",
             "max_steps_per_episode", "grp_sampling", "model_specs",
             "meta_dims", "crop"],
            "eval",
        )

        self.town = town
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.host = host
        self.port = port
        self.seed = seed if seed is not None else 42

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results_path = self.results_dir / "eval_results.json"

    # -- Public entry point ----------------------------------------------------

    def run(self) -> list[dict[str, Any]]:
        """Evaluate all models for self.town. Returns list of episode records."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.cfg

        models = self._load_models()
        if not models:
            raise RuntimeError(
                f"No model checkpoints found in {self.models_dir}. "
                "Run BehaviorCloningAgent and PPOAgent first."
            )

        env = CarlaEnv(
            host=self.host,
            port=self.port,
            town=self.town,
            image_width=400,
            image_height=300,
        )

        from agents.navigation.global_route_planner import GlobalRoutePlanner
        grp = GlobalRoutePlanner(
            env.world.get_map(), sampling_resolution=cfg["grp_sampling"]
        )

        episode_records: list[dict[str, Any]] = []
        rng = np.random.default_rng(self.seed)
        try:
            for spec, model in models:
                for weather in cfg["eval_weathers"]:
                    make_weather(env, weather)
                    for ep in range(cfg["episodes_per_condition"]):
                        record = self._run_episode(
                            env, grp, spec, model, weather, ep, rng
                        )
                        episode_records.append(record)
                        log.info(
                            "%s | %s | ep%d -- route=%.2f  collisions=%d  "
                            "lane=%.2f  speed=%.1f  dist=%.1fm",
                            spec["name"], weather, ep,
                            record["route_completion"],
                            record["collision_count"],
                            record["lane_keeping_frac"],
                            record["avg_speed_kmh"],
                            record["distance_m"],
                        )
        finally:
            env.close()

        self._append_results(episode_records)
        self._run_significance_tests()
        return episode_records

    # -- Single episode --------------------------------------------------------

    def _run_episode(
        self,
        env: CarlaEnv,
        grp: Any,
        spec: dict[str, Any],
        model: torch.nn.Module,
        weather: str,
        ep_index: int,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        """Execute one evaluation episode and return a metrics record."""
        cfg = self.cfg
        for attempt in range(3):
            try:
                obs, _ = env.reset()
                break
            except RuntimeError as exc:
                if attempt == 2:
                    log.error("reset() failed 3 times for %s/%s: %s", spec["name"], weather, exc)
                    return self._failed_record(spec, weather, ep_index)
                log.warning("reset() attempt %d failed: %s", attempt + 1, exc)

        # -- Sample a destination >50m away --
        spawn_points = env.world.get_map().get_spawn_points()
        ego_loc = env.vehicle.get_location()
        far_spawns = [
            sp for sp in spawn_points
            if sp.location.distance(ego_loc) > 50.0
        ]
        dest_transform = rng.choice(far_spawns if far_spawns else spawn_points)
        dest_loc = dest_transform.location

        # -- Plan route --
        try:
            route = grp.trace_route(ego_loc, dest_loc)
            route_locs = [wp.transform.location for wp, _ in route]
        except RuntimeError:
            route_locs = [ego_loc, dest_loc]

        if len(route_locs) < 2:
            route_locs = [ego_loc, dest_loc]

        route_total_arc = sum(
            route_locs[i].distance(route_locs[i + 1])
            for i in range(len(route_locs) - 1)
        )

        # -- Step loop --
        route_idx = 0
        total_distance = 0.0
        collision_count = 0
        lane_invasion_steps = 0
        speed_sum = 0.0
        total_steps = 0
        survived = True
        prev_loc = ego_loc

        for step_i in range(cfg["max_steps_per_episode"]):
            action = self._get_action(model, spec, obs)

            lane_flag_before = env._lane_invaded

            obs, _, terminated, truncated, info = env.step(action)
            total_steps += 1

            cur_loc = env.vehicle.get_location()
            dx = cur_loc.x - prev_loc.x
            dy = cur_loc.y - prev_loc.y
            step_dist = math.sqrt(dx ** 2 + dy ** 2)
            total_distance += step_dist
            prev_loc = cur_loc

            if info.get("collision", False):
                collision_count += 1

            if lane_flag_before:
                lane_invasion_steps += 1

            vel = env.vehicle.get_velocity()
            speed_sum += 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

            while route_idx < len(route_locs) and cur_loc.distance(route_locs[route_idx]) < 3.0:
                route_idx += 1

            reached_dest = (
                route_idx >= len(route_locs)
                or cur_loc.distance(route_locs[-1]) < 5.0
            )
            if reached_dest:
                break
            if terminated:
                survived = False
                break
            if truncated:
                break

        # -- Compute route completion --
        if route_idx < len(route_locs):
            arc_remaining = sum(
                route_locs[i].distance(route_locs[i + 1])
                for i in range(route_idx, len(route_locs) - 1)
            )
        else:
            arc_remaining = 0.0

        route_completion = 1.0 - (arc_remaining / (route_total_arc + 1e-6))
        route_completion = max(0.0, min(1.0, route_completion))

        avg_speed = speed_sum / max(total_steps, 1)
        lane_keeping_frac = 1.0 - (lane_invasion_steps / max(total_steps, 1))
        collision_rate = collision_count / max(total_distance / 100.0, 1e-6)

        return {
            "model": spec["name"],
            "model_type": spec["type"],
            "driving_style": spec.get("driving_style", "n/a"),
            "town": self.town,
            "weather": weather,
            "episode": ep_index,
            "route_completion": round(float(route_completion), 4),
            "collision_count": int(collision_count),
            "collision_rate": round(float(collision_rate), 4),
            "lane_keeping_frac": round(float(lane_keeping_frac), 4),
            "avg_speed_kmh": round(float(avg_speed), 2),
            "distance_m": round(float(total_distance), 2),
            "survived": bool(survived),
            "total_steps": int(total_steps),
        }

    def _get_action(
        self,
        model: torch.nn.Module,
        spec: dict[str, Any],
        obs: dict[str, Any],
    ) -> np.ndarray:
        """Run one forward pass through the model and return action as numpy."""
        crop_cfg = self.cfg["crop"]
        is_ppo = spec["type"] == "ppo"
        crop_top = crop_cfg["ppo_crop_top"] if is_ppo else crop_cfg["bc_crop_top"]
        crop_bot = crop_cfg["ppo_crop_bottom"] if is_ppo else crop_cfg["bc_crop_bottom"]

        img = obs["camera"][crop_top:crop_bot, :, :]
        img = cv2.resize(img, (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_AREA)
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_t = img_t.unsqueeze(0).to(self.device)
        state_t = torch.from_numpy(obs["state"]).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if is_ppo:
                action, _, _, _ = model.get_action_and_value(img_t, state_t)
                return action.squeeze(0).cpu().numpy()
            else:
                action = model(img_t, state_t, meta=None)
                return action.squeeze(0).cpu().numpy()

    # -- Model loading ---------------------------------------------------------

    def _load_models(self) -> list[tuple[dict[str, Any], torch.nn.Module]]:
        """Load BC models from config and discover style-specific PPO checkpoints."""
        cfg = self.cfg
        results: list[tuple[dict[str, Any], torch.nn.Module]] = []

        # BC models from config
        for spec in cfg["model_specs"]:
            path = self.models_dir / spec["file"]
            if not path.exists():
                log.warning("Checkpoint not found, skipping: %s", path)
                continue
            state_dict = torch.load(path, map_location="cpu")
            meta_dims = cfg["meta_dims"] if spec["meta"] else None
            model = DriveNet(
                dropout=0.3, state_dim=3, action_dim=3, meta_dims=meta_dims
            )
            model.load_state_dict(state_dict)
            model.eval().to(self.device)
            bc_spec = dict(spec)
            bc_spec["driving_style"] = "n/a"
            results.append((bc_spec, model))
            log.info("Loaded %s from %s.", spec["name"], path.name)

        # PPO models -- try style-specific checkpoints first
        loaded_any_style = False
        for style in _DRIVING_STYLES:
            ppo_file = f"ppo_{self.town}_{style}_best.pt"
            ppo_path = self.models_dir / ppo_file
            if ppo_path.exists():
                ppo_spec = {
                    "name": f"ppo_{self.town}_{style}",
                    "file": ppo_file,
                    "type": "ppo",
                    "meta": False,
                    "driving_style": style,
                }
                ppo_state = torch.load(ppo_path, map_location="cpu")
                _dummy_bc = DriveNet(dropout=0.3, state_dim=3, action_dim=3)
                ppo_model = ActorCritic(bc_state_dict=_dummy_bc.state_dict())
                ppo_model.load_state_dict(ppo_state)
                ppo_model.eval().to(self.device)
                results.append((ppo_spec, ppo_model))
                loaded_any_style = True
                log.info("Loaded ppo_%s_%s from %s.", self.town, style, ppo_file)

        # Fallback: load legacy ppo_{town}_best.pt as "standard" style
        if not loaded_any_style:
            legacy_file = f"ppo_{self.town}_best.pt"
            legacy_path = self.models_dir / legacy_file
            if legacy_path.exists():
                ppo_spec = {
                    "name": f"ppo_{self.town}",
                    "file": legacy_file,
                    "type": "ppo",
                    "meta": False,
                    "driving_style": "standard",
                }
                ppo_state = torch.load(legacy_path, map_location="cpu")
                _dummy_bc = DriveNet(dropout=0.3, state_dim=3, action_dim=3)
                ppo_model = ActorCritic(bc_state_dict=_dummy_bc.state_dict())
                ppo_model.load_state_dict(ppo_state)
                ppo_model.eval().to(self.device)
                results.append((ppo_spec, ppo_model))
                log.info(
                    "Loaded legacy ppo_%s from %s (mapped to style='standard').",
                    self.town, legacy_file,
                )
            else:
                log.warning("No PPO checkpoint for %s.", self.town)

        return results

    # -- Results persistence ---------------------------------------------------

    def _append_results(self, new_records: list[dict[str, Any]]) -> None:
        """Merge new episode records into eval_results.json, deduplicating by town."""
        existing: list[dict[str, Any]] = []
        if self.results_path.exists():
            with open(self.results_path) as f:
                existing = json.load(f)
        # Remove any existing records for this town to avoid duplicates on re-run
        existing = [r for r in existing if r.get("town") != self.town]
        all_records = existing + new_records
        with open(self.results_path, "w") as f:
            json.dump(all_records, f, indent=2)
        log.info(
            "Saved %d total episode records to %s.", len(all_records), self.results_path
        )

    def _run_significance_tests(self) -> None:
        """Run Mann-Whitney U test comparing PPO vs BC route completion."""
        if not self.results_path.exists():
            return
        with open(self.results_path) as f:
            records = json.load(f)
        if len(records) < 10:
            return

        ppo_rc = [r["route_completion"] for r in records if r["model_type"] == "ppo"]
        bc_rc = [r["route_completion"] for r in records if r["model_type"] == "bc"]
        if len(ppo_rc) < 2 or len(bc_rc) < 2:
            return

        _, p = stats.mannwhitneyu(ppo_rc, bc_rc, alternative="greater")
        summary_path = self.results_dir / "eval_summary.json"
        summary = {"ppo_vs_bc_mannwhitney_p": float(p)}
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        log.info("Mann-Whitney U test PPO vs BC: p=%.4f", p)

    @staticmethod
    def _failed_record(
        spec: dict[str, Any], weather: str, ep: int
    ) -> dict[str, Any]:
        """Return a zeroed-out record for an episode that could not run."""
        return {
            "model": spec["name"],
            "model_type": spec["type"],
            "driving_style": spec.get("driving_style", "n/a"),
            "town": "UNKNOWN",
            "weather": weather,
            "episode": ep,
            "route_completion": 0.0,
            "collision_count": 0,
            "collision_rate": 0.0,
            "lane_keeping_frac": 0.0,
            "avg_speed_kmh": 0.0,
            "distance_m": 0.0,
            "survived": False,
            "total_steps": 0,
        }
