"""EvaluationAgent -- WAT Framework / Workflow 04

Reads: workflows/04_evaluation.md
Sequences: model loading, CarlaEnv, make_weather, per-episode metric
collection, and side-by-side video rendering.

The agent evaluates all three sensor suites (single_cam, multi_cam, lidar)
across three eval towns (Town01, Town03, Town05) and ten episodes per
weather condition. It manages the CARLA process lifecycle autonomously, so
you can start Notebook 04 and step away.

For each evaluation episode, the agent records a side-by-side video showing
the model's input view (agent view) next to a third-person chase camera
(spectator view). Videos are saved to results/videos/.

All episode records are written to results/eval_results.json. Significance
tests comparing sensor suites are saved to results/eval_summary.json.

Usage:
    from src.agents.eval_agent import EvaluationAgent
    agent = EvaluationAgent(sensor_suite="single_cam")
    agent.run()
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import socket
import subprocess
import time
import weakref
from pathlib import Path
from queue import Queue
from typing import Any

import carla
import cv2
import numpy as np
import torch
from scipy import stats

from src.carla_env import CarlaEnv
from src.carla_utils import make_weather
from src.road_rule_monitor import RoadRuleMonitor
from src.config import load_config, require_keys
from src.drivenet import DriveNet
from src.drivenet_lidar import LidarDriveNet
from src.drivenet_multicam import MultiCamDriveNet
from src.ppo import ActorCritic, MultiCamActorCritic
from src.preprocessing import RESIZE_H, RESIZE_W

log = logging.getLogger(__name__)

_DRIVING_STYLES = ["chill", "standard", "hurry"]
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CARLA_EXE = _PROJECT_ROOT / "CARLA_0.9.16" / "CarlaUE4.exe"


class EvaluationAgent:
    """Evaluates all models for a given sensor suite across all eval towns.

    Manages the CARLA process lifecycle autonomously. Each town gets its own
    fresh CARLA process to avoid the in-place map-switch crash on RTX 5080
    Blackwell hardware.
    """

    def __init__(
        self,
        sensor_suite: str = "single_cam",
        models_dir: str = "models",
        results_dir: str = "results",
        host: str = "localhost",
        port: int = 2000,
        seed: int | None = None,
        carla_exe: str | Path | None = None,
        record_video: bool = True,
    ) -> None:
        self.cfg = load_config("eval")
        require_keys(
            self.cfg,
            ["eval_towns", "eval_weathers", "episodes_per_condition",
             "max_steps_per_episode", "grp_sampling", "bc_model_specs",
             "meta_dims", "crop", "video", "sensor_suites", "style_codes"],
            "eval",
        )

        if sensor_suite not in CarlaEnv.VALID_SUITES:
            raise ValueError(
                f"sensor_suite must be one of {CarlaEnv.VALID_SUITES}, "
                f"got '{sensor_suite}'."
            )

        self.sensor_suite = sensor_suite
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.host = host
        self.port = port
        self.seed = seed if seed is not None else 42
        self.carla_exe = Path(carla_exe) if carla_exe else _DEFAULT_CARLA_EXE
        self.record_video = record_video

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Force CPU: simultaneous CUDA + DX12 CARLA rendering crashes RTX 5080.
        self.device = torch.device("cpu")
        self.results_path = self.results_dir / "eval_results.json"

    # -- Public entry point ---------------------------------------------------

    def run(self) -> list[dict[str, Any]]:
        """Evaluate all models for self.sensor_suite across all eval towns.

        Returns
        -------
        list of episode records (one dict per episode)
        """
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "videos").mkdir(parents=True, exist_ok=True)

        models = self._load_models()
        if not models:
            raise RuntimeError(
                f"No model checkpoints found in {self.models_dir} for "
                f"sensor_suite='{self.sensor_suite}'. "
                "Run BehaviorCloningAgent and PPOAgent first."
            )

        all_records: list[dict[str, Any]] = []

        for town in self.cfg["eval_towns"]:
            log.info(
                "Evaluating %s in %s (%d models, %d weathers, %d episodes each).",
                self.sensor_suite, town,
                len(models), len(self.cfg["eval_weathers"]),
                self.cfg["episodes_per_condition"],
            )
            proc = self._launch_carla()
            try:
                self._wait_for_carla()
                env = RoadRuleMonitor(CarlaEnv(
                    host=self.host,
                    port=self.port,
                    town=town,
                    image_width=400,
                    image_height=300,
                    sensor_suite=self.sensor_suite,
                ))
                spectator_sensor = self._setup_spectator_camera(env)
                try:
                    from agents.navigation.global_route_planner import (
                        GlobalRoutePlanner,
                    )
                    grp = GlobalRoutePlanner(
                        env.world.get_map(),
                        sampling_resolution=self.cfg["grp_sampling"],
                    )
                    rng = np.random.default_rng(self.seed)

                    for spec, model in models:
                        for weather in self.cfg["eval_weathers"]:
                            make_weather(env, weather)
                            for ep in range(self.cfg["episodes_per_condition"]):
                                record, frames = self._run_episode(
                                    env, grp, spec, model, weather, ep, rng,
                                    town=town,
                                )
                                all_records.append(record)
                                log.info(
                                    "%s | %s | %s | ep%d -- "
                                    "route=%.2f  cols=%d  lane=%.2f  spd=%.1f",
                                    spec["name"], town, weather, ep,
                                    record["route_completion"],
                                    record["collision_count"],
                                    record["lane_keeping_frac"],
                                    record["avg_speed_kmh"],
                                )
                                if self.record_video and frames:
                                    self._write_video(
                                        frames, spec["name"], town, weather, ep
                                    )
                finally:
                    if spectator_sensor is not None:
                        try:
                            spectator_sensor.stop()
                            spectator_sensor.destroy()
                        except RuntimeError:
                            pass
                    env.close()
            finally:
                self._kill_carla(proc)
                time.sleep(6.0)

        self._append_results(all_records)
        self._run_significance_tests()
        return all_records

    # -- Spectator camera ------------------------------------------------------

    def _setup_spectator_camera(self, env: CarlaEnv) -> carla.Actor | None:
        """Attach a third-person chase camera to the ego vehicle.

        This camera is used only for video rendering and does not affect the
        model's input or the environment's observation space.
        """
        if not self.record_video:
            return None
        vcfg = self.cfg["video"]
        try:
            bpl = env.world.get_blueprint_library()
            cam_bp = bpl.find("sensor.camera.rgb")
            cam_bp.set_attribute("image_size_x", str(vcfg["spectator_w"]))
            cam_bp.set_attribute("image_size_y", str(vcfg["spectator_h"]))
            transform = carla.Transform(
                carla.Location(x=vcfg["spectator_x"], z=vcfg["spectator_z"]),
                carla.Rotation(pitch=vcfg["spectator_pitch"]),
            )
            sensor = env.world.spawn_actor(
                cam_bp, transform, attach_to=env.vehicle
            )
            self._spectator_queue: Queue = Queue(maxsize=10)
            weak_q = weakref.ref(self._spectator_queue)

            def _on_spec(image, wq=weak_q):
                q = wq()
                if q is not None:
                    try:
                        q.put_nowait(image)
                    except Exception:
                        pass

            sensor.listen(_on_spec)
            self._spectator_sensor = sensor
            return sensor
        except Exception as exc:
            log.warning("Could not set up spectator camera: %s", exc)
            return None

    def _get_spectator_frame(self, h: int, w: int) -> np.ndarray:
        """Retrieve the latest spectator camera frame."""
        try:
            while not self._spectator_queue.empty():
                img = self._spectator_queue.get_nowait()
            arr = np.frombuffer(img.raw_data, dtype=np.uint8)
            arr = arr.reshape((h, w, 4))[:, :, :3][:, :, ::-1].copy()
            return arr
        except Exception:
            return np.zeros((h, w, 3), dtype=np.uint8)

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
        town: str,
    ) -> tuple[dict[str, Any], list[np.ndarray]]:
        """Execute one evaluation episode.

        Returns
        -------
        record : dict  -- metrics for this episode
        frames : list of ndarray -- side-by-side video frames (may be empty)
        """
        cfg = self.cfg
        vcfg = cfg["video"]
        frames: list[np.ndarray] = []

        for attempt in range(3):
            try:
                obs, _ = env.reset()
                break
            except RuntimeError as exc:
                if attempt == 2:
                    log.error(
                        "reset() failed 3 times for %s/%s: %s",
                        spec["name"], weather, exc,
                    )
                    return self._failed_record(spec, weather, ep_index, town), []
                log.warning("reset() attempt %d failed: %s", attempt + 1, exc)

        # Sample a destination at least 50 m away
        spawn_points = env.world.get_map().get_spawn_points()
        ego_loc = env.vehicle.get_location()
        far_spawns = [sp for sp in spawn_points if sp.location.distance(ego_loc) > 50.0]
        dest_transform = rng.choice(far_spawns if far_spawns else spawn_points)
        dest_loc = dest_transform.location

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

        route_idx = 0
        total_distance = 0.0
        collision_count = 0
        lane_invasion_steps = 0
        speed_sum = 0.0
        total_steps = 0
        survived = True
        prev_loc = ego_loc
        viol_red_light = 0
        viol_wrong_way = 0
        viol_off_road = 0
        viol_double_solid = 0
        viol_speeding_steps = 0
        viol_tailgating_steps = 0
        viol_stop_sign = 0
        viol_solid_lane_steps = 0
        viol_yield = 0

        for _ in range(cfg["max_steps_per_episode"]):
            action = self._get_action(model, spec, obs)
            obs, _, terminated, truncated, info = env.step(action)
            total_steps += 1

            cur_loc = env.vehicle.get_location()
            dx, dy = cur_loc.x - prev_loc.x, cur_loc.y - prev_loc.y
            total_distance += math.sqrt(dx ** 2 + dy ** 2)
            prev_loc = cur_loc

            if info.get("collision", False):
                collision_count += 1
            if info.get("lane_invaded", False):
                lane_invasion_steps += 1
            rm = info.get("road_rule_monitor", {})
            if rm.get("red_light"):
                viol_red_light += 1
            if rm.get("wrong_way"):
                viol_wrong_way += 1
            if rm.get("off_road"):
                viol_off_road += 1
            if rm.get("double_solid_crossing"):
                viol_double_solid += 1
            if rm.get("speeding"):
                viol_speeding_steps += 1
            if rm.get("tailgating"):
                viol_tailgating_steps += 1
            if rm.get("stop_sign_violation"):
                viol_stop_sign += 1
            if rm.get("solid_lane_crossing"):
                viol_solid_lane_steps += 1
            if rm.get("failure_to_yield"):
                viol_yield += 1

            vel = env.vehicle.get_velocity()
            speed_sum += 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

            # Advance route waypoint tracker
            while (route_idx < len(route_locs)
                   and cur_loc.distance(route_locs[route_idx]) < 3.0):
                route_idx += 1

            # Record side-by-side frame
            if self.record_video:
                agent_frame = self._render_agent_view(obs)
                spec_frame = self._get_spectator_frame(
                    vcfg["spectator_h"], vcfg["spectator_w"]
                )
                frames.append(self._compose_frame(agent_frame, spec_frame))

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

        if route_idx < len(route_locs):
            arc_remaining = sum(
                route_locs[i].distance(route_locs[i + 1])
                for i in range(route_idx, len(route_locs) - 1)
            )
        else:
            arc_remaining = 0.0

        route_completion = max(
            0.0, min(1.0, 1.0 - arc_remaining / (route_total_arc + 1e-6))
        )
        avg_speed = speed_sum / max(total_steps, 1)
        lane_keeping_frac = 1.0 - lane_invasion_steps / max(total_steps, 1)
        collision_rate = collision_count / max(total_distance / 100.0, 1e-6)

        tier1_count = viol_red_light + viol_wrong_way + viol_off_road + viol_double_solid
        record = {
            "model": spec["name"],
            "model_type": spec["type"],
            "driving_style": spec.get("driving_style", "n/a"),
            "sensor_suite": self.sensor_suite,
            "town": town,
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
            # Road rule violation counts
            "viol_red_light": int(viol_red_light),
            "viol_wrong_way": int(viol_wrong_way),
            "viol_off_road": int(viol_off_road),
            "viol_double_solid": int(viol_double_solid),
            "viol_tier1_total": int(tier1_count),
            "viol_speeding_steps": int(viol_speeding_steps),
            "viol_tailgating_steps": int(viol_tailgating_steps),
            "viol_stop_sign": int(viol_stop_sign),
            "viol_solid_lane_steps": int(viol_solid_lane_steps),
            "viol_yield": int(viol_yield),
        }
        return record, frames

    # -- Action inference ------------------------------------------------------

    def _get_action(
        self,
        model: torch.nn.Module,
        spec: dict[str, Any],
        obs: dict[str, Any],
    ) -> np.ndarray:
        """Run one forward pass and return the action as a numpy array."""
        cfg = self.cfg
        crop_cfg = cfg["crop"]
        is_ppo = spec["type"] == "ppo"
        crop_top = crop_cfg["ppo_crop_top"] if is_ppo else crop_cfg["bc_crop_top"]
        crop_bot = crop_cfg["ppo_crop_bottom"] if is_ppo else crop_cfg["bc_crop_bottom"]

        state_t = torch.from_numpy(obs["state"]).unsqueeze(0).to(self.device)

        if self.sensor_suite == "single_cam":
            img = obs["camera"][crop_top:crop_bot, :, :]
            img = cv2.resize(img, (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_AREA)
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            img_t = img_t.unsqueeze(0).to(self.device)

        elif self.sensor_suite == "multi_cam":
            imgs = []
            for i in range(3):
                cam = obs["cameras"][i, crop_top:crop_bot, :, :]
                cam = cv2.resize(cam, (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_AREA)
                imgs.append(torch.from_numpy(cam).permute(2, 0, 1).float() / 255.0)
            img_t = torch.stack(imgs, dim=0).unsqueeze(0).to(self.device)

        else:  # lidar
            bev = obs["bev"]
            img_t = torch.from_numpy(bev).permute(2, 0, 1).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            if is_ppo:
                style_code = cfg["style_codes"].get(spec.get("driving_style", "standard"), 1)
                style_t = torch.tensor([style_code], dtype=torch.long, device=self.device)
                action, _, _, _ = model.get_action_and_value(img_t, state_t, style_t)
            else:
                action = model(img_t, state_t, meta=None)

        return action.squeeze(0).cpu().numpy()

    # -- Frame rendering -------------------------------------------------------

    def _render_agent_view(self, obs: dict[str, Any]) -> np.ndarray:
        """Return an upscaled version of what the model sees."""
        vcfg = self.cfg["video"]
        h, w = vcfg["agent_display_h"], vcfg["agent_display_w"]

        if self.sensor_suite == "single_cam":
            img = obs["camera"]
        elif self.sensor_suite == "multi_cam":
            img = obs["cameras"][0]  # front camera
        else:  # lidar
            img = (obs["bev"] * 255).astype(np.uint8)

        return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)

    def _compose_frame(
        self, agent_view: np.ndarray, spectator_view: np.ndarray
    ) -> np.ndarray:
        """Stack agent view and spectator view side by side into one frame."""
        h = agent_view.shape[0]
        spec_resized = cv2.resize(
            spectator_view,
            (self.cfg["video"]["spectator_w"], h),
            interpolation=cv2.INTER_AREA,
        )
        return np.concatenate([agent_view, spec_resized], axis=1)

    def _write_video(
        self,
        frames: list[np.ndarray],
        model_name: str,
        town: str,
        weather: str,
        ep: int,
    ) -> None:
        """Write a list of BGR frames to an mp4 video file."""
        if not frames:
            return
        fps = self.cfg["video"]["fps"]
        h, w = frames[0].shape[:2]
        video_dir = self.results_dir / "videos"
        filename = video_dir / f"{model_name}_{self.sensor_suite}_{town}_{weather}_ep{ep:02d}.mp4"
        writer = cv2.VideoWriter(
            str(filename),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )
        for frame in frames:
            # OpenCV expects BGR; our frames are RGB
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        log.info("Video saved: %s", filename.name)

    # -- Model loading ---------------------------------------------------------

    def _load_models(self) -> list[tuple[dict[str, Any], torch.nn.Module]]:
        """Load BC and PPO checkpoints for the current sensor suite."""
        cfg = self.cfg
        results: list[tuple[dict[str, Any], torch.nn.Module]] = []

        # BC model
        for spec in cfg["bc_model_specs"]:
            if spec["sensor_suite"] != self.sensor_suite:
                continue
            path = self.models_dir / spec["file"]
            if not path.exists():
                log.warning("BC checkpoint not found, skipping: %s", path)
                continue
            state = torch.load(path, map_location="cpu")
            model = self._build_bc_model(state)
            model.eval().to(self.device)
            bc_spec = {
                "name": spec["name"],
                "type": "bc",
                "driving_style": "n/a",
                "sensor_suite": self.sensor_suite,
            }
            results.append((bc_spec, model))
            log.info("Loaded BC model from %s.", path.name)

        # PPO models (one per style)
        for style in _DRIVING_STYLES:
            ppo_file = f"ppo_{style}_{self.sensor_suite}_best.pt"
            ppo_path = self.models_dir / ppo_file
            if not ppo_path.exists():
                log.warning("PPO checkpoint not found, skipping: %s", ppo_file)
                continue
            state = torch.load(ppo_path, map_location="cpu")
            model = self._build_ppo_model(state)
            model.eval().to(self.device)
            ppo_spec = {
                "name": f"ppo_{style}_{self.sensor_suite}",
                "type": "ppo",
                "driving_style": style,
                "sensor_suite": self.sensor_suite,
            }
            results.append((ppo_spec, model))
            log.info("Loaded PPO %s/%s from %s.", style, self.sensor_suite, ppo_file)

        return results

    def _build_bc_model(self, state_dict: dict) -> torch.nn.Module:
        """Instantiate and load a BC model for the current sensor suite."""
        meta_dims = self.cfg["meta_dims"]
        if self.sensor_suite == "single_cam":
            model = DriveNet(dropout=0.3, state_dim=6, meta_dims=meta_dims)
        elif self.sensor_suite == "multi_cam":
            model = MultiCamDriveNet(dropout=0.3, state_dim=6, meta_dims=meta_dims)
        else:  # lidar
            model = LidarDriveNet(dropout=0.3, state_dim=6, meta_dims=meta_dims)
        model.load_state_dict(state_dict, strict=False)
        return model

    def _build_ppo_model(self, state_dict: dict) -> torch.nn.Module:
        """Instantiate and load a PPO model for the current sensor suite."""
        dummy_bc = DriveNet().state_dict()
        if self.sensor_suite == "multi_cam":
            model = MultiCamActorCritic(bc_state_dict=dummy_bc)
        else:
            model = ActorCritic(bc_state_dict=dummy_bc)
        model.load_state_dict(state_dict, strict=False)
        return model

    # -- Results persistence ---------------------------------------------------

    def _append_results(self, new_records: list[dict[str, Any]]) -> None:
        """Merge new episode records into eval_results.json."""
        existing: list[dict[str, Any]] = []
        if self.results_path.exists():
            with open(self.results_path) as f:
                existing = json.load(f)
        # Remove stale records for this sensor suite to avoid duplicates on re-run
        existing = [
            r for r in existing if r.get("sensor_suite") != self.sensor_suite
        ]
        all_records = existing + new_records
        with open(self.results_path, "w") as f:
            json.dump(all_records, f, indent=2)
        log.info(
            "Saved %d total records to %s.", len(all_records), self.results_path
        )

    def _run_significance_tests(self) -> None:
        """Compare sensor suites on route completion via Mann-Whitney U test."""
        if not self.results_path.exists():
            return
        with open(self.results_path) as f:
            records = json.load(f)
        if len(records) < 10:
            return

        summary: dict[str, Any] = {}
        suites = list({r["sensor_suite"] for r in records})
        for i, s1 in enumerate(suites):
            for s2 in suites[i + 1:]:
                rc1 = [r["route_completion"] for r in records if r["sensor_suite"] == s1]
                rc2 = [r["route_completion"] for r in records if r["sensor_suite"] == s2]
                if len(rc1) >= 2 and len(rc2) >= 2:
                    _, p = stats.mannwhitneyu(rc1, rc2, alternative="two-sided")
                    key = f"mannwhitney_{s1}_vs_{s2}_p"
                    summary[key] = round(float(p), 6)
                    log.info("Mann-Whitney U %s vs %s: p=%.4f", s1, s2, p)

        summary_path = self.results_dir / "eval_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    # -- CARLA lifecycle -------------------------------------------------------

    def _launch_carla(self) -> subprocess.Popen:
        if not self.carla_exe.exists():
            raise FileNotFoundError(
                f"CARLA executable not found: {self.carla_exe}."
            )
        cmd = [
            str(self.carla_exe),
            "-dx12", "-quality-level=Low", "-fps=20",
            "-benchmark", "-windowed", "-ResX=800", "-ResY=600",
            "-nosound", "-NoSplash",
        ]
        env_vars = dict(os.environ)
        env_vars["DXGI_GPU_PREFERENCE"] = "2"
        proc = subprocess.Popen(
            cmd, env=env_vars,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        log.info("CARLA process started (PID %d).", proc.pid)
        return proc

    def _wait_for_carla(
        self, max_wait: float = 40.0, poll_interval: float = 3.0
    ) -> None:
        deadline = time.time() + max_wait
        while time.time() < deadline:
            try:
                with socket.create_connection((self.host, self.port), timeout=2.0):
                    log.info("CARLA is reachable.")
                    time.sleep(5.0)
                    return
            except (ConnectionRefusedError, OSError):
                time.sleep(poll_interval)
        raise TimeoutError(
            f"CARLA did not become reachable within {max_wait:.0f}s."
        )

    def _kill_carla(self, proc: subprocess.Popen | None = None) -> None:
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except (subprocess.TimeoutExpired, OSError):
                proc.kill()
            log.info("Terminated CARLA PID %d.", proc.pid)
        for exe in ["CarlaUE4-Win64-Shipping.exe", "CarlaUE4.exe"]:
            try:
                subprocess.run(
                    ["taskkill", "/F", "/IM", exe],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError:
                pass

    # -- Helpers ---------------------------------------------------------------

    @staticmethod
    def _failed_record(
        spec: dict[str, Any], weather: str, ep: int, town: str
    ) -> dict[str, Any]:
        return {
            "model": spec["name"],
            "model_type": spec["type"],
            "driving_style": spec.get("driving_style", "n/a"),
            "sensor_suite": spec.get("sensor_suite", "unknown"),
            "town": town,
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
