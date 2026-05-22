"""
DataCollectionAgent -- WAT Framework / Workflow 01

Reads: workflows/01_data_collection.md
Sequences: CarlaEnv, autopilot, NPC spawning, np.savez_compressed

Supports two modes:
  - run(town)      : collect one town (CARLA must already be running with that map)
  - run_all_towns(): autonomously launch/shutdown CARLA per town, collecting
                     all 6 towns back-to-back with no manual intervention.

A dedicated follow car trails the ego vehicle during collection to produce
more realistic traffic patterns (visible in mirrors, affects ego autopilot
lane-change decisions).

Hardware constraint: no runtime map switching on RTX 5080 Blackwell.

Usage:
    from src.agents.collection_agent import DataCollectionAgent
    agent = DataCollectionAgent(town="Town03")
    agent.run()                  # single town (CARLA pre-launched)
    agent.run_all_towns()        # all 6 towns autonomously
"""

from __future__ import annotations

import logging
import math
import random
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import carla
import numpy as np

from src.carla_env import CarlaEnv
from src.config import load_config, require_keys

log = logging.getLogger(__name__)

# Default CARLA executable path relative to project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CARLA_EXE = _PROJECT_ROOT / "CARLA_0.9.16" / "CarlaUE4.exe"

# All towns in collection order
_ALL_TOWNS = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD"]

# Follow-car parameters
_FOLLOW_DISTANCE_M = 8.0     # target gap behind ego (metres)
_FOLLOW_SPEED_GAIN = 0.04    # P-gain for throttle control
_FOLLOW_STEER_GAIN = 1.5     # P-gain for steering toward ego waypoint
_FOLLOW_MAX_RESPAWN_DIST = 50.0  # respawn follower if it falls this far behind


class DataCollectionAgent:
    """Coordinates expert data collection for one town across all 54 conditions.

    Does not drive the vehicle -- sequences CarlaEnv, CARLA autopilot, and
    numpy persistence tools per the spec in workflows/01_data_collection.md.

    The optional follow car is a PID-controlled vehicle that maintains a
    configurable gap behind the ego, creating more realistic rear-traffic
    interactions visible during turns and lane changes.
    """

    def __init__(
        self,
        town: str = "Town01",
        data_dir: str = "data",
        frames_per_condition: int | None = None,
        chunk_size: int | None = None,
        host: str = "localhost",
        port: int = 2000,
        seed: int | None = None,
        carla_exe: str | Path | None = None,
        enable_follow_car: bool = True,
        enable_viz: bool = False,
    ) -> None:
        self.cfg = load_config("collection")
        require_keys(
            self.cfg,
            ["weather_presets", "weather_params", "tod_sun_angles",
             "traffic_vehicle_counts", "frames_per_condition", "chunk_size",
             "image_width", "image_height", "seed"],
            "collection",
        )

        self.town = town
        self.data_dir_root = Path(data_dir)
        self.data_dir = self.data_dir_root / town
        self.frames_per_condition = (
            frames_per_condition if frames_per_condition is not None
            else self.cfg["frames_per_condition"]
        )
        self.chunk_size = (
            chunk_size if chunk_size is not None
            else self.cfg["chunk_size"]
        )
        self.host = host
        self.port = port
        self.enable_follow_car = enable_follow_car

        self.carla_exe    = Path(carla_exe) if carla_exe else _DEFAULT_CARLA_EXE
        self.enable_viz   = enable_viz
        self._viz         = None  # DriveNetVisualizer, created lazily in run()

        _seed = seed if seed is not None else self.cfg["seed"]
        random.seed(_seed)
        np.random.seed(_seed)

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._chunk_index = 0
        self._buffer = _FrameBuffer()
        self._collision_log: list[dict[str, Any]] = []

        # Follow car actor reference (managed per-episode)
        self._follow_car: carla.Actor | None = None

    # -- Public entry points ---------------------------------------------------

    def run(self) -> None:
        """Collect all 54 conditions for self.town.

        Requires CARLA to be running with the target town already loaded.
        """
        if self.enable_viz:
            from src.visualizer import DriveNetVisualizer
            self._viz = DriveNetVisualizer()

        env = CarlaEnv(
            host=self.host,
            port=self.port,
            town=self.town,
            image_width=self.cfg["image_width"],
            image_height=self.cfg["image_height"],
        )
        try:
            self._collect_all_conditions(env)
        finally:
            log.info("run(): collect loop done -- destroying follow car ...")
            self._destroy_follow_car()
            log.info("run(): flushing buffer ...")
            self._flush_buffer()
            log.info("run(): saving collision log ...")
            self._save_collision_log()
            log.info("run(): closing env ...")
            env.close()
            log.info("run(): env closed")
            if self._viz is not None:
                self._viz.close()
                self._viz = None
        log.info(
            "Collection complete for %s. %d chunks saved to %s.",
            self.town, self._chunk_index, self.data_dir,
        )

    def run_all_towns(
        self,
        towns: list[str] | None = None,
        startup_wait: float = 40.0,
        shutdown_wait: float = 8.0,
    ) -> dict[str, int]:
        """Autonomously collect all towns by launching/shutting CARLA per town.

        Parameters
        ----------
        towns : list[str] | None
            Towns to collect. Defaults to all 6 towns.
        startup_wait : float
            Seconds to wait after launching CARLA before connecting.
        shutdown_wait : float
            Seconds to wait after killing CARLA before relaunching.

        Returns
        -------
        dict[str, int]
            Mapping of town name to number of chunks saved.
        """
        if not self.carla_exe.exists():
            raise FileNotFoundError(
                f"CARLA executable not found: {self.carla_exe}. "
                "Set carla_exe= in __init__ or place CARLA at CARLA_0.9.16/."
            )

        towns = towns or _ALL_TOWNS
        results: dict[str, int] = {}

        for town in towns:
            log.info("=" * 60)
            log.info("Starting collection for %s", town)
            log.info("=" * 60)

            # Reconfigure for this town
            self.town = town
            self.data_dir = self.data_dir_root / town
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self._chunk_index = 0
            self._buffer = _FrameBuffer()
            self._collision_log = []

            # Launch CARLA, collect, shut down
            self._kill_carla()
            time.sleep(shutdown_wait)

            carla_proc = self._launch_carla(town)
            try:
                self._wait_for_carla(startup_wait)
                self._load_town(town)
                self.run()
                results[town] = self._chunk_index
            except Exception as exc:
                log.error("Collection failed for %s: %s", town, exc)
                results[town] = self._chunk_index
            finally:
                self._kill_carla(carla_proc)
                time.sleep(shutdown_wait)

        log.info("All-town collection complete: %s", results)
        return results

    # -- CARLA process management ----------------------------------------------

    def _launch_carla(self, _town: str) -> subprocess.Popen:
        """Launch a CARLA server process in the background.

        *_town* is accepted for call-site clarity but is not passed on the
        command line -- CARLA ignores CLI map args.  The map is loaded
        afterward via ``_load_town()``.

        Uses the same flags as scripts/launch_carla.bat: -dx12, low quality,
        20 FPS, benchmark mode, small viewport, no sound.
        """
        cmd = [
            str(self.carla_exe),
            "-dx12",
            "-quality-level=Low",
            "-fps=20",
            "-benchmark",
            "-windowed",
            "-ResX=800",
            "-ResY=600",
            "-nosound",
            "-NoSplash",
        ]
        log.info("Launching CARLA: %s", " ".join(cmd))

        env_vars = dict(__import__("os").environ)
        env_vars["DXGI_GPU_PREFERENCE"] = "2"

        proc = subprocess.Popen(
            cmd,
            env=env_vars,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log.info("CARLA process started (PID %d).", proc.pid)
        return proc

    def _wait_for_carla(
        self,
        max_wait: float = 40.0,
        poll_interval: float = 3.0,
    ) -> None:
        """Block until CARLA accepts TCP connections on self.port."""
        deadline = time.time() + max_wait
        log.info(
            "Waiting up to %.0fs for CARLA on %s:%d ...",
            max_wait, self.host, self.port,
        )
        while time.time() < deadline:
            try:
                with socket.create_connection(
                    (self.host, self.port), timeout=2.0
                ):
                    log.info("CARLA is reachable.")
                    # Extra settle time for world initialization
                    time.sleep(5.0)
                    return
            except (ConnectionRefusedError, OSError):
                time.sleep(poll_interval)

        raise TimeoutError(
            f"CARLA did not become reachable on {self.host}:{self.port} "
            f"within {max_wait:.0f}s."
        )

    def _load_town(self, town: str) -> None:
        """Load the target town map via client.load_world().

        This is called immediately after a fresh CARLA launch -- the
        load_world crash only happens when switching maps on a running
        server, not on the first load after startup.
        """
        client = carla.Client(self.host, self.port)
        client.set_timeout(30.0)
        current_map = client.get_world().get_map().name
        current_short = current_map.split("/")[-1]

        if town in current_short or current_short in town:
            log.info("Town %s already loaded (%s).", town, current_map)
            return

        log.info("Loading town %s (current: %s) ...", town, current_map)
        client.load_world(town)
        # Wait for the new world to stabilise
        time.sleep(5.0)
        new_map = client.get_world().get_map().name
        log.info("Town loaded: %s", new_map)

    def _kill_carla(self, proc: subprocess.Popen | None = None) -> None:
        """Terminate CARLA server process(es).

        Kills the specific process if given, plus any stray CARLA instances
        via taskkill (Windows).
        """
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except (subprocess.TimeoutExpired, OSError):
                proc.kill()
            log.info("Terminated CARLA process PID %d.", proc.pid)

        # Belt-and-suspenders: kill any leftover CARLA processes
        for exe_name in ["CarlaUE4-Win64-Shipping.exe", "CarlaUE4.exe"]:
            try:
                subprocess.run(
                    ["taskkill", "/F", "/IM", exe_name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError:
                pass  # Not on Windows

    # -- Condition iteration ---------------------------------------------------

    def _collect_all_conditions(self, env: CarlaEnv) -> None:
        """Iterate over the full 54-condition grid for the current town."""
        weather_presets: list[str] = self.cfg["weather_presets"]
        tod_sun_angles: dict[str, float] = self.cfg["tod_sun_angles"]
        traffic_counts: dict[str, int] = self.cfg["traffic_vehicle_counts"]

        conditions = [
            (weather, tod, traffic)
            for weather in weather_presets
            for tod in tod_sun_angles
            for traffic in traffic_counts
        ]
        for i, (weather, tod, traffic) in enumerate(conditions):
            log.info(
                "[%d/54] town=%s weather=%s tod=%s traffic=%s",
                i + 1, self.town, weather, tod, traffic,
            )
            self._apply_weather(env, weather, tod)
            npcs = self._spawn_npcs(env, traffic_counts[traffic])
            try:
                self._collect_condition(env, weather, tod, traffic, i + 1)
            finally:
                self._destroy_npcs(npcs)

    # -- Single-condition collection -------------------------------------------

    def _collect_condition(
        self,
        env: CarlaEnv,
        weather: str,
        tod: str,
        traffic: str,
        condition_idx: int = 0,
    ) -> None:
        """Collect frames_per_condition frames for one weather/tod/traffic combo."""
        frames_saved = 0
        attempts = 0

        while frames_saved < self.frames_per_condition:
            attempts += 1
            if attempts > 10:
                log.warning(
                    "Giving up on %s/%s/%s after 10 reset attempts (%d/%d frames).",
                    weather, tod, traffic, frames_saved, self.frames_per_condition,
                )
                break
            try:
                obs, _ = env.reset()
            except RuntimeError as exc:
                log.warning("reset() failed (attempt %d): %s", attempts, exc)
                continue

            # Reattach viz cameras to the newly spawned vehicle
            if self._viz is not None:
                self._viz.reattach(env.world, env.vehicle)

            env.vehicle.set_autopilot(True, env.traffic_manager.get_port())

            # Spawn follow car behind the ego
            if self.enable_follow_car:
                self._spawn_follow_car(env)

            while frames_saved < self.frames_per_condition:
                try:
                    # Update follow car before tick so it reacts to ego's
                    # current position and both advance in the same tick
                    if self.enable_follow_car and self._follow_car is not None:
                        self._update_follow_car(env)

                    env.world.tick()
                    obs = env._get_observation()
                except RuntimeError as exc:
                    log.warning("tick/obs failed: %s -- resetting.", exc)
                    break

                if self._viz is not None:
                    self._viz.update({
                        "town":         self.town,
                        "weather":      weather,
                        "tod":          tod,
                        "traffic":      traffic,
                        "condition":    condition_idx,
                        "frames_saved": frames_saved,
                        "frames_total": self.frames_per_condition,
                        "chunks_saved": self._chunk_index,
                        "speed_kmh":    env._speed_kmh,
                    })

                frame = self._read_frame(env, obs, weather, tod, traffic)
                self._buffer.append(frame)
                frames_saved += 1

                if self._buffer.size() >= self.chunk_size:
                    self._flush_buffer()

                if env._collision_flag:
                    env._collision_flag = False
                    env._collision_intensity = 0.0
                    self._collision_log.append(
                        {"weather": weather, "tod": tod, "traffic": traffic,
                         "frame": frames_saved}
                    )
                    self._destroy_follow_car()
                    break  # reset ego; continue accumulating frames

            # Clean up follow car between episodes
            self._destroy_follow_car()

    # -- Follow car ------------------------------------------------------------

    def _spawn_follow_car(self, env: CarlaEnv) -> None:
        """Spawn a follow car behind the ego vehicle.

        The follower is placed on the same lane, _FOLLOW_DISTANCE_M metres
        behind the ego.  It is not put on autopilot -- instead,
        _update_follow_car() applies manual control each tick so it
        dynamically tracks the ego's trajectory.
        """
        self._destroy_follow_car()

        ego_transform = env.vehicle.get_transform()
        ego_loc = ego_transform.location
        ego_yaw = math.radians(ego_transform.rotation.yaw)

        # Position behind the ego along its heading
        spawn_loc = carla.Location(
            x=ego_loc.x - _FOLLOW_DISTANCE_M * math.cos(ego_yaw),
            y=ego_loc.y - _FOLLOW_DISTANCE_M * math.sin(ego_yaw),
            z=ego_loc.z + 0.5,
        )
        spawn_rot = ego_transform.rotation
        spawn_transform = carla.Transform(spawn_loc, spawn_rot)

        # Pick a non-ego vehicle blueprint
        blueprints = [
            bp for bp in env.world.get_blueprint_library().filter("vehicle.*")
            if int(bp.get_attribute("number_of_wheels")) == 4
            and "tesla" not in bp.id.lower()
        ]
        if not blueprints:
            log.debug("No suitable follow-car blueprints found.")
            return

        bp = random.choice(blueprints)
        bp.set_attribute("role_name", "follower")

        actor = env.world.try_spawn_actor(bp, spawn_transform)
        if actor is None:
            # Fall back: try the waypoint-projected location
            try:
                wp = env.world.get_map().get_waypoint(
                    spawn_loc, project_to_road=True
                )
                spawn_transform = wp.transform
                spawn_transform.location.z += 0.5
                actor = env.world.try_spawn_actor(bp, spawn_transform)
            except RuntimeError:
                pass

        if actor is not None:
            self._follow_car = actor
            log.debug("Follow car spawned (id=%d).", actor.id)
        else:
            log.debug("Could not spawn follow car -- continuing without.")

    def _update_follow_car(self, env: CarlaEnv) -> None:
        """Apply one tick of PID-like control to keep the follower behind the ego.

        The controller steers toward the waypoint closest to the ego's rear
        and modulates throttle/brake to maintain the target gap.
        """
        if self._follow_car is None:
            return

        # Check the follower is still alive
        if not self._follow_car.is_alive:
            self._follow_car = None
            return

        ego_loc = env.vehicle.get_location()
        follow_loc = self._follow_car.get_location()
        gap = follow_loc.distance(ego_loc)

        # Respawn if follower fell too far behind (e.g. stuck on geometry)
        if gap > _FOLLOW_MAX_RESPAWN_DIST:
            self._spawn_follow_car(env)
            return

        # Target: the waypoint behind the ego
        ego_transform = env.vehicle.get_transform()
        ego_yaw = math.radians(ego_transform.rotation.yaw)
        target = carla.Location(
            x=ego_loc.x - _FOLLOW_DISTANCE_M * math.cos(ego_yaw),
            y=ego_loc.y - _FOLLOW_DISTANCE_M * math.sin(ego_yaw),
            z=ego_loc.z,
        )

        # -- Steering: angle between follower heading and target --
        follow_transform = self._follow_car.get_transform()
        follow_yaw = math.radians(follow_transform.rotation.yaw)

        dx = target.x - follow_loc.x
        dy = target.y - follow_loc.y
        target_angle = math.atan2(dy, dx)
        angle_error = target_angle - follow_yaw
        # Normalize to [-pi, pi]
        angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi
        steer = float(np.clip(angle_error * _FOLLOW_STEER_GAIN, -1.0, 1.0))

        # -- Throttle/brake: proportional to gap error --
        gap_error = gap - _FOLLOW_DISTANCE_M
        if gap_error > 1.0:
            # Too far -- speed up
            throttle = float(np.clip(gap_error * _FOLLOW_SPEED_GAIN, 0.0, 0.8))
            brake = 0.0
        elif gap_error < -2.0:
            # Too close -- brake
            throttle = 0.0
            brake = float(np.clip(-gap_error * _FOLLOW_SPEED_GAIN, 0.0, 0.6))
        else:
            # In the sweet spot -- match ego speed approximately
            ego_vel = env.vehicle.get_velocity()
            ego_speed = math.sqrt(ego_vel.x**2 + ego_vel.y**2 + ego_vel.z**2)
            follow_vel = self._follow_car.get_velocity()
            follow_speed = math.sqrt(
                follow_vel.x**2 + follow_vel.y**2 + follow_vel.z**2
            )
            speed_diff = ego_speed - follow_speed
            if speed_diff > 0:
                throttle = float(np.clip(speed_diff * 0.3, 0.0, 0.5))
                brake = 0.0
            else:
                throttle = 0.1  # idle cruise
                brake = 0.0

        control = carla.VehicleControl(
            steer=steer, throttle=throttle, brake=brake
        )
        self._follow_car.apply_control(control)

    def _destroy_follow_car(self) -> None:
        """Destroy the follow car actor if it exists."""
        if self._follow_car is not None:
            try:
                self._follow_car.destroy()
            except RuntimeError:
                pass
            self._follow_car = None

    # -- Frame extraction ------------------------------------------------------

    def _read_frame(
        self,
        env: CarlaEnv,
        obs: dict[str, Any],
        weather: str,
        tod: str,
        traffic: str,
    ) -> dict[str, Any]:
        """Extract a single frame of data from the current simulation state."""
        velocity = env.vehicle.get_velocity()
        speed_kmh = 3.6 * math.sqrt(
            velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2
        )
        heading_deg = env.vehicle.get_transform().rotation.yaw
        ctrl = env.vehicle.get_control()
        loc = env.vehicle.get_transform().location

        tl_state = env.vehicle.get_traffic_light_state()
        tl_int = int(tl_state) if tl_state is not None else 3  # 3 = Off

        speed_limit = env.vehicle.get_speed_limit()
        road_type, lane_count, is_junction = self._get_road_features(env)

        # Raw state stores un-normalized values so that DrivingDataset can
        # apply its own normalization. Layout:
        #   [speed_kmh, heading_deg, speed_limit_kmh, lane_count, is_junction]
        return {
            "image": obs["camera"],
            "state": np.array(
                [speed_kmh, heading_deg, float(speed_limit),
                 float(lane_count), float(is_junction)],
                dtype=np.float32,
            ),
            "action": np.array([ctrl.steer, ctrl.throttle, ctrl.brake], dtype=np.float32),
            "location": np.array([loc.x, loc.y], dtype=np.float32),
            "tl_state": np.uint8(tl_int),
            "speed_limit": np.float32(speed_limit),
            "weather_preset": weather,
            "road_type": road_type,
            "time_of_day": tod,
            "traffic_density": traffic,
            "style": "standard",
        }

    def _get_road_features(self, env: CarlaEnv) -> tuple[str, int, int]:
        """Return road type, lane count, and junction flag at the ego's location.

        Returns
        -------
        road_type : str
            One of "highway", "rural", or "urban".
        lane_count : int
            Total number of drivable lanes (capped at 4).
        is_junction : int
            1 if the vehicle is at a junction, 0 otherwise.
        """
        try:
            wp = env.world.get_map().get_waypoint(
                env.vehicle.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            is_junction = int(wp.is_junction)

            lane_count = 1
            wp_iter = wp.get_left_lane()
            while (wp_iter is not None
                   and wp_iter.lane_type == carla.LaneType.Driving
                   and lane_count < 4):
                lane_count += 1
                wp_iter = wp_iter.get_left_lane()
            wp_iter = wp.get_right_lane()
            while (wp_iter is not None
                   and wp_iter.lane_type == carla.LaneType.Driving
                   and lane_count < 4):
                lane_count += 1
                wp_iter = wp_iter.get_right_lane()

            if is_junction:
                road_type = "urban"
            elif wp.lane_width > 4.5:
                road_type = "highway"
            else:
                road_type = "rural"

            return road_type, lane_count, is_junction
        except RuntimeError:
            return "urban", 1, 0

    # -- Weather / NPC tools ---------------------------------------------------

    def _apply_weather(self, env: CarlaEnv, preset: str, tod: str) -> None:
        """Apply a weather preset with time-of-day sun angle override."""
        params: dict[str, Any] = self.cfg["weather_params"][preset]
        tod_angles: dict[str, float] = self.cfg["tod_sun_angles"]
        w = carla.WeatherParameters(
            cloudiness=params["cloudiness"],
            precipitation=params["precipitation"],
            precipitation_deposits=params["precipitation_deposits"],
            wind_intensity=params["wind_intensity"],
            fog_density=params["fog_density"],
            wetness=params["wetness"],
            sun_altitude_angle=tod_angles[tod],
        )
        env.world.set_weather(w)

    def _spawn_npcs(self, env: CarlaEnv, count: int) -> list[carla.Actor]:
        """Spawn *count* NPC vehicles with autopilot enabled."""
        blueprints = [
            bp for bp in env.world.get_blueprint_library().filter("vehicle.*")
            if int(bp.get_attribute("number_of_wheels")) == 4
        ]
        spawn_points = env.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        npcs: list[carla.Actor] = []
        for sp in spawn_points[:count]:
            bp = random.choice(blueprints)
            actor = env.world.try_spawn_actor(bp, sp)
            if actor is not None:
                actor.set_autopilot(True, env.traffic_manager.get_port())
                npcs.append(actor)
        if len(npcs) < count:
            log.debug("Spawned %d/%d NPCs.", len(npcs), count)
        return npcs

    def _destroy_npcs(self, npcs: list[carla.Actor]) -> None:
        """Destroy all NPC actors, ignoring individual failures."""
        for actor in npcs:
            try:
                actor.destroy()
            except RuntimeError:
                pass

    # -- Persistence -----------------------------------------------------------

    def _flush_buffer(self) -> None:
        """Write buffered frames to a compressed .npz chunk file."""
        if self._buffer.size() == 0:
            return
        path = self.data_dir / f"chunk_{self._chunk_index:04d}.npz"
        data = self._buffer.as_arrays()
        np.savez_compressed(path, **data)
        log.info("Saved %s (%d frames).", path.name, self._buffer.size())
        self._chunk_index += 1
        self._buffer.clear()

    def _save_collision_log(self) -> None:
        """Persist the collision event log as a compressed .npz file."""
        if not self._collision_log:
            return
        path = self.data_dir / "collision_log.npz"
        np.savez_compressed(
            path,
            weather=np.array([e["weather"] for e in self._collision_log]),
            tod=np.array([e["tod"] for e in self._collision_log]),
            traffic=np.array([e["traffic"] for e in self._collision_log]),
            frame=np.array([e["frame"] for e in self._collision_log]),
        )
        log.info("Collision log: %d events -> %s", len(self._collision_log), path)


class _FrameBuffer:
    """In-memory accumulator for a single chunk's worth of frames."""

    def __init__(self) -> None:
        self._frames: list[dict[str, Any]] = []

    def append(self, frame: dict[str, Any]) -> None:
        self._frames.append(frame)

    def size(self) -> int:
        return len(self._frames)

    def clear(self) -> None:
        self._frames.clear()

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "images": np.array([f["image"] for f in self._frames], dtype=np.uint8),
            "states": np.array([f["state"] for f in self._frames], dtype=np.float32),
            "actions": np.array([f["action"] for f in self._frames], dtype=np.float32),
            "locations": np.array([f["location"] for f in self._frames], dtype=np.float32),
            "tl_states": np.array([f["tl_state"] for f in self._frames], dtype=np.uint8),
            "speed_limits": np.array([f["speed_limit"] for f in self._frames], dtype=np.float32),
            "weather_preset": np.array([f["weather_preset"] for f in self._frames]),
            "road_type": np.array([f["road_type"] for f in self._frames]),
            "time_of_day": np.array([f["time_of_day"] for f in self._frames]),
            "traffic_density": np.array([f["traffic_density"] for f in self._frames]),
            "style": np.array([f["style"] for f in self._frames]),
        }
