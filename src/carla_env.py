import math
import time
import weakref
from queue import Empty, Queue

import carla
import gymnasium
import numpy as np
from gymnasium import spaces

from src.drivenet_lidar import points_to_bev, RESIZE_H, RESIZE_W

# Camera transforms for the multi-camera suite.
# Front camera looks straight ahead. Left and right cameras are mounted
# on the A-pillars and angled 60 degrees outward, covering the blind
# spots that a front-only camera misses at intersections.
_CAM_TRANSFORMS = [
    carla.Transform(carla.Location(x=1.5, z=2.4)),                               # front
    carla.Transform(carla.Location(x=1.0, y=-1.0, z=2.4), carla.Rotation(yaw=-60)),  # left
    carla.Transform(carla.Location(x=1.0, y=1.0,  z=2.4), carla.Rotation(yaw=60)),   # right
]

# Lidar sensor parameters. 64 channels at 50 m range gives dense coverage
# of the area around the vehicle without the memory overhead of higher ranges.
_LIDAR_CHANNELS = 64
_LIDAR_RANGE_M = 50.0
_LIDAR_PPS = 1_000_000   # points per second
_LIDAR_TRANSFORM = carla.Transform(carla.Location(x=0.0, z=2.8))  # rooftop center


def _short_map_name(name: str) -> str:
    """Normalize a CARLA map path to a short town ID.

    Example: '/Game/Carla/Maps/Town03_Opt' -> 'Town03'.
    """
    short = name.split("/")[-1]
    if short.endswith("_Opt"):
        short = short[: -len("_Opt")]
    return short


def _wait_for_map(client: carla.Client, expected: str, timeout: float = 60.0) -> carla.World:
    """Poll until the expected town is active, then return the world object.

    Used after client.load_world() to make sure the new map has finished
    streaming before any spawn_actor() calls.
    """
    deadline = time.time() + timeout
    last_loaded = "<unknown>"
    while time.time() < deadline:
        try:
            world = client.get_world()
            loaded = _short_map_name(world.get_map().name)
            last_loaded = loaded
            if loaded == expected:
                return world
        except Exception:
            pass
        time.sleep(1.0)
    raise RuntimeError(
        f"Map '{expected}' not ready within {timeout:.0f}s "
        f"(actual map after timeout: '{last_loaded}')"
    )


class CarlaEnv(gymnasium.Env):
    """Gymnasium environment wrapper for CARLA.

    Supports three sensor suites that correspond to the causal experiment:

    - "single_cam" (default): one front-facing RGB camera. Used as the
      control condition in the sensor comparison.
    - "multi_cam": front, left, and right RGB cameras with a shared CNN
      backbone (late fusion). Mimics Tesla's HydraNet approach.
    - "lidar": a 64-channel rooftop lidar whose point cloud is projected
      to a Bird's Eye View image before being fed to the model. Mimics
      Waymo's sensor approach.

    The observation dict keys differ by suite:
    - "single_cam": {"camera": (H, W, 3) uint8, "state": (6,) float32}
    - "multi_cam":  {"cameras": (3, H, W, 3) uint8, "state": (6,) float32}
    - "lidar":      {"bev": (H, W, 3) float32, "state": (6,) float32}

    The state vector is always 6 values:
    [speed_norm, sin_heading, cos_heading, speed_limit_norm, lane_count_norm,
    is_junction].
    """

    metadata = {"render_modes": ["rgb_array"]}

    VALID_SUITES = ("single_cam", "multi_cam", "lidar")

    def __init__(
        self,
        host="localhost",
        port=2000,
        town="Town03",
        fps=20,
        image_width=400,
        image_height=300,
        sensor_suite="single_cam",
    ):
        super().__init__()
        if sensor_suite not in self.VALID_SUITES:
            raise ValueError(
                f"sensor_suite must be one of {self.VALID_SUITES}, "
                f"got '{sensor_suite}'."
            )
        self.host = host
        self.port = port
        self.town = town
        self.fps = fps
        self.image_width = image_width
        self.image_height = image_height
        self.sensor_suite = sensor_suite

        state_space = spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32)
        if sensor_suite == "single_cam":
            self.observation_space = spaces.Dict({
                "camera": spaces.Box(0, 255, (image_height, image_width, 3), dtype=np.uint8),
                "state": state_space,
            })
        elif sensor_suite == "multi_cam":
            self.observation_space = spaces.Dict({
                "cameras": spaces.Box(0, 255, (3, image_height, image_width, 3), dtype=np.uint8),
                "state": state_space,
            })
        else:  # lidar
            self.observation_space = spaces.Dict({
                "bev": spaces.Box(0.0, 1.0, (RESIZE_H, RESIZE_W, 3), dtype=np.float32),
                "state": state_space,
            })

        self.action_space = spaces.Box(
            np.array([-1.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        try:
            self.client.get_server_version()
        except RuntimeError as exc:
            raise RuntimeError(
                f"Cannot reach CARLA at {host}:{port}. "
                "Launch CarlaUE4-Win64-Shipping.exe and wait for the "
                "'ready' message before running this notebook."
            ) from exc

        # client.load_world() works at runtime on RTX 5000-series (Blackwell)
        # GPUs when CARLA is launched with -dx12 (the project default in
        # scripts/launch_carla.bat). -dx11 alone deadlocks the camera
        # rendering pipeline after ~5 frames; -dx12 avoids that deadlock and
        # also permits in-place map switches. See scenarios 10 and 20 in
        # tests/test_crash_scenarios.py for the empirical evidence.
        _init_world = self.client.get_world()
        try:
            _init_world.wait_for_tick(seconds=15.0)
        except Exception:
            pass
        _map_short = _short_map_name(_init_world.get_map().name)
        _town_short = _short_map_name(town)
        if _town_short == _map_short:
            self.world = _init_world
        else:
            self.client.load_world(_town_short)
            time.sleep(3.0)
            self.world = _wait_for_map(self.client, _town_short, timeout=60.0)
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / fps
        self.world.apply_settings(settings)

        self.traffic_manager = self.client.get_trafficmanager()
        self.traffic_manager.set_synchronous_mode(True)

        self.vehicle = None
        self._cam_sensors: list[carla.Actor] = []
        self._cam_queues: list[Queue] = []
        self._lidar_sensor: carla.Actor | None = None
        self._lidar_queue: Queue | None = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self._collision_flag = False
        self._collision_intensity = 0.0
        self._lane_invaded = False
        self._last_image: np.ndarray | None = None
        self._speed_kmh = 0.0

    # -- Actor setup -----------------------------------------------------------

    def _setup_vehicle(self) -> None:
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
        vehicle_bp.set_attribute("role_name", "hero")

        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available on map " + self.town)

        self.vehicle = None
        for sp in spawn_points:
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, sp)
            if self.vehicle is not None:
                break
        if self.vehicle is None:
            raise RuntimeError(
                "Failed to spawn ego vehicle -- all spawn points occupied."
            )

    def _setup_sensors(self) -> None:
        bpl = self.world.get_blueprint_library()
        weak_self = weakref.ref(self)

        if self.sensor_suite in ("single_cam", "multi_cam"):
            n_cams = 1 if self.sensor_suite == "single_cam" else 3
            cam_bp = bpl.find("sensor.camera.rgb")
            cam_bp.set_attribute("image_size_x", str(self.image_width))
            cam_bp.set_attribute("image_size_y", str(self.image_height))
            for i in range(n_cams):
                cam = self.world.spawn_actor(
                    cam_bp, _CAM_TRANSFORMS[i], attach_to=self.vehicle,
                )
                q: Queue = Queue(maxsize=5)
                # Each closure captures its own queue index via default arg
                cam.listen(
                    lambda img, qi=i: CarlaEnv._on_camera_data(weak_self, img, qi)
                )
                self._cam_sensors.append(cam)
                self._cam_queues.append(q)

        else:  # lidar
            lidar_bp = bpl.find("sensor.lidar.ray_cast")
            lidar_bp.set_attribute("channels", str(_LIDAR_CHANNELS))
            lidar_bp.set_attribute("range", str(_LIDAR_RANGE_M))
            lidar_bp.set_attribute("points_per_second", str(_LIDAR_PPS))
            lidar_bp.set_attribute("rotation_frequency", str(self.fps))
            self._lidar_sensor = self.world.spawn_actor(
                lidar_bp, _LIDAR_TRANSFORM, attach_to=self.vehicle,
            )
            self._lidar_queue = Queue(maxsize=5)
            self._lidar_sensor.listen(
                lambda data: CarlaEnv._on_lidar_data(weak_self, data)
            )

        col_bp = bpl.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(
            col_bp, carla.Transform(), attach_to=self.vehicle,
        )
        self.collision_sensor.listen(
            lambda event: CarlaEnv._on_collision(weak_self, event)
        )

        lane_bp = bpl.find("sensor.other.lane_invasion")
        self.lane_invasion_sensor = self.world.spawn_actor(
            lane_bp, carla.Transform(), attach_to=self.vehicle,
        )
        self.lane_invasion_sensor.listen(
            lambda event: CarlaEnv._on_lane_invasion(weak_self, event)
        )

    # -- Sensor callbacks ------------------------------------------------------

    # Every sensor.listen() callback below is wrapped in a top-level
    # try/except. CARLA dispatches these from a C++ Boost.Asio thread; any
    # Python exception that propagates out can corrupt the io_context and
    # silently kill the host process (STATUS_STACK_BUFFER_OVERRUN 0xC0000409
    # on this hardware). Dropping a frame is always preferable to that.
    # See .claude/agent-memory/.../project_sensor_callback_guards.md.

    @staticmethod
    def _on_camera_data(weak_self, image, cam_index: int) -> None:
        try:
            self = weak_self()
            if self is None:
                return
            # Re-read length inside the try -- _destroy_actors() may have
            # cleared _cam_queues between the bound check and .put().
            queues = self._cam_queues
            if cam_index < len(queues):
                queues[cam_index].put_nowait(image)
        except Exception:
            pass  # drop frame rather than destabilize the C++ thread

    @staticmethod
    def _on_lidar_data(weak_self, data) -> None:
        try:
            self = weak_self()
            if self is None:
                return
            q = self._lidar_queue
            if q is not None:
                q.put_nowait(data)
        except Exception:
            pass  # drop scan rather than destabilize the C++ thread

    @staticmethod
    def _on_collision(weak_self, event) -> None:
        try:
            self = weak_self()
            if self is None:
                return
            impulse = event.normal_impulse
            intensity = math.sqrt(
                impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2
            )
            self._collision_flag = True
            self._collision_intensity = intensity
        except Exception:
            # Still record the collision so the agent resets the episode,
            # even if we couldn't read the impulse vector.
            try:
                s = weak_self()
                if s is not None:
                    s._collision_flag = True
            except Exception:
                pass

    @staticmethod
    def _on_lane_invasion(weak_self, event) -> None:
        try:
            self = weak_self()
            if self is not None:
                self._lane_invaded = True
        except Exception:
            pass

    # -- Sensor data retrieval -------------------------------------------------

    def _retrieve_camera_frame(self, queue: Queue) -> object:
        """Block until a frame at or after the current tick arrives."""
        target_frame = self.world.get_snapshot().frame
        while True:
            try:
                data = queue.get(timeout=2.0)
            except Empty:
                raise RuntimeError(
                    f"Camera sensor timed out waiting for frame {target_frame}."
                )
            if data.frame >= target_frame:
                return data

    def _parse_image(self, image_data) -> np.ndarray:
        array = np.frombuffer(image_data.raw_data, dtype=np.uint8)
        array = array.reshape((self.image_height, self.image_width, 4))
        return array[:, :, :3][:, :, ::-1].copy()

    def _retrieve_lidar_points(self) -> np.ndarray:
        """Block until a lidar scan at or after the current tick arrives."""
        target_frame = self.world.get_snapshot().frame
        while True:
            try:
                data = self._lidar_queue.get(timeout=2.0)
            except Empty:
                raise RuntimeError(
                    f"Lidar sensor timed out waiting for frame {target_frame}."
                )
            if data.frame >= target_frame:
                pts = np.frombuffer(data.raw_data, dtype=np.float32)
                pts = pts.reshape(-1, 4)
                # CARLA lidar uses left-hand Z-up coordinates in sensor frame.
                # Negate y so that positive y points left (matching the ego frame
                # convention used by points_to_bev).
                pts_ego = pts.copy()
                pts_ego[:, 1] = -pts_ego[:, 1]
                return pts_ego

    # -- Road topology ---------------------------------------------------------

    def _get_road_topology(self) -> tuple[float, float, float]:
        """Return normalized road topology features at the ego's location.

        Returns
        -------
        speed_limit_norm : float
            Speed limit divided by 130.0, clamped to [0, 1].
        lane_count_norm : float
            Number of drivable lanes (capped at 4) divided by 4.0.
        is_junction : float
            1.0 if the vehicle is at a junction, 0.0 otherwise.

        Falls back to (0.5, 0.5, 0.0) if the waypoint query fails.
        """
        try:
            wp = self.world.get_map().get_waypoint(
                self.vehicle.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            speed_limit_norm = float(
                np.clip(self.vehicle.get_speed_limit() / 130.0, 0.0, 1.0)
            )
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
            is_junction = 1.0 if wp.is_junction else 0.0
            return speed_limit_norm, lane_count / 4.0, is_junction
        except RuntimeError:
            return 0.5, 0.5, 0.0

    # -- Observation -----------------------------------------------------------

    def _build_state(self) -> np.ndarray:
        velocity = self.vehicle.get_velocity()
        speed_kmh = 3.6 * math.sqrt(
            velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2
        )
        self._speed_kmh = speed_kmh
        heading_rad = math.radians(self.vehicle.get_transform().rotation.yaw)
        sl_norm, lc_norm, is_junc = self._get_road_topology()
        return np.array(
            [speed_kmh / 60.0, math.sin(heading_rad), math.cos(heading_rad),
             sl_norm, lc_norm, is_junc],
            dtype=np.float32,
        )

    def _get_observation(self) -> dict:
        state = self._build_state()

        if self.sensor_suite == "single_cam":
            img_data = self._retrieve_camera_frame(self._cam_queues[0])
            img = self._parse_image(img_data)
            self._last_image = img
            return {"camera": img, "state": state}

        elif self.sensor_suite == "multi_cam":
            imgs = []
            for q in self._cam_queues:
                img_data = self._retrieve_camera_frame(q)
                imgs.append(self._parse_image(img_data))
            self._last_image = imgs[0]
            return {"cameras": np.stack(imgs, axis=0), "state": state}

        else:  # lidar
            pts = self._retrieve_lidar_points()
            bev = points_to_bev(pts).transpose(1, 2, 0)  # (H, W, 3) HWC
            self._last_image = (bev * 255).astype(np.uint8)
            return {"bev": bev, "state": state}

    # -- Actor teardown --------------------------------------------------------

    def _destroy_actors(self) -> None:
        # Stop all sensor callbacks before destroying actors. This prevents
        # Boost.Asio Fast-Fail crashes when a subsequent load_world() or
        # process kill arrives while a callback is still running.
        all_sensors = (
            self._cam_sensors
            + ([self._lidar_sensor] if self._lidar_sensor else [])
            + [self.collision_sensor, self.lane_invasion_sensor]
        )
        for sensor in all_sensors:
            if sensor is not None:
                try:
                    sensor.stop()
                except RuntimeError:
                    pass
        for actor in all_sensors + [self.vehicle]:
            if actor is not None:
                try:
                    actor.destroy()
                except RuntimeError:
                    pass
        self._cam_sensors = []
        self._cam_queues = []
        self._lidar_sensor = None
        self._lidar_queue = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.vehicle = None

    # -- Gymnasium interface ---------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._destroy_actors()

        # Drain stale sensor data
        for q in self._cam_queues:
            while not q.empty():
                try:
                    q.get_nowait()
                except Empty:
                    break
        if self._lidar_queue is not None:
            while not self._lidar_queue.empty():
                try:
                    self._lidar_queue.get_nowait()
                except Empty:
                    break

        self._collision_flag = False
        self._collision_intensity = 0.0
        self._lane_invaded = False

        self._setup_vehicle()
        self._setup_sensors()

        # Warm-up ticks to let sensors initialize and the vehicle settle
        for _ in range(10):
            self.world.tick()
            for q in self._cam_queues:
                try:
                    q.get(timeout=1.0)
                except Empty:
                    pass
            if self._lidar_queue is not None:
                try:
                    self._lidar_queue.get(timeout=1.0)
                except Empty:
                    pass

        self.world.tick()
        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        """Execute one simulation step.

        This environment does not enforce a step limit. It only terminates
        on collision (terminated=True). Callers must track episode length
        externally and truncate when needed.
        """
        steer = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1], 0.0, 1.0))
        brake = float(np.clip(action[2], 0.0, 1.0))

        self.vehicle.apply_control(
            carla.VehicleControl(steer=steer, throttle=throttle, brake=brake)
        )
        self.world.tick()

        obs = self._get_observation()
        speed_kmh = self._speed_kmh

        reward = 1.0
        terminated = False

        if self._collision_flag:
            reward -= 200.0
            terminated = True
            self._collision_flag = False
            self._collision_intensity = 0.0

        lane_invaded = self._lane_invaded
        if self._lane_invaded:
            reward -= 10.0
            self._lane_invaded = False

        info = {
            "speed_kmh": speed_kmh,
            "collision": terminated,
            "lane_invaded": lane_invaded,
        }
        return obs, reward, terminated, False, info

    def render(self):
        if self._last_image is not None:
            return self._last_image
        return np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)

    def close(self):
        # Empirical teardown order for RTX 5080 + CARLA 0.9.16:
        #   1. set_autopilot(False) so the TrafficManager stops driving the ego
        #   2. sensor.stop() so listen() callbacks drain
        #   3. brief sleep for io_context to flush
        #   4. destroy actors
        #   5. release client / world / TM handles
        # We deliberately do NOT call world.apply_settings(sync=False) here.
        # That call invokes abort() inside libcarla on this hardware after a
        # full collection cycle, killing the Python process. Since the agent
        # always kills the CARLA server after close() before launching a new
        # one for the next town, the sync-mode state is moot.
        if self.vehicle is not None:
            try:
                self.vehicle.set_autopilot(False)
            except RuntimeError:
                pass
        all_sensors = (
            self._cam_sensors
            + ([self._lidar_sensor] if self._lidar_sensor else [])
            + [self.collision_sensor, self.lane_invasion_sensor]
        )
        for sensor in all_sensors:
            if sensor is not None:
                try:
                    sensor.stop()
                except RuntimeError:
                    pass
        time.sleep(0.1)
        self._destroy_actors()
        self.traffic_manager = None
        self.world = None
        self.client = None
        time.sleep(0.5)
