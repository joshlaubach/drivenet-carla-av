import math
import weakref
from queue import Queue, Empty

import carla
import gymnasium
import numpy as np
from gymnasium import spaces


class CarlaEnv(gymnasium.Env):

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        host="localhost",
        port=2000,
        town="Town03",
        fps=20,
        image_width=400,
        image_height=300,
    ):
        super().__init__()
        self.host = host
        self.port = port
        self.town = town
        self.fps = fps
        self.image_width = image_width
        self.image_height = image_height

        self.observation_space = spaces.Dict(
            {
                "camera": spaces.Box(
                    0, 255, (image_height, image_width, 3), dtype=np.uint8
                ),
                "state": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
            }
        )
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
        # client.load_world() triggers a Vulkan null-pointer crash (address 0x98
        # in TaskGraphThreadHP) on RTX 5000-series (Blackwell) GPUs during the
        # world transition.  Skip the load if the requested map is already active
        # so the testing notebook can run without switching maps.
        # For data collection, launch CARLA with the target map on the command
        # line (e.g. CarlaUE4.exe Town03) instead of switching at runtime.
        _init_world = self.client.get_world()
        try:
            _init_world.wait_for_tick(seconds=15.0)
        except Exception:
            pass
        _current_map = _init_world.get_map().name
        # Normalise to short name (e.g. "/Game/Carla/Maps/Town03" -> "Town03")
        _map_short = _current_map.split("/")[-1]
        _town_short = town.split("/")[-1]
        if _town_short == _map_short or _town_short in _map_short or _map_short.endswith(_town_short):
            self.world = _init_world
        else:
            # client.load_world() triggers a Vulkan null-pointer crash on
            # RTX 5000-series (Blackwell) GPUs with UE4 4.26.  Never call it.
            # Restart CARLA with the target map on the command line instead:
            #   CarlaUE4.exe /Game/Carla/Maps/<town>
            raise RuntimeError(
                f"Cannot switch maps at runtime on this hardware.\n"
                f"  Requested: {town!r}  (short: {_town_short!r})\n"
                f"  Active:    {_current_map!r}  (short: {_map_short!r})\n"
                f"Restart CARLA with the target map on the command line:\n"
                f"  CarlaUE4.exe /Game/Carla/Maps/{_town_short}"
            )
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

        self._original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / fps
        self.world.apply_settings(settings)

        self.traffic_manager = self.client.get_trafficmanager()
        self.traffic_manager.set_synchronous_mode(True)

        self.vehicle = None
        self.camera_sensor = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self._camera_queue = Queue(maxsize=5)
        self._collision_flag = False
        self._collision_intensity = 0.0
        self._lane_invaded = False
        self._last_image = None
        self._speed_kmh = 0.0

    def _setup_vehicle(self):
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
                "Failed to spawn ego vehicle -- all spawn points occupied"
            )

    def _setup_sensors(self):
        blueprint_library = self.world.get_blueprint_library()

        cam_bp = blueprint_library.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self.image_width))
        cam_bp.set_attribute("image_size_y", str(self.image_height))
        cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera_sensor = self.world.spawn_actor(
            cam_bp, cam_transform, attach_to=self.vehicle
        )
        weak_self = weakref.ref(self)
        self.camera_sensor.listen(
            lambda image: CarlaEnv._on_camera_data(weak_self, image)
        )

        col_bp = blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(
            col_bp, carla.Transform(), attach_to=self.vehicle
        )
        self.collision_sensor.listen(
            lambda event: CarlaEnv._on_collision(weak_self, event)
        )

        lane_bp = blueprint_library.find("sensor.other.lane_invasion")
        self.lane_invasion_sensor = self.world.spawn_actor(
            lane_bp, carla.Transform(), attach_to=self.vehicle
        )
        self.lane_invasion_sensor.listen(
            lambda event: CarlaEnv._on_lane_invasion(weak_self, event)
        )

    @staticmethod
    def _on_camera_data(weak_self, image):
        self = weak_self()
        if self is not None:
            self._camera_queue.put(image)

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if self is not None:
            impulse = event.normal_impulse
            intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
            self._collision_flag = True
            self._collision_intensity = intensity

    @staticmethod
    def _on_lane_invasion(weak_self, event):
        self = weak_self()
        if self is not None:
            self._lane_invaded = True

    def _retrieve_camera_data(self):
        target_frame = self.world.get_snapshot().frame
        while True:
            try:
                data = self._camera_queue.get(timeout=2.0)
            except Empty:
                raise RuntimeError(
                    f"Camera sensor timed out waiting for frame {target_frame}"
                )
            if data.frame >= target_frame:
                return data

    def _parse_image(self, image_data):
        array = np.frombuffer(image_data.raw_data, dtype=np.uint8)
        array = array.reshape((self.image_height, self.image_width, 4))
        array = array[:, :, :3][:, :, ::-1].copy()
        return array

    def _get_observation(self):
        image_data = self._retrieve_camera_data()
        image_array = self._parse_image(image_data)
        self._last_image = image_array

        velocity = self.vehicle.get_velocity()
        speed_kmh = 3.6 * math.sqrt(
            velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2
        )
        self._speed_kmh = speed_kmh
        heading_rad = math.radians(self.vehicle.get_transform().rotation.yaw)

        return {
            "camera": image_array,
            "state": np.array(
                [speed_kmh / 60.0, math.sin(heading_rad), math.cos(heading_rad)],
                dtype=np.float32,
            ),
        }

    def _destroy_actors(self):
        for actor in [
            self.camera_sensor,
            self.collision_sensor,
            self.lane_invasion_sensor,
            self.vehicle,
        ]:
            if actor is not None:
                actor.destroy()
        self.camera_sensor = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.vehicle = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._destroy_actors()

        # Drain any stale camera data
        while not self._camera_queue.empty():
            try:
                self._camera_queue.get_nowait()
            except Empty:
                break

        self._collision_flag = False
        self._collision_intensity = 0.0
        self._lane_invaded = False

        self._setup_vehicle()
        self._setup_sensors()

        # Warm-up ticks to let sensors initialize and vehicle settle
        for _ in range(10):
            self.world.tick()
            try:
                self._camera_queue.get(timeout=1.0)
            except Empty:
                pass

        self.world.tick()
        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        """Execute one simulation step.

        Note: this environment does NOT enforce a step limit. It only
        terminates on collision (terminated=True). Callers must implement
        their own episode length limit by counting steps externally.
        The truncated return value is always False.
        """
        steer = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1], 0.0, 1.0))
        brake = float(np.clip(action[2], 0.0, 1.0))

        control = carla.VehicleControl(
            steer=steer, throttle=throttle, brake=brake
        )
        self.vehicle.apply_control(control)
        self.world.tick()

        obs = self._get_observation()
        speed_kmh = self._speed_kmh

        reward = speed_kmh / 40.0
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

        info = {"speed_kmh": speed_kmh, "collision": terminated, "lane_invaded": lane_invaded}
        return obs, reward, terminated, False, info

    def render(self):
        if self._last_image is not None:
            return self._last_image
        return np.zeros(
            (self.image_height, self.image_width, 3), dtype=np.uint8
        )

    def close(self):
        self._destroy_actors()
        self.world.apply_settings(self._original_settings)
