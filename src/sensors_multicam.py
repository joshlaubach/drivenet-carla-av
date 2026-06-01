"""Multi-camera + LiDAR sensor manager for CARLA.

Multi-camera + LiDAR fusion architecture. Implemented for future retraining;
current trained models use single front camera (see src/drivenet.py).

Attaches four RGB cameras and one 64-channel LiDAR to a CARLA ego vehicle,
captures synchronised observations via per-sensor queues, and returns a
dict suitable for FSDNet inference.

LiDAR BEV encoding
-------------------
Raw point cloud (x, y, z, intensity) is projected onto a 256x256 grid
covering 50 m in each horizontal direction (resolution ~0.39 m/cell).
Points are binned into 5 height slices plus an intensity channel:
  ch 0 - occupancy z in [-2.0, 0.0)
  ch 1 - occupancy z in [ 0.0, 0.5)
  ch 2 - occupancy z in [ 0.5, 1.5)
  ch 3 - occupancy z in [ 1.5, 3.0)
  ch 4 - occupancy z in [ 3.0, inf)
  ch 5 - mean normalised intensity per cell (0 if no points)

Output shape: float32 (6, 256, 256), values in [0, 1].
"""
from __future__ import annotations

import queue
import weakref
from typing import Any

import numpy as np

try:
    import carla
except ImportError:  # allow import in CI without CARLA installed
    carla = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IMAGE_W = 400
_IMAGE_H = 300
_FOV = 90

_BEV_GRID = 256          # cells per side
_BEV_RANGE = 50.0        # metres -- grid covers [-50, 50] in x and y
_BEV_RES = 2 * _BEV_RANGE / _BEV_GRID   # metres per cell (~0.39)

_HEIGHT_BINS = [-2.0, 0.0, 0.5, 1.5, 3.0]  # lower edges; last bin is open

_LIDAR_CHANNELS = 64
_LIDAR_RANGE = 50.0
_LIDAR_PPS = 1_000_000   # points per second

# Camera names in observation dict order
_CAM_NAMES = ["front", "front_left", "front_right", "rear"]

# Camera transforms relative to ego vehicle (x forward, y left, z up)
_CAM_TRANSFORMS: dict[str, tuple[float, float, float, float]] = {
    #             x     y     z    yaw( deg)
    "front":       ( 1.5,  0.0,  2.4,   0),
    "front_left":  ( 1.5,  0.0,  2.4, -45),
    "front_right": ( 1.5,  0.0,  2.4,  45),
    "rear":        (-1.5,  0.0,  2.4, 180),
}

_LIDAR_TRANSFORM = (0.0, 0.0, 2.8, 0)  # rooftop mount


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cam_transform(x: float, y: float, z: float, yaw: float) -> "carla.Transform":
    return carla.Transform(
        carla.Location(x=x, y=y, z=z),
        carla.Rotation(yaw=yaw),
    )


def _image_to_rgb(image: "carla.Image") -> np.ndarray:
    """Convert CARLA BGRA image to float32 RGB array in [0, 1]."""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    rgb = arr[:, :, :3][:, :, ::-1].astype(np.float32) / 255.0
    return rgb


def _points_to_bev(raw_data: bytes) -> np.ndarray:
    """Convert raw LiDAR bytes to 6-channel BEV float32 array (6, 256, 256)."""
    pts = np.frombuffer(raw_data, dtype=np.float32).reshape(-1, 4)
    # CARLA LiDAR: x forward, y right (positive), z up
    px, py, pz, intensity = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]

    bev = np.zeros((6, _BEV_GRID, _BEV_GRID), dtype=np.float32)
    intensity_sum = np.zeros((_BEV_GRID, _BEV_GRID), dtype=np.float32)
    intensity_cnt = np.zeros((_BEV_GRID, _BEV_GRID), dtype=np.int32)

    # Grid indices: origin at centre, x->row (forward), y->col (right)
    row = np.floor((_BEV_RANGE - px) / _BEV_RES).astype(np.int32)
    col = np.floor((py + _BEV_RANGE) / _BEV_RES).astype(np.int32)

    valid = (row >= 0) & (row < _BEV_GRID) & (col >= 0) & (col < _BEV_GRID)
    row, col, pz_v, intensity_v = row[valid], col[valid], pz[valid], intensity[valid]

    # Height slice channels
    for ch, z_lo in enumerate(_HEIGHT_BINS):
        z_hi = _HEIGHT_BINS[ch + 1] if ch + 1 < len(_HEIGHT_BINS) else np.inf
        mask = (pz_v >= z_lo) & (pz_v < z_hi)
        np.add.at(bev[ch], (row[mask], col[mask]), 1.0)

    # Intensity channel (ch 5): mean normalised intensity per occupied cell
    np.add.at(intensity_sum, (row, col), intensity_v)
    np.add.at(intensity_cnt, (row, col), 1)
    occupied = intensity_cnt > 0
    bev[5][occupied] = intensity_sum[occupied] / intensity_cnt[occupied]

    # Clip height channels to binary occupancy
    bev[:5] = np.clip(bev[:5], 0.0, 1.0)

    return bev


# ---------------------------------------------------------------------------
# SensorManager
# ---------------------------------------------------------------------------

class SensorManager:
    """Attaches four RGB cameras and one 64-channel LiDAR to a CARLA vehicle.

    Parameters
    ----------
    world:
        Active carla.World instance (synchronous mode expected).
    vehicle:
        Ego vehicle actor to attach sensors to.
    queue_maxsize:
        Maximum backlog per sensor queue.  When full, the oldest frame is
        dropped to prevent memory growth during slow processing.
    """

    def __init__(
        self,
        world: "carla.World",
        vehicle: "carla.Vehicle",
        queue_maxsize: int = 2,
    ) -> None:
        self._world = world
        self._vehicle = vehicle
        self._queue_maxsize = queue_maxsize

        self._cam_queues: dict[str, queue.Queue] = {}
        self._cam_actors: dict[str, "carla.Actor"] = {}
        self._lidar_queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._lidar_actor: "carla.Actor | None" = None

        self._last_obs: dict[str, Any] = {}

        self._attach_sensors()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_obs(self) -> dict[str, Any]:
        """Return the latest synchronised observation dict.

        Keys
        ----
        front, front_left, front_right, rear : np.ndarray (H, W, 3) float32
            RGB images in [0, 1] from the four cameras.
        lidar : np.ndarray (6, 256, 256) float32
            BEV occupancy + intensity channels in [0, 1].
        speed : float
            Ego vehicle speed in m/s (non-negative).
        """
        obs: dict[str, Any] = {}

        for name in _CAM_NAMES:
            q = self._cam_queues[name]
            try:
                image = q.get(timeout=0.1)
                obs[name] = _image_to_rgb(image)
            except queue.Empty:
                obs[name] = self._last_obs.get(name, np.zeros((_IMAGE_H, _IMAGE_W, 3), np.float32))

        try:
            lidar_data = self._lidar_queue.get(timeout=0.1)
            obs["lidar"] = _points_to_bev(lidar_data.raw_data)
        except queue.Empty:
            obs["lidar"] = self._last_obs.get("lidar", np.zeros((6, _BEV_GRID, _BEV_GRID), np.float32))

        v = self._vehicle.get_velocity()
        obs["speed"] = (v.x ** 2 + v.y ** 2 + v.z ** 2) ** 0.5

        self._last_obs = obs
        return obs

    def destroy(self) -> None:
        """Stop and destroy all sensor actors."""
        for actor in self._cam_actors.values():
            try:
                actor.stop()
                actor.destroy()
            except Exception:
                pass
        self._cam_actors.clear()

        if self._lidar_actor is not None:
            try:
                self._lidar_actor.stop()
                self._lidar_actor.destroy()
            except Exception:
                pass
            self._lidar_actor = None

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _attach_sensors(self) -> None:
        bp_lib = self._world.get_blueprint_library()

        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(_IMAGE_W))
        cam_bp.set_attribute("image_size_y", str(_IMAGE_H))
        cam_bp.set_attribute("fov", str(_FOV))

        for name in _CAM_NAMES:
            x, y, z, yaw = _CAM_TRANSFORMS[name]
            transform = _make_cam_transform(x, y, z, yaw)
            actor = self._world.spawn_actor(cam_bp, transform, attach_to=self._vehicle)
            q: queue.Queue = queue.Queue(maxsize=self._queue_maxsize)
            self._cam_queues[name] = q
            self._cam_actors[name] = actor

            self_ref = weakref.ref(self)

            def _callback(image: "carla.Image", _q: queue.Queue = q) -> None:
                sm = self_ref()
                if sm is None:
                    return
                try:
                    if _q.full():
                        _q.get_nowait()
                    _q.put_nowait(image)
                except Exception:
                    pass

            actor.listen(_callback)

        lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("channels", str(_LIDAR_CHANNELS))
        lidar_bp.set_attribute("range", str(_LIDAR_RANGE))
        lidar_bp.set_attribute("points_per_second", str(_LIDAR_PPS))
        lidar_bp.set_attribute("rotation_frequency", "20")

        lx, ly, lz, lyaw = _LIDAR_TRANSFORM
        lidar_transform = _make_cam_transform(lx, ly, lz, lyaw)
        self._lidar_actor = self._world.spawn_actor(
            lidar_bp, lidar_transform, attach_to=self._vehicle
        )
        lidar_q = self._lidar_queue
        self_ref = weakref.ref(self)

        def _lidar_callback(data: "carla.LidarMeasurement") -> None:
            sm = self_ref()
            if sm is None:
                return
            try:
                if lidar_q.full():
                    lidar_q.get_nowait()
                lidar_q.put_nowait(data)
            except Exception:
                pass

        self._lidar_actor.listen(_lidar_callback)
