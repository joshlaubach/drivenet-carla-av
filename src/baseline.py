"""PID-based rule-following baseline policy for evaluation benchmarking.

Acts as a floor in sensor suite comparisons -- every metric should beat this.
No neural network. Uses a pure proportional controller for steering and speed.

Waypoint following requires the vehicle's current position. Callers pass
(x, y) via obs["_vehicle_loc"] before each act() call; the eval agent
injects this key when running baseline specs.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class PIDBaselinePolicy:
    """P-controller that follows a GlobalRoutePlanner waypoint list.

    Parameters
    ----------
    target_speed_kmh : float
        Cruise speed target.
    lookahead_m : float
        Waypoint advance threshold in metres (advance when closer than this).
    kp_steer : float
        Proportional gain on the cross-product steering error.
    """

    def __init__(
        self,
        target_speed_kmh: float = 30.0,
        lookahead_m: float = 5.0,
        kp_steer: float = 0.5,
    ) -> None:
        self._target_kmh = target_speed_kmh
        self._lookahead = lookahead_m
        self._kp_steer = kp_steer
        self._wps: list[tuple[float, float]] = []
        self._wp_idx: int = 0

    def set_route(self, waypoints: list[Any]) -> None:
        """Store a route as (x, y) pairs extracted from carla.Location objects."""
        self._wps = [(float(wp.x), float(wp.y)) for wp in waypoints]
        self._wp_idx = 0

    def reset(self) -> None:
        """Clear route state between episodes."""
        self._wps = []
        self._wp_idx = 0

    def act(self, obs: dict[str, Any]) -> np.ndarray:
        """Return [steer, throttle, brake] given the current observation.

        obs["state"] : 6-dim [speed/60, sin_h, cos_h, speed_lim/130, lanes/4, junction]
        obs["_vehicle_loc"] : (x, y) tuple injected by EvaluationAgent
        """
        state = obs["state"]
        speed_kmh = float(state[0]) * 60.0
        sin_h = float(state[1])
        cos_h = float(state[2])

        # Speed control: proportional gain on error
        err = self._target_kmh - speed_kmh
        throttle = float(np.clip(err * 0.05, 0.0, 1.0))
        brake = float(np.clip(-err * 0.05, 0.0, 1.0)) if err < -5.0 else 0.0
        if err < -5.0:
            throttle = 0.0

        # Steering: pure pursuit toward next waypoint
        steer = self._compute_steer(obs.get("_vehicle_loc"), sin_h, cos_h)

        return np.array([steer, throttle, brake], dtype=np.float32)

    def _compute_steer(
        self,
        vehicle_loc: tuple[float, float] | None,
        sin_h: float,
        cos_h: float,
    ) -> float:
        if vehicle_loc is None or not self._wps:
            return 0.0

        vx, vy = vehicle_loc

        # Advance waypoint index when within lookahead distance
        while (
            self._wp_idx < len(self._wps) - 1
            and math.hypot(
                vx - self._wps[self._wp_idx][0],
                vy - self._wps[self._wp_idx][1],
            )
            < self._lookahead
        ):
            self._wp_idx += 1

        tx, ty = self._wps[self._wp_idx]
        dx, dy = tx - vx, ty - vy
        dist = math.hypot(dx, dy) + 1e-6

        # Target heading components (CARLA: +X forward, +Y right)
        target_cos = dx / dist  # forward component
        target_sin = dy / dist  # rightward component

        # Signed cross product: positive = target is to the right of heading
        cross = cos_h * target_sin - sin_h * target_cos
        return float(np.clip(cross * self._kp_steer, -1.0, 1.0))
