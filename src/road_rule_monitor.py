"""CA driving rule enforcement wrapper for CARLA PPO training.

Wraps CarlaEnv as a Gymnasium Wrapper and injects three tiers of
California DMV rule-based rewards into the step() output:

Tier 1 -- Episode-terminating violations (-200 each, CVC references):
  - Collision          : pass-through from CarlaEnv
  - Red light running  : CVC 21453
  - Wrong-way driving  : CVC 21650
  - Off-road           : CVC 21663
  - Double-solid line  : CVC 21460 (crossing into opposing traffic)

Tier 2 -- Per-step or one-time penalties, no termination:
  - Speeding           : linear -(excess_frac * 3.0)/step  CVC 22349/22350
  - Tailgating         : linear inside CA 3-second envelope CVC 21703
  - Stop sign          : -5 one-time on zone exit without stop  CVC 22450
  - Solid lane cross   : -7 additive on CarlaEnv -3 baseline   CVC 21658
  - Failure to yield   : -1.5 one-time on uncontrolled junction entry CVC 21800

All violations are logged to info["road_rule_monitor"] every step.
The Tier 3 comfort/style penalties (jerk, abrupt steering) live in
compute_style_reward() in ppo.py -- they are style-weight-dependent
and belong at the PPOAgent layer, not here.
"""

from __future__ import annotations

import math

import carla
import gymnasium

# ---------------------------------------------------------------------------
# Tier 1 constants
# ---------------------------------------------------------------------------

_TIER1_PENALTY = -200.0

# Wrong-way: dot product between vehicle forward and road forward below this
# threshold (outside junctions only) triggers the counter.
_WRONG_WAY_DOT_THRESH = -0.7
_WRONG_WAY_GRACE_FRAMES = 30   # 1.5 s at 20 FPS

# Red light: must be within _RED_LIGHT_DIST_M of the stop line, travelling
# above _RED_LIGHT_SPEED_KMH, AND not decelerating at >= _RED_LIGHT_DECEL_THRESH
# for _RED_LIGHT_CONFIRM_FRAMES consecutive steps.
_RED_LIGHT_DIST_M = 12.0
_RED_LIGHT_SPEED_KMH = 5.0
_RED_LIGHT_DECEL_THRESH = -0.3     # km/h per step (lenient -- CVC 21453)
_RED_LIGHT_CONFIRM_FRAMES = 3      # 0.15 s confirmation window

# Off-road: grace period before terminating (avoids false positives at
# road edges during sharp turns).
_OFF_ROAD_GRACE_FRAMES = 10

# ---------------------------------------------------------------------------
# Tier 2 constants
# ---------------------------------------------------------------------------

# Speeding: penalty = -(excess_fraction * scale) per step (CVC 22349/22350)
_SPEED_PENALTY_SCALE = 3.0

# Tailgating: CA 3-second rule (CVC 21703). Actor scan within _TAILGATE_MAX_M,
# forward cone half-angle _TAILGATE_CONE_DEG.
_SAFE_FOLLOW_S = 3.0
_TAILGATE_MAX_M = 40.0
_TAILGATE_CONE_DEG = 30.0
_TAILGATE_MIN_M = 5.0   # minimum safe distance regardless of speed

# Stop sign: zone radius, stop-speed threshold, cooldown after zone exit.
_STOP_ZONE_M = 15.0
_STOP_SPEED_KMH = 3.0
_STOP_COOLDOWN_STEPS = 160   # 8 s * 20 FPS

# Lane markings: additive Tier 2 penalty on top of CarlaEnv's -3 baseline.
_SOLID_LANE_ADDITIVE = -7.0

# Failure to yield: must be below this speed when entering an uncontrolled
# junction (no affecting traffic light, no stop sign within 15 m).
_YIELD_ENTRY_SPEED_KMH = 10.0
_YIELD_PENALTY = -1.5


class RoadRuleMonitor(gymnasium.Wrapper):
    """Gymnasium wrapper that enforces CA driving rules via reward shaping.

    Instantiate by wrapping a CarlaEnv:
        env = RoadRuleMonitor(CarlaEnv(...))

    The wrapper is transparent to the observation and action spaces -- it
    only modifies the reward and terminated flag returned by step(), and
    appends info["road_rule_monitor"] with a per-violation breakdown.
    """

    def __init__(self, env: gymnasium.Env) -> None:
        super().__init__(env)
        # Per-episode counters and state machines reset in reset().
        self._wrong_way_counter = 0

    # -- Attribute forwarding -------------------------------------------------
    # gymnasium.Wrapper in v1.x does not define __getattr__, so we must
    # explicitly expose the CarlaEnv attributes that agents and utilities
    # (e.g. make_weather, GlobalRoutePlanner) access directly on the env object.

    @property
    def vehicle(self):
        return self.env.vehicle

    @property
    def world(self):
        return self.env.world

    @property
    def map(self):
        return self.env.map
        self._red_light_counter = 0
        self._off_road_counter = 0
        self._prev_speed_kmh = 0.0
        self._prev_in_junction = False
        # Keyed by stop-sign actor ID. Value: {in_zone, has_stopped, cooldown}.
        self._stop_sign_states: dict[int, dict] = {}
        # Cached stop sign actors (refreshed on reset -- static per town).
        self._stop_sign_actors: list = []

    # -- Gymnasium interface ---------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._wrong_way_counter = 0
        self._red_light_counter = 0
        self._off_road_counter = 0
        self._prev_speed_kmh = 0.0
        self._prev_in_junction = False
        self._stop_sign_states = {}
        try:
            self._stop_sign_actors = list(
                self.env.world.get_actors().filter("traffic.stop")
            )
        except Exception:
            self._stop_sign_actors = []
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        vehicle = self.env.vehicle
        speed_kmh = info["speed_kmh"]

        violations = {
            # Tier 1
            "collision": info.get("collision", False),
            "red_light": False,
            "wrong_way": False,
            "off_road": False,
            "double_solid_crossing": False,
            # Tier 2
            "speeding": False,
            "speeding_penalty": 0.0,
            "tailgating": False,
            "tailgating_penalty": 0.0,
            "stop_sign_violation": False,
            "solid_lane_crossing": False,
            "solid_lane_penalty": 0.0,
            "failure_to_yield": False,
            "yield_penalty": 0.0,
            "total_rule_penalty": 0.0,
        }

        if terminated or vehicle is None:
            # Episode already ending from collision -- skip further checks.
            info["road_rule_monitor"] = violations
            self._prev_speed_kmh = speed_kmh
            return obs, reward, terminated, truncated, info

        rule_penalty = 0.0
        carla_map = self.env.map

        # -- Lane marking violations (Tier 1 and Tier 2) ---------------------
        for mt in info.get("crossed_lane_marking_types", []):
            if mt == carla.LaneMarkingType.SolidSolid:
                violations["double_solid_crossing"] = True
                rule_penalty += _TIER1_PENALTY
                terminated = True
                break
            elif mt in (
                carla.LaneMarkingType.Solid,
                carla.LaneMarkingType.SolidBroken,
                carla.LaneMarkingType.BrokenSolid,
            ):
                violations["solid_lane_crossing"] = True
                violations["solid_lane_penalty"] = _SOLID_LANE_ADDITIVE
                rule_penalty += _SOLID_LANE_ADDITIVE

        if not terminated:
            # -- Wrong-way detection (Tier 1) ---------------------------------
            try:
                wp = carla_map.get_waypoint(
                    vehicle.get_location(), project_to_road=True
                )
                if not wp.is_junction:
                    fwd_v = vehicle.get_transform().get_forward_vector()
                    fwd_r = wp.transform.get_forward_vector()
                    dot = fwd_v.x * fwd_r.x + fwd_v.y * fwd_r.y
                    if dot < _WRONG_WAY_DOT_THRESH:
                        self._wrong_way_counter += 1
                    else:
                        self._wrong_way_counter = 0
                    if self._wrong_way_counter >= _WRONG_WAY_GRACE_FRAMES:
                        violations["wrong_way"] = True
                        rule_penalty += _TIER1_PENALTY
                        terminated = True
                        self._wrong_way_counter = 0
                else:
                    self._wrong_way_counter = 0
            except Exception:
                self._wrong_way_counter = 0

        if not terminated:
            # -- Off-road detection (Tier 1) ----------------------------------
            try:
                wp_raw = carla_map.get_waypoint(
                    vehicle.get_location(), project_to_road=False
                )
                is_off = wp_raw is None or wp_raw.lane_type not in (
                    carla.LaneType.Driving,
                    carla.LaneType.Parking,
                    carla.LaneType.Bidirectional,
                )
                if is_off:
                    self._off_road_counter += 1
                else:
                    self._off_road_counter = 0
                if self._off_road_counter >= _OFF_ROAD_GRACE_FRAMES:
                    violations["off_road"] = True
                    rule_penalty += _TIER1_PENALTY
                    terminated = True
                    self._off_road_counter = 0
            except Exception:
                self._off_road_counter = 0

        if not terminated:
            # -- Red light running (Tier 1) -----------------------------------
            try:
                if vehicle.is_at_traffic_light():
                    tl_state = vehicle.get_traffic_light_state()
                    if tl_state == carla.TrafficLightState.Red:
                        tl = vehicle.get_traffic_light()
                        dist = float("inf")
                        if tl is not None:
                            try:
                                stop_wps = tl.get_stop_waypoints()
                                if stop_wps:
                                    veh_loc = vehicle.get_location()
                                    dist = min(
                                        math.sqrt(
                                            (veh_loc.x - swp.transform.location.x) ** 2
                                            + (veh_loc.y - swp.transform.location.y) ** 2
                                        )
                                        for swp in stop_wps
                                    )
                            except Exception:
                                dist = float("inf")
                        accel = speed_kmh - self._prev_speed_kmh   # km/h per step
                        if (
                            dist < _RED_LIGHT_DIST_M
                            and speed_kmh > _RED_LIGHT_SPEED_KMH
                            and accel > _RED_LIGHT_DECEL_THRESH
                        ):
                            self._red_light_counter += 1
                        else:
                            self._red_light_counter = 0
                        if self._red_light_counter >= _RED_LIGHT_CONFIRM_FRAMES:
                            violations["red_light"] = True
                            rule_penalty += _TIER1_PENALTY
                            terminated = True
                            self._red_light_counter = 0
                    else:
                        self._red_light_counter = 0
                else:
                    self._red_light_counter = 0
            except Exception:
                self._red_light_counter = 0

        if not terminated:
            # -- Speeding (Tier 2) --------------------------------------------
            try:
                speed_limit = float(vehicle.get_speed_limit())
                if speed_limit > 0.0 and speed_kmh > speed_limit:
                    excess_frac = (speed_kmh - speed_limit) / speed_limit
                    spd_pen = -(excess_frac * _SPEED_PENALTY_SCALE)
                    violations["speeding"] = True
                    violations["speeding_penalty"] = spd_pen
                    rule_penalty += spd_pen
            except Exception:
                pass

            # -- Tailgating (Tier 2) ------------------------------------------
            try:
                ego_loc = vehicle.get_location()
                ego_fwd = vehicle.get_transform().get_forward_vector()
                cone_thresh = math.cos(math.radians(_TAILGATE_CONE_DEG))
                closest_dist = float("inf")
                for other in self.env.world.get_actors().filter("vehicle.*"):
                    if other.id == vehicle.id:
                        continue
                    o_loc = other.get_location()
                    dx = o_loc.x - ego_loc.x
                    dy = o_loc.y - ego_loc.y
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist > _TAILGATE_MAX_M or dist < 0.1:
                        continue
                    dot = (dx * ego_fwd.x + dy * ego_fwd.y) / dist
                    if dot >= cone_thresh:
                        closest_dist = min(closest_dist, dist)
                safe_dist = max((speed_kmh / 3.6) * _SAFE_FOLLOW_S, _TAILGATE_MIN_M)
                if closest_dist < safe_dist:
                    tg_pen = -1.0 * (1.0 - closest_dist / safe_dist)
                    violations["tailgating"] = True
                    violations["tailgating_penalty"] = tg_pen
                    rule_penalty += tg_pen
            except Exception:
                pass

            # -- Stop sign (Tier 2) -------------------------------------------
            try:
                ego_loc = vehicle.get_location()
                ego_fwd = vehicle.get_transform().get_forward_vector()
                for sign in self._stop_sign_actors:
                    try:
                        sign_id = sign.id
                        s_loc = sign.get_location()
                        dx = s_loc.x - ego_loc.x
                        dy = s_loc.y - ego_loc.y
                        dist = math.sqrt(dx * dx + dy * dy)

                        if sign_id not in self._stop_sign_states:
                            self._stop_sign_states[sign_id] = {
                                "in_zone": False,
                                "has_stopped": False,
                                "cooldown": 0,
                            }
                        state = self._stop_sign_states[sign_id]

                        if state["cooldown"] > 0:
                            state["cooldown"] -= 1
                            continue

                        # Only consider signs roughly ahead (forward hemisphere)
                        dot_to_sign = (
                            (dx * ego_fwd.x + dy * ego_fwd.y) / dist
                            if dist > 0.1 else 0.0
                        )
                        in_zone = dist < _STOP_ZONE_M and dot_to_sign > 0.0

                        if in_zone:
                            if not state["in_zone"]:
                                state["in_zone"] = True
                                state["has_stopped"] = False
                            if speed_kmh < _STOP_SPEED_KMH:
                                state["has_stopped"] = True
                        elif state["in_zone"]:
                            # Exiting zone -- check if agent stopped
                            if not state["has_stopped"]:
                                violations["stop_sign_violation"] = True
                                rule_penalty += -5.0
                            state["in_zone"] = False
                            state["has_stopped"] = False
                            state["cooldown"] = _STOP_COOLDOWN_STEPS
                    except Exception:
                        continue
            except Exception:
                pass

            # -- Failure to yield at uncontrolled junction (Tier 2) ----------
            try:
                wp = carla_map.get_waypoint(
                    vehicle.get_location(), project_to_road=True
                )
                in_junction = wp.is_junction
                # One-time penalty on junction entry only
                if in_junction and not self._prev_in_junction:
                    # Uncontrolled = no traffic light AND no stop sign nearby
                    has_tl = vehicle.is_at_traffic_light()
                    has_stop = any(
                        math.sqrt(
                            (sign.get_location().x - vehicle.get_location().x) ** 2
                            + (sign.get_location().y - vehicle.get_location().y) ** 2
                        ) < _STOP_ZONE_M
                        for sign in self._stop_sign_actors
                    )
                    if not has_tl and not has_stop:
                        if speed_kmh > _YIELD_ENTRY_SPEED_KMH:
                            violations["failure_to_yield"] = True
                            violations["yield_penalty"] = _YIELD_PENALTY
                            rule_penalty += _YIELD_PENALTY
                self._prev_in_junction = in_junction
            except Exception:
                pass

        violations["total_rule_penalty"] = rule_penalty
        reward += rule_penalty
        self._prev_speed_kmh = speed_kmh
        info["road_rule_monitor"] = violations

        return obs, reward, terminated, truncated, info
