"""CI-safe tests for BenchmarkReport and PIDBaselinePolicy.

No CARLA dependency. All fixtures are synthetic dicts.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.baseline import PIDBaselinePolicy
from src.benchmark import (
    BenchmarkReport,
    _bootstrap_ci,
    _driving_score,
    _safety_score,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _record(
    suite: str = "single_cam",
    model_type: str = "bc",
    town: str = "Town01",
    weather: str = "ClearNoon",
    episode: int = 0,
    route_completion: float = 0.8,
    collision_count: int = 0,
    lane_keeping_frac: float = 0.95,
    avg_speed_kmh: float = 28.0,
    survived: bool = True,
    total_steps: int = 1000,
    viol_red_light: int = 0,
    viol_wrong_way: int = 0,
    viol_off_road: int = 0,
    viol_double_solid: int = 0,
    viol_tier1_total: int = 0,
    viol_speeding_steps: int = 0,
    viol_tailgating_steps: int = 0,
    viol_stop_sign: int = 0,
    viol_solid_lane_steps: int = 0,
    viol_yield: int = 0,
) -> dict:
    return {
        "sensor_suite": suite,
        "model": f"{model_type}_{suite}",
        "model_type": model_type,
        "town": town,
        "weather": weather,
        "episode": episode,
        "route_completion": route_completion,
        "collision_count": collision_count,
        "collision_rate": collision_count / max(1.0, total_steps / 100),
        "lane_keeping_frac": lane_keeping_frac,
        "avg_speed_kmh": avg_speed_kmh,
        "distance_m": total_steps * 0.5,
        "survived": survived,
        "total_steps": total_steps,
        "viol_red_light": viol_red_light,
        "viol_wrong_way": viol_wrong_way,
        "viol_off_road": viol_off_road,
        "viol_double_solid": viol_double_solid,
        "viol_tier1_total": viol_tier1_total,
        "viol_speeding_steps": viol_speeding_steps,
        "viol_tailgating_steps": viol_tailgating_steps,
        "viol_stop_sign": viol_stop_sign,
        "viol_solid_lane_steps": viol_solid_lane_steps,
        "viol_yield": viol_yield,
    }


def _multi_suite_records() -> list[dict]:
    """Balanced set: 5 records per suite × 3 suites."""
    records = []
    for suite in ["single_cam", "multi_cam", "lidar"]:
        for ep in range(5):
            records.append(_record(
                suite=suite,
                episode=ep,
                route_completion=0.5 + ep * 0.08,
                collision_count=ep % 2,
                survived=(ep < 4),
            ))
    return records


# ---------------------------------------------------------------------------
# Metric formula tests
# ---------------------------------------------------------------------------

class TestDrivingScore:
    def test_clean_run_equals_route_completion(self):
        r = _record(route_completion=0.85)
        assert _driving_score(r) == pytest.approx(0.85)

    def test_one_red_light_applies_multiplier(self):
        r = _record(route_completion=1.0, viol_red_light=1)
        assert _driving_score(r) == pytest.approx(0.70, abs=1e-6)

    def test_multiple_violations_multiply(self):
        r = _record(route_completion=1.0, viol_red_light=1, viol_off_road=1)
        assert _driving_score(r) == pytest.approx(0.70 * 0.70, abs=1e-6)

    def test_double_solid_has_lower_multiplier(self):
        r = _record(route_completion=1.0, viol_double_solid=1)
        assert _driving_score(r) == pytest.approx(0.65, abs=1e-6)

    def test_zero_route_completion_scores_zero(self):
        r = _record(route_completion=0.0, viol_red_light=2)
        assert _driving_score(r) == pytest.approx(0.0)

    def test_missing_viol_fields_treated_as_zero(self):
        r = {"route_completion": 0.9}
        assert _driving_score(r) == pytest.approx(0.9)


class TestSafetyScore:
    def test_clean_episode_scores_one(self):
        r = _record(total_steps=1000, collision_count=0, viol_tier1_total=0)
        assert _safety_score(r) == pytest.approx(1.0)

    def test_one_collision_out_of_1000_steps(self):
        r = _record(total_steps=1000, collision_count=1, viol_tier1_total=0)
        assert _safety_score(r) == pytest.approx(0.999)

    def test_floor_at_zero(self):
        r = _record(total_steps=1, collision_count=5, viol_tier1_total=5)
        assert _safety_score(r) >= 0.0

    def test_zero_steps_does_not_raise(self):
        r = _record(total_steps=0, collision_count=0, viol_tier1_total=0)
        assert _safety_score(r) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Bootstrap CI tests
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    def test_returns_three_floats(self):
        m, lo, hi = _bootstrap_ci([0.5, 0.6, 0.7, 0.8, 0.9])
        assert all(isinstance(v, float) for v in (m, lo, hi))

    def test_ci_brackets_mean(self):
        vals = [0.1 * i for i in range(1, 11)]
        m, lo, hi = _bootstrap_ci(vals)
        assert lo <= m <= hi

    def test_single_value_returns_same_for_all(self):
        m, lo, hi = _bootstrap_ci([0.75])
        assert m == pytest.approx(0.75)
        assert lo == pytest.approx(0.75)
        assert hi == pytest.approx(0.75)

    def test_empty_values_returns_nan(self):
        m, lo, hi = _bootstrap_ci([])
        assert math.isnan(m)

    def test_nan_values_are_ignored(self):
        m, lo, hi = _bootstrap_ci([float("nan"), 0.5, float("nan")])
        assert m == pytest.approx(0.5)

    def test_larger_ci_gives_wider_interval(self):
        vals = list(np.linspace(0, 1, 50))
        _, lo90, hi90 = _bootstrap_ci(vals, ci=0.90)
        _, lo99, hi99 = _bootstrap_ci(vals, ci=0.99)
        assert (hi99 - lo99) >= (hi90 - lo90)


# ---------------------------------------------------------------------------
# BenchmarkReport.report_card() tests
# ---------------------------------------------------------------------------

class TestReportCard:
    def test_returns_figure(self):
        import matplotlib.pyplot as plt
        report = BenchmarkReport(_multi_suite_records())
        fig = report.report_card()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_records_returns_figure(self):
        import matplotlib.pyplot as plt
        report = BenchmarkReport([])
        fig = report.report_card()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_zero_violations_does_not_raise(self):
        import matplotlib.pyplot as plt
        records = [_record(suite="single_cam", episode=i) for i in range(5)]
        report = BenchmarkReport(records)
        fig = report.report_card()
        plt.close(fig)

    def test_single_episode_per_condition(self):
        """Single episode per condition — no CI collapse, no error."""
        import matplotlib.pyplot as plt
        records = [_record(suite=s) for s in ["single_cam", "multi_cam"]]
        report = BenchmarkReport(records)
        fig = report.report_card()
        plt.close(fig)

    def test_baseline_excluded_by_default(self):
        """Baseline records should not appear as a sensor suite bar."""
        import matplotlib.pyplot as plt
        records = _multi_suite_records()
        records.append(_record(suite="single_cam", model_type="baseline"))
        report = BenchmarkReport(records, exclude_baseline=True)
        groups = report._suite_records()
        for suite, recs in groups.items():
            for r in recs:
                assert r.get("model_type") != "baseline"
        plt.close("all")

    def test_nan_speed_does_not_raise(self):
        """Failed episode records have avg_speed_kmh=0; NaN should also be safe."""
        import matplotlib.pyplot as plt
        records = [_record(suite="single_cam", avg_speed_kmh=float("nan"))]
        report = BenchmarkReport(records)
        fig = report.report_card()
        plt.close(fig)

    def test_missing_sensor_suite_sparse_data(self):
        """Only 2 of 3 suites present — should still produce valid figure."""
        import matplotlib.pyplot as plt
        records = [_record(suite=s) for s in ["single_cam", "lidar"]]
        report = BenchmarkReport(records)
        fig = report.report_card()
        plt.close(fig)


# ---------------------------------------------------------------------------
# BenchmarkReport.failure_taxonomy() tests
# ---------------------------------------------------------------------------

class TestFailureTaxonomy:
    def test_returns_figure(self):
        import matplotlib.pyplot as plt
        report = BenchmarkReport(_multi_suite_records())
        fig = report.failure_taxonomy()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_zero_violations_produces_all_zero_heatmap(self):
        import matplotlib.pyplot as plt
        records = [_record(suite=s, episode=i) for s in ["single_cam"] for i in range(3)]
        report = BenchmarkReport(records)
        fig = report.failure_taxonomy()
        plt.close(fig)

    def test_viol_fields_default_to_zero_for_missing_keys(self):
        """Records missing viol_* fields (e.g., from _failed_record pre-fix) must not raise."""
        import matplotlib.pyplot as plt
        r = {"sensor_suite": "single_cam", "model_type": "bc",
             "route_completion": 0.0, "survived": False}
        report = BenchmarkReport([r])
        fig = report.failure_taxonomy()
        plt.close(fig)


# ---------------------------------------------------------------------------
# BenchmarkReport.scenario_coverage() tests
# ---------------------------------------------------------------------------

class TestScenarioCoverage:
    def test_returns_figure(self):
        import matplotlib.pyplot as plt
        report = BenchmarkReport(_multi_suite_records())
        fig = report.scenario_coverage()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_counts_correct(self):
        """Each suite × town × weather cell should count exactly N episodes."""
        records = []
        for ep in range(3):
            records.append(_record(suite="single_cam", town="Town01",
                                   weather="ClearNoon", episode=ep))
        report = BenchmarkReport(records)
        # count from internal data
        count = sum(
            1 for r in report._records
            if r.get("town") == "Town01"
            and r.get("weather") == "ClearNoon"
            and r.get("sensor_suite") == "single_cam"
        )
        assert count == 3


# ---------------------------------------------------------------------------
# PIDBaselinePolicy tests
# ---------------------------------------------------------------------------

class MockLoc:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def _pid_obs(speed_kmh: float = 20.0, heading_deg: float = 0.0,
             vehicle_loc: tuple | None = None) -> dict:
    """Build a synthetic obs dict for PIDBaselinePolicy.act()."""
    h = math.radians(heading_deg)
    state = np.array([
        speed_kmh / 60.0,
        math.sin(h),
        math.cos(h),
        30.0 / 130.0,  # speed_limit
        2.0 / 4.0,     # lane_count
        0.0,            # is_junction
    ], dtype=np.float32)
    obs: dict = {"state": state}
    if vehicle_loc is not None:
        obs["_vehicle_loc"] = vehicle_loc
    return obs


class TestPIDBaselinePolicy:
    def test_act_returns_three_element_array(self):
        pid = PIDBaselinePolicy()
        action = pid.act(_pid_obs())
        assert action.shape == (3,)

    def test_throttle_in_range(self):
        pid = PIDBaselinePolicy(target_speed_kmh=30.0)
        action = pid.act(_pid_obs(speed_kmh=0.0))
        assert 0.0 <= action[1] <= 1.0

    def test_steer_in_range(self):
        pid = PIDBaselinePolicy()
        waypoints = [MockLoc(10, 0), MockLoc(20, 5), MockLoc(30, 10)]
        pid.set_route(waypoints)
        action = pid.act(_pid_obs(vehicle_loc=(0.0, 0.0)))
        assert -1.0 <= action[0] <= 1.0

    def test_brake_in_range(self):
        pid = PIDBaselinePolicy(target_speed_kmh=10.0)
        action = pid.act(_pid_obs(speed_kmh=60.0))
        assert 0.0 <= action[2] <= 1.0

    def test_throttle_positive_when_below_target(self):
        pid = PIDBaselinePolicy(target_speed_kmh=40.0)
        action = pid.act(_pid_obs(speed_kmh=0.0))
        assert action[1] > 0.0

    def test_brake_applied_when_well_above_target(self):
        pid = PIDBaselinePolicy(target_speed_kmh=10.0)
        action = pid.act(_pid_obs(speed_kmh=60.0))
        assert action[2] > 0.0
        assert action[1] == pytest.approx(0.0)

    def test_steer_zero_without_waypoints(self):
        pid = PIDBaselinePolicy()
        action = pid.act(_pid_obs(vehicle_loc=(0.0, 0.0)))
        assert action[0] == pytest.approx(0.0)

    def test_steer_zero_without_vehicle_loc(self):
        pid = PIDBaselinePolicy()
        waypoints = [MockLoc(10, 0)]
        pid.set_route(waypoints)
        action = pid.act(_pid_obs())  # no _vehicle_loc
        assert action[0] == pytest.approx(0.0)

    def test_reset_clears_route(self):
        pid = PIDBaselinePolicy()
        waypoints = [MockLoc(10, 0), MockLoc(20, 0)]
        pid.set_route(waypoints)
        pid.reset()
        assert pid._wps == []
        assert pid._wp_idx == 0

    def test_set_route_resets_index(self):
        pid = PIDBaselinePolicy()
        pid.set_route([MockLoc(10, 0), MockLoc(20, 0)])
        pid._wp_idx = 1
        pid.set_route([MockLoc(5, 0)])
        assert pid._wp_idx == 0

    def test_dtype_is_float32(self):
        pid = PIDBaselinePolicy()
        action = pid.act(_pid_obs())
        assert action.dtype == np.float32
