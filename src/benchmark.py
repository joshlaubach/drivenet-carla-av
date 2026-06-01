"""BenchmarkReport -- publication-quality analysis of eval_results.json.

Reads eval_results.json and produces three figures:
1. report_card()       -- sensor suite x 5 metrics grouped bar chart (95% bootstrap CIs)
2. failure_taxonomy()  -- violation type x sensor suite heat map
3. scenario_coverage() -- town x weather x sensor suite coverage matrix

No CARLA dependency. All methods accept an optional output_dir; if given,
figures are saved as PNG (300 DPI) and PDF.

CARLA Leaderboard Infraction Score multipliers (D8 decision):
    IS = 0.70^red_light x 0.70^off_road x 0.70^wrong_way x 0.65^double_solid
    Driving Score = route_completion x IS
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

# Infraction Score multipliers per Tier-1 violation type (CARLA Leaderboard)
_IS_MULTIPLIERS: dict[str, float] = {
    "viol_red_light": 0.70,
    "viol_off_road": 0.70,
    "viol_wrong_way": 0.70,
    "viol_double_solid": 0.65,
}

_VIOL_FIELDS = [
    "viol_red_light",
    "viol_wrong_way",
    "viol_off_road",
    "viol_double_solid",
    "viol_speeding_steps",
    "viol_tailgating_steps",
    "viol_stop_sign",
    "viol_solid_lane_steps",
    "viol_yield",
]

_VIOL_LABELS = [
    "Red Light",
    "Wrong Way",
    "Off Road",
    "Double Solid",
    "Speeding",
    "Tailgating",
    "Stop Sign",
    "Solid Lane",
    "Yield",
]

_COLOR_BLIND_PALETTE = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
]


def _driving_score(record: dict[str, Any]) -> float:
    """CARLA Leaderboard Driving Score = route_completion x infraction_score."""
    is_score = 1.0
    for field, mult in _IS_MULTIPLIERS.items():
        count = int(record.get(field, 0))
        is_score *= mult ** count
    return record["route_completion"] * is_score


def _safety_score(record: dict[str, Any]) -> float:
    """Fraction of steps without a collision or Tier-1 violation."""
    steps = max(int(record.get("total_steps", 0)), 1)
    collisions = int(record.get("collision_count", 0))
    tier1 = int(record.get("viol_tier1_total", 0))
    return max(0.0, 1.0 - (collisions + tier1) / steps)


def _bootstrap_ci(
    values: list[float],
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (mean, lower_bound, upper_bound) using percentile bootstrap."""
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return math.nan, math.nan, math.nan
    if len(arr) == 1:
        v = float(arr[0])
        return v, v, v
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_resamples)
    ])
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, 100 * alpha))
    upper = float(np.percentile(boot_means, 100 * (1.0 - alpha)))
    return float(arr.mean()), lower, upper


class BenchmarkReport:
    """Analysis and figure generation from eval_results.json.

    Parameters
    ----------
    records_path : str or Path
        Path to eval_results.json, or a list of record dicts for testing.
    exclude_baseline : bool
        When True (default), baseline records are excluded from sensor suite
        comparisons. Baseline still appears as its own bar when False.
    """

    def __init__(
        self,
        records_path: str | Path | list[dict[str, Any]],
        exclude_baseline: bool = True,
    ) -> None:
        if isinstance(records_path, list):
            self._records = records_path
        else:
            with open(records_path) as f:
                self._records = json.load(f)

        self._exclude_baseline = exclude_baseline

    # -- Internal helpers -------------------------------------------------------

    def _suite_records(self) -> dict[str, list[dict[str, Any]]]:
        """Group records by sensor_suite, optionally excluding baseline."""
        groups: dict[str, list[dict[str, Any]]] = {}
        for r in self._records:
            suite = r.get("sensor_suite", "unknown")
            if self._exclude_baseline and r.get("model_type") == "baseline":
                continue
            groups.setdefault(suite, []).append(r)
        return groups

    def _save(self, fig: plt.Figure, name: str, output_dir: Path | None) -> None:
        if output_dir is None:
            return
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")

    # -- Public API -------------------------------------------------------------

    def report_card(self, output_dir: str | Path | None = None) -> plt.Figure:
        """Sensor suite x 5 metrics grouped bar chart with 95% bootstrap CIs."""
        metric_extractors = {
            "Driving Score": _driving_score,
            "Safety Score": _safety_score,
            "Lane Keeping": lambda r: float(r.get("lane_keeping_frac", 0.0)),
            "Avg Speed (km/h)": lambda r: float(r.get("avg_speed_kmh", 0.0)),
            "Survival Rate": lambda r: float(r.get("survived", False)),
        }
        metric_names = list(metric_extractors.keys())
        groups = self._suite_records()
        suites = sorted(groups.keys())

        n_metrics = len(metric_names)
        n_suites = len(suites)
        if n_suites == 0:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return fig

        x = np.arange(n_metrics)
        width = 0.8 / max(n_suites, 1)

        fig, ax = plt.subplots(figsize=(13, 6))

        for i, suite in enumerate(suites):
            records = groups[suite]
            means, lowers, uppers = [], [], []
            for metric, fn in metric_extractors.items():
                vals = [fn(r) for r in records]
                m, lo, hi = _bootstrap_ci(vals)
                means.append(m if not math.isnan(m) else 0.0)
                lowers.append(m - lo if not math.isnan(lo) else 0.0)
                uppers.append(hi - m if not math.isnan(hi) else 0.0)

            offset = (i - n_suites / 2 + 0.5) * width
            color = _COLOR_BLIND_PALETTE[i % len(_COLOR_BLIND_PALETTE)]
            ax.bar(
                x + offset, means, width * 0.9,
                label=suite,
                color=color,
                alpha=0.85,
                yerr=[lowers, uppers],
                capsize=4,
                error_kw={"elinewidth": 1.2, "capthick": 1.2},
            )

        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, rotation=15, ha="right")
        ax.set_ylabel("Score")
        ax.set_title("Sensor Suite Report Card — 95% Bootstrap CIs")
        ax.set_ylim(bottom=0)
        ax.legend(title="Sensor Suite", loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()

        self._save(fig, "report_card", output_dir)
        return fig

    def failure_taxonomy(self, output_dir: str | Path | None = None) -> plt.Figure:
        """Violation type x sensor suite heat map (mean violations per episode)."""
        groups = self._suite_records()
        suites = sorted(groups.keys())
        n_viols = len(_VIOL_FIELDS)
        n_suites = len(suites)

        if n_suites == 0:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return fig

        matrix = np.zeros((n_viols, n_suites))
        for j, suite in enumerate(suites):
            records = groups[suite]
            n = max(len(records), 1)
            for i, field in enumerate(_VIOL_FIELDS):
                matrix[i, j] = sum(r.get(field, 0) for r in records) / n

        fig, ax = plt.subplots(figsize=(max(6, n_suites * 2.5), 7))
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
        fig.colorbar(im, ax=ax, label="Mean violations per episode")

        ax.set_xticks(range(n_suites))
        ax.set_xticklabels(suites, rotation=20, ha="right")
        ax.set_yticks(range(n_viols))
        ax.set_yticklabels(_VIOL_LABELS)
        ax.set_title("Failure Taxonomy — Mean Violations per Episode")

        for i in range(n_viols):
            for j in range(n_suites):
                val = matrix[i, j]
                text_color = "white" if val > matrix.max() * 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color=text_color)

        fig.tight_layout()
        self._save(fig, "failure_taxonomy", output_dir)
        return fig

    def scenario_coverage(self, output_dir: str | Path | None = None) -> plt.Figure:
        """Town x weather x sensor suite coverage matrix (episode count per cell)."""
        all_records = self._records
        towns = sorted({r.get("town", "unknown") for r in all_records})
        weathers = sorted({r.get("weather", "unknown") for r in all_records})
        suites = sorted({r.get("sensor_suite", "unknown") for r in all_records
                         if not (self._exclude_baseline
                                 and r.get("model_type") == "baseline")})

        n_suites = len(suites)
        if n_suites == 0:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return fig

        n_cols = n_suites
        n_rows = len(towns)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * 4, n_rows * 2.5 + 1),
            squeeze=False,
        )

        for row, town in enumerate(towns):
            for col, suite in enumerate(suites):
                ax = axes[row][col]
                matrix = np.zeros((len(weathers), 1))
                for wi, weather in enumerate(weathers):
                    count = sum(
                        1 for r in all_records
                        if r.get("town") == town
                        and r.get("weather") == weather
                        and r.get("sensor_suite") == suite
                        and not (self._exclude_baseline
                                 and r.get("model_type") == "baseline")
                    )
                    matrix[wi, 0] = count

                vmax = max(matrix.max(), 1)
                ax.imshow(matrix, aspect="auto", cmap="Blues",
                          vmin=0, vmax=vmax)
                ax.set_xticks([0])
                ax.set_xticklabels([suite], fontsize=9)
                ax.set_yticks(range(len(weathers)))
                ax.set_yticklabels(weathers if col == 0 else [], fontsize=8)
                if row == 0:
                    ax.set_title(suite, fontsize=10)
                if col == 0:
                    ax.set_ylabel(town, fontsize=10)

                for wi in range(len(weathers)):
                    cnt = int(matrix[wi, 0])
                    text_color = "white" if cnt > vmax * 0.6 else "black"
                    ax.text(0, wi, str(cnt), ha="center", va="center",
                            fontsize=9, color=text_color)

        fig.suptitle("Scenario Coverage Matrix (episode count)", fontsize=13, y=1.01)
        fig.tight_layout()
        self._save(fig, "scenario_coverage", output_dir)
        return fig
