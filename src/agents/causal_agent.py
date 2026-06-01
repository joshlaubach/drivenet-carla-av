"""CausalAnalysisAgent -- WAT Framework / Workflow 05

Reads: workflows/05_causal_analysis.md
Sequences: eval_results.json loading, LogisticRegression, cKDTree PSM,
           bootstrap CI, Rosenbaum sensitivity, matplotlib plots

The primary causal question is: how much does sensor suite (single-camera
vs. multi-camera vs. lidar) affect driving performance? Sensor suite is
the primary treatment; weather, town, and driving style are confounders.

Analysis is restricted to PPO model records only. BC records are excluded
because BC is an intermediate training artifact, not the deployed policy.

Rosenbaum sensitivity bounds are computed only for the primary (sensor suite)
treatments, since those are the main empirical claims of the project.

Usage:
    from src.agents.causal_agent import CausalAnalysisAgent
    agent = CausalAnalysisAgent()
    agent.run()
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from src.config import load_config, require_keys

log = logging.getLogger(__name__)


class CausalAnalysisAgent:
    """Estimates causal effects of driving conditions on model performance via
    propensity score matching, per workflows/05_causal_analysis.md.

    Does not implement PSM math -- sequences LogisticRegression,
    linear_sum_assignment (optimal 1:1 matching without replacement),
    bootstrap resampling, and Rosenbaum sensitivity bounds.
    """

    def __init__(
        self,
        results_dir: str = "results",
        plots_dir: str = "results/causal_plots",
    ) -> None:
        self.cfg = load_config("causal")
        require_keys(
            self.cfg,
            ["n_bootstrap", "min_treated", "random_seed",
             "primary_treatments", "secondary_treatments", "ppo_only",
             "did_comparisons"],
            "causal",
        )

        self.results_dir = Path(results_dir)
        self.plots_dir = Path(plots_dir)
        self.rng = np.random.default_rng(self.cfg["random_seed"])

        # Merge primary and secondary into a flat list for iteration,
        # preserving the rosenbaum flag from each treatment spec.
        raw_primary = self.cfg["primary_treatments"]
        raw_secondary = self.cfg["secondary_treatments"]
        self.treatments = self._build_treatments(raw_primary + raw_secondary)

    # -- Public entry point ---------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run PSM and DiD analyses. Returns dict with 'psm' and 'did' result lists."""
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        all_records = self._load_eval_results()
        if not all_records:
            raise RuntimeError(
                f"eval_results.json is empty or missing in {self.results_dir}. "
                "Run EvaluationAgent first."
            )

        # DiD uses both BC and PPO records (baseline/PID excluded).
        did_records = [
            r for r in all_records
            if r.get("model_type") in ("bc", "ppo")
        ]
        did_results = self._run_did_analysis(did_records)

        # PSM uses PPO records only (existing behaviour).
        psm_records = all_records
        if self.cfg.get("ppo_only", True):
            n_before = len(psm_records)
            psm_records = [r for r in psm_records if r.get("model_type") == "ppo"]
            log.info(
                "PPO-only filter: kept %d of %d records.", len(psm_records), n_before
            )
            if not psm_records:
                raise RuntimeError(
                    "No PPO records found in eval_results.json. "
                    "Run EvaluationAgent with PPO models first."
                )

        df = self._build_dataframe(psm_records)
        psm_results: list[dict[str, Any]] = []

        for treatment in self.treatments:
            log.info("Analysing treatment: %s", treatment["name"])
            result = self._analyse_treatment(df, treatment)
            if result is not None:
                psm_results.append(result)
                log.info(
                    "  ATE=%.4f  95%% CI=[%.4f, %.4f]  Gamma=%.2f",
                    result["ate"], result["ci_lower"], result["ci_upper"],
                    result["rosenbaum_gamma"],
                )
            else:
                log.warning(
                    "Treatment '%s' skipped (insufficient units).",
                    treatment["name"],
                )

        output = {"psm": psm_results, "did": did_results}
        out_path = self.results_dir / "causal_results.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        log.info("Causal results saved to %s.", out_path)

        self._generate_pdf_report(psm_results, did_results)
        return output

    # -- Filter DSL -----------------------------------------------------------

    @staticmethod
    def _build_filter(
        condition: dict[str, Any] | list[dict[str, Any]],
    ) -> Callable[[dict[str, Any]], bool]:
        """Convert a declarative filter spec to a callable predicate.

        Supports two operators:
        - eq: field == value
        - in: field in value (value must be a list)

        A list of conditions is combined with AND.
        """
        if isinstance(condition, list):
            filters = [CausalAnalysisAgent._build_filter(c) for c in condition]
            return lambda r: all(f(r) for f in filters)

        field = condition["field"]
        op = condition["op"]
        value = condition["value"]

        if op == "eq":
            return lambda r, _f=field, _v=value: r.get(_f) == _v
        elif op == "in":
            _vset = set(value)
            return lambda r, _f=field, _vs=_vset: r.get(_f) in _vs
        else:
            raise ValueError(
                f"Unsupported filter operator '{op}'. Supported: 'eq', 'in'."
            )

    def _build_treatments(
        self, raw_treatments: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Reconstruct treatment dicts with callable filters from YAML specs."""
        treatments: list[dict[str, Any]] = []
        for t in raw_treatments:
            treatments.append({
                "name": t["name"],
                "outcome": t["outcome"],
                "treated_filter": self._build_filter(t["treated_condition"]),
                "control_filter": self._build_filter(t["control_condition"]),
                "covariates": t["covariates"],
                "rosenbaum": bool(t.get("rosenbaum", False)),
            })
        return treatments

    # -- PSM pipeline per treatment -------------------------------------------

    def _analyse_treatment(
        self, df: dict[str, np.ndarray], treatment: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Run the full PSM pipeline for one treatment. Returns result or None."""
        min_treated: int = self.cfg["min_treated"]
        n_bootstrap: int = self.cfg["n_bootstrap"]

        treated_idx = [
            i for i, r in enumerate(df["records"])
            if treatment["treated_filter"](r)
        ]
        control_idx = [
            i for i, r in enumerate(df["records"])
            if treatment["control_filter"](r)
        ]

        if len(treated_idx) < min_treated or len(control_idx) < min_treated:
            return None

        all_idx = treated_idx + control_idx
        T = np.array([1] * len(treated_idx) + [0] * len(control_idx))
        X = np.column_stack([df[cov][all_idx] for cov in treatment["covariates"]])

        lr = LogisticRegression(
            C=1.0, max_iter=500, random_state=self.cfg["random_seed"]
        )
        lr.fit(X, T)
        pscore = lr.predict_proba(X)[:, 1]

        pscore_treated = pscore[T == 1]
        pscore_control = pscore[T == 0]

        # Optimal 1:1 matching without replacement via linear sum assignment.
        # Cost matrix: |pscore_treated[i] - pscore_control[j]| -- shape (n_t, n_c).
        # linear_sum_assignment minimises total cost, selecting at most min(n_t, n_c) pairs.
        cost = np.abs(pscore_treated[:, None] - pscore_control[None, :])
        row_ind, col_ind = linear_sum_assignment(cost)
        # Keep only the matched subset (row_ind is a contiguous range when n_t <= n_c)
        pscore_treated = pscore_treated[row_ind]
        control_pool = np.where(T == 0)[0]
        matched_control_local = control_pool[col_ind]
        pscore_control_matched = pscore_control[col_ind]

        outcome = df[treatment["outcome"]]
        treated_pool = np.where(T == 1)[0]
        outcomes_treated = outcome[treated_pool[row_ind]]
        outcomes_control = outcome[matched_control_local]

        ate = float(np.mean(outcomes_treated) - np.mean(outcomes_control))

        # Bootstrap confidence interval
        n_pairs = len(outcomes_treated)
        boot_ates = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            resample = self.rng.integers(0, n_pairs, size=n_pairs)
            boot_ates[b] = (
                outcomes_treated[resample].mean()
                - outcomes_control[resample].mean()
            )
        ci_lower = float(np.percentile(boot_ates, 2.5))
        ci_upper = float(np.percentile(boot_ates, 97.5))

        # Rosenbaum sensitivity bound (primary treatments only)
        if treatment["rosenbaum"]:
            gamma = self._rosenbaum_gamma(outcomes_treated, outcomes_control)
        else:
            gamma = float("nan")

        overlap_ok = self._check_overlap(pscore_treated, pscore_control, treatment["name"])
        if not overlap_ok:
            log.warning(
                "Treatment '%s': poor propensity overlap -- ATE may be unreliable.",
                treatment["name"],
            )

        try:
            self._plot_pscore(
                pscore_treated, pscore_control,
                pscore_treated, pscore_control_matched,
                treatment["name"],
            )
        except Exception as exc:
            log.debug("Plot failed for %s: %s", treatment["name"], exc)

        return {
            "treatment": treatment["name"],
            "outcome": treatment["outcome"],
            "is_primary": treatment["rosenbaum"],
            "n_treated": int(len(treated_idx)),
            "n_control_matched": int(n_pairs),
            "ate": round(ate, 6),
            "ci_lower": round(ci_lower, 6),
            "ci_upper": round(ci_upper, 6),
            "rosenbaum_gamma": round(gamma, 2) if not math.isnan(gamma) else None,
            "overlap_ok": overlap_ok,
        }

    # -- Difference-in-Differences --------------------------------------------

    def _run_did_analysis(
        self, records: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Run DiD for all suite pairs defined in configs/causal.yaml."""
        results: list[dict[str, Any]] = []
        for spec in self.cfg["did_comparisons"]:
            log.info("DiD analysis: %s", spec["name"])
            result = self._analyse_did(
                records,
                treated_suite=spec["treated_suite"],
                control_suite=spec["control_suite"],
                name=spec["name"],
            )
            if result is not None:
                results.append(result)
                log.info(
                    "  DiD=%.4f  95%% CI=[%.4f, %.4f]",
                    result["did"], result["ci_lower"], result["ci_upper"],
                )
            else:
                log.warning("DiD '%s' skipped (insufficient records).", spec["name"])
        return results

    def _analyse_did(
        self,
        records: list[dict[str, Any]],
        treated_suite: str,
        control_suite: str,
        name: str,
    ) -> dict[str, Any] | None:
        """Compute DiD = (PPO_treated - BC_treated) - (PPO_control - BC_control).

        Pre-period = BC model output; post-period = PPO model output.
        No PSM needed: suite assignment is fixed by design, so the DiD
        estimator already controls for all suite-invariant confounders.
        """
        min_n = self.cfg["min_treated"]
        n_bootstrap = self.cfg["n_bootstrap"]

        def _rc(suite: str, mtype: str) -> np.ndarray:
            return np.array([
                r["route_completion"] for r in records
                if r.get("sensor_suite") == suite and r.get("model_type") == mtype
            ])

        ppo_t = _rc(treated_suite, "ppo")
        bc_t = _rc(treated_suite, "bc")
        ppo_c = _rc(control_suite, "ppo")
        bc_c = _rc(control_suite, "bc")

        if any(len(x) < min_n for x in (ppo_t, bc_t, ppo_c, bc_c)):
            return None

        did = float(
            (ppo_t.mean() - bc_t.mean()) - (ppo_c.mean() - bc_c.mean())
        )

        boot_dids = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            pt = ppo_t[self.rng.integers(0, len(ppo_t), len(ppo_t))]
            bt = bc_t[self.rng.integers(0, len(bc_t), len(bc_t))]
            pc = ppo_c[self.rng.integers(0, len(ppo_c), len(ppo_c))]
            bc = bc_c[self.rng.integers(0, len(bc_c), len(bc_c))]
            boot_dids[b] = (pt.mean() - bt.mean()) - (pc.mean() - bc.mean())

        return {
            "name": name,
            "treated_suite": treated_suite,
            "control_suite": control_suite,
            "ppo_treated_mean": round(float(ppo_t.mean()), 4),
            "bc_treated_mean": round(float(bc_t.mean()), 4),
            "ppo_control_mean": round(float(ppo_c.mean()), 4),
            "bc_control_mean": round(float(bc_c.mean()), 4),
            "did": round(did, 6),
            "ci_lower": round(float(np.percentile(boot_dids, 2.5)), 6),
            "ci_upper": round(float(np.percentile(boot_dids, 97.5)), 6),
            "n_ppo_treated": int(len(ppo_t)),
            "n_bc_treated": int(len(bc_t)),
            "n_ppo_control": int(len(ppo_c)),
            "n_bc_control": int(len(bc_c)),
        }

    # -- Rosenbaum sensitivity ------------------------------------------------

    def _rosenbaum_gamma(
        self,
        outcomes_treated: np.ndarray,
        outcomes_control: np.ndarray,
        gamma_max: float = 3.0,
        n_steps: int = 28,
    ) -> float:
        """Find the smallest Gamma at which the Wilcoxon signed-rank test
        exceeds p = 0.05 under worst-case unmeasured confounding."""
        diff = outcomes_treated - outcomes_control
        if len(diff) < 4:
            return 1.0

        w_obs, _ = stats.wilcoxon(diff, alternative="greater")
        for gamma in np.linspace(1.0, gamma_max, n_steps):
            if self._wilcoxon_sensitivity_p(diff, w_obs, gamma) > 0.05:
                return float(gamma)
        return float(gamma_max)

    def _wilcoxon_sensitivity_p(
        self, diff: np.ndarray, w_obs: float, gamma: float
    ) -> float:
        """Upper bound on the one-sided Wilcoxon p-value under confounding of size Gamma."""
        p_max = gamma / (1.0 + gamma)
        nonzero = diff[diff != 0]
        n_nz = len(nonzero)
        if n_nz == 0:
            return 1.0
        ranks = np.arange(1, n_nz + 1, dtype=float)
        mu = p_max * ranks.sum()
        var = p_max * (1.0 - p_max) * (ranks ** 2).sum()
        if var == 0:
            return 1.0
        z = (w_obs - mu) / math.sqrt(var)
        return float(stats.norm.sf(z))

    # -- Overlap diagnostic ---------------------------------------------------

    def _check_overlap(
        self, pscore_t: np.ndarray, pscore_c: np.ndarray, name: str
    ) -> bool:
        overlap = (
            min(pscore_t.max(), pscore_c.max())
            - max(pscore_t.min(), pscore_c.min())
        )
        if overlap < 0:
            log.warning("%s: no overlap in propensity scores.", name)
            return False
        frac_near_boundary = (
            np.mean(pscore_t > 0.9) + np.mean(pscore_c < 0.1)
        ) / 2
        return bool(frac_near_boundary < 0.5)

    # -- Data preparation -----------------------------------------------------

    def _build_dataframe(
        self, records: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Convert episode records to parallel numpy arrays for PSM.

        Each column in the returned dict is a 1-D numpy array of length N,
        where N is the number of records. The 'records' key holds the original
        list for filter functions that operate on raw record dicts.
        """
        weather_enc = LabelEncoder()
        town_enc = LabelEncoder()
        suite_enc = LabelEncoder()

        weather_enc.fit([r["weather"] for r in records])
        town_enc.fit([r["town"] for r in records])
        suite_enc.fit([r.get("sensor_suite", "single_cam") for r in records])

        df: dict[str, Any] = {
            "records": records,
            "route_completion": np.array(
                [r["route_completion"] for r in records]
            ),
            "collision_rate": np.array(
                [float(r.get("collision_rate", 0.0)) for r in records]
            ),
            "lane_keeping_frac": np.array(
                [float(r.get("lane_keeping_frac", 0.0)) for r in records]
            ),
            "avg_speed_kmh": np.array(
                [float(r.get("avg_speed_kmh", 0.0)) for r in records]
            ),
            "weather_code": weather_enc.transform(
                [r["weather"] for r in records]
            ).astype(float),
            "town_code": town_enc.transform(
                [r["town"] for r in records]
            ).astype(float),
            "sensor_suite_code": suite_enc.transform(
                [r.get("sensor_suite", "single_cam") for r in records]
            ).astype(float),
        }

        if any("driving_style" in r for r in records):
            style_enc = LabelEncoder()
            styles = [r.get("driving_style", "n/a") for r in records]
            style_enc.fit(styles)
            df["driving_style_code"] = style_enc.transform(styles).astype(float)

        return df

    def _load_eval_results(self) -> list[dict[str, Any]]:
        path = self.results_dir / "eval_results.json"
        if not path.exists():
            return []
        with open(path) as f:
            return json.load(f)

    # -- PDF report -----------------------------------------------------------

    def _generate_pdf_report(
        self,
        causal_results: list[dict[str, Any]],
        did_results: list[dict[str, Any]] | None = None,
    ) -> None:
        """Generate a professional PDF report summarizing the causal analysis.

        The report is written at undergraduate reading level and covers:
        - Study design and DAG description
        - Primary sensor suite results table with Rosenbaum bounds
        - Secondary condition effect results table
        - Interpretation notes

        Requires matplotlib. Skips silently if matplotlib is unavailable.
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
        except ImportError:
            log.warning("matplotlib not available -- skipping PDF report.")
            return

        pdf_path = self.results_dir / "causal_report.pdf"
        primary = [r for r in causal_results if r.get("is_primary")]
        secondary = [r for r in causal_results if not r.get("is_primary")]

        with PdfPages(str(pdf_path)) as pdf:
            # -- Page 1: Title and study design --
            fig = plt.figure(figsize=(8.5, 11))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")

            title_lines = [
                "DriveNet Causal Analysis Report",
                "Sensor Suite Effects on Autonomous Driving Performance",
            ]
            body_lines = [
                "",
                "Study Design",
                "------------",
                "We trained three separate DriveNet models on the CARLA simulator,",
                "one for each sensor suite: single front-facing camera (control),",
                "three-camera late-fusion (multi_cam), and 64-channel rooftop lidar",
                "projected to a Bird's Eye View image (lidar).",
                "",
                "All models used the same PPO fine-tuning pipeline and were evaluated",
                "on three held-out towns (Town01, Town03, Town05) across three weather",
                "conditions (ClearNoon, HardRainNoon, ClearNight) with 10 episodes per",
                "condition. This produced 90 episodes per model per sensor suite.",
                "",
                "Causal Identification Strategy",
                "-------------------------------",
                "Sensor suite is assigned by design (not by the environment), so there",
                "is no structural backdoor path. Propensity score matching (PSM) controls",
                "for accidental imbalances in weather and town across sensor suites.",
                "Rosenbaum sensitivity bounds quantify how strong hidden confounding",
                "would need to be to overturn the sensor suite findings.",
                "",
                "Confounders: weather condition, evaluation town.",
                "Outcome: route completion fraction (proportion of route driven).",
                "",
                "This analysis uses PPO model records only. BC (behavior cloning)",
                "records are excluded because BC is an intermediate training step,",
                "not the deployed policy.",
            ]

            y = 0.95
            ax.text(0.5, y, title_lines[0], ha="center", va="top",
                    fontsize=16, fontweight="bold", transform=ax.transAxes)
            y -= 0.04
            ax.text(0.5, y, title_lines[1], ha="center", va="top",
                    fontsize=11, color="#444444", transform=ax.transAxes)
            y -= 0.06
            for line in body_lines:
                if line and not line.startswith("--"):
                    weight = "bold" if line.endswith(("Design", "Strategy")) else "normal"
                    ax.text(0.08, y, line, ha="left", va="top",
                            fontsize=9, fontweight=weight, transform=ax.transAxes)
                y -= 0.035 if line else 0.02

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # -- Page 2: Primary results table --
            if primary:
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis("off")
                ax.text(0.5, 0.97, "Primary Results: Sensor Suite Comparison",
                        ha="center", va="top", fontsize=14, fontweight="bold",
                        transform=ax.transAxes)
                ax.text(
                    0.5, 0.93,
                    "Average treatment effect (ATE) on route completion. "
                    "Positive ATE means the treated sensor suite performs better.",
                    ha="center", va="top", fontsize=9, color="#444444",
                    transform=ax.transAxes,
                )

                col_labels = ["Comparison", "ATE", "95% CI", "Rosenbaum Gamma", "Overlap OK"]
                table_data = []
                for r in primary:
                    name = r["treatment"].replace("sensor_", "").replace("_vs_", " vs. ")
                    ate_str = f"{r['ate']:+.4f}"
                    ci_str = f"[{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]"
                    gamma_str = f"{r['rosenbaum_gamma']:.2f}" if r["rosenbaum_gamma"] else "n/a"
                    overlap_str = "Yes" if r["overlap_ok"] else "No"
                    table_data.append([name, ate_str, ci_str, gamma_str, overlap_str])

                tbl = ax.table(
                    cellText=table_data,
                    colLabels=col_labels,
                    loc="center",
                    bbox=[0.0, 0.55, 1.0, 0.30],
                )
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(9)

                ax.text(
                    0.08, 0.52,
                    "How to read this table:",
                    ha="left", va="top", fontsize=10, fontweight="bold",
                    transform=ax.transAxes,
                )
                interp_lines = [
                    "ATE: the estimated difference in route completion between the treated and",
                    "     control sensor suite, after matching on weather and town.",
                    "95% CI: bootstrap confidence interval (1,000 resamples). If this does",
                    "        not include zero, the effect is statistically significant.",
                    "Rosenbaum Gamma: the minimum strength of hidden confounding needed to",
                    "                 overturn the finding. Higher is better (more robust).",
                    "Overlap OK: whether the propensity scores overlap well enough for PSM",
                    "            to give reliable estimates.",
                ]
                y = 0.49
                for line in interp_lines:
                    ax.text(0.08, y, line, ha="left", va="top", fontsize=8,
                            transform=ax.transAxes)
                    y -= 0.032

                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

            # -- Page 3: Secondary results table --
            if secondary:
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis("off")
                ax.text(0.5, 0.97, "Secondary Results: Condition Effects",
                        ha="center", va="top", fontsize=14, fontweight="bold",
                        transform=ax.transAxes)
                ax.text(
                    0.5, 0.93,
                    "ATE of driving conditions on route completion within PPO data.",
                    ha="center", va="top", fontsize=9, color="#444444",
                    transform=ax.transAxes,
                )

                col_labels = ["Treatment", "Outcome", "ATE", "95% CI", "Overlap OK"]
                table_data = []
                for r in secondary:
                    ate_str = f"{r['ate']:+.4f}"
                    ci_str = f"[{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]"
                    overlap_str = "Yes" if r["overlap_ok"] else "No"
                    table_data.append([
                        r["treatment"], r["outcome"], ate_str, ci_str, overlap_str
                    ])

                tbl = ax.table(
                    cellText=table_data,
                    colLabels=col_labels,
                    loc="center",
                    bbox=[0.0, 0.65, 1.0, 0.22],
                )
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(9)

                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

            # -- Page 4: DiD results table --
            if did_results:
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis("off")
                ax.text(
                    0.5, 0.97,
                    "Difference-in-Differences: PPO Fine-Tuning Benefit by Suite",
                    ha="center", va="top", fontsize=14, fontweight="bold",
                    transform=ax.transAxes,
                )
                ax.text(
                    0.5, 0.93,
                    "DiD = (PPO_treated − BC_treated) − (PPO_control − BC_control). "
                    "Positive DiD means PPO fine-tuning helped the treated suite more.",
                    ha="center", va="top", fontsize=9, color="#444444",
                    wrap=True, transform=ax.transAxes,
                )

                col_labels = [
                    "Comparison", "PPO−BC (treated)", "PPO−BC (control)",
                    "DiD", "95% CI",
                ]
                table_data = []
                for r in did_results:
                    name = r["name"].replace("did_", "").replace("_vs_", " vs. ")
                    delta_t = r["ppo_treated_mean"] - r["bc_treated_mean"]
                    delta_c = r["ppo_control_mean"] - r["bc_control_mean"]
                    table_data.append([
                        name,
                        f"{delta_t:+.4f}",
                        f"{delta_c:+.4f}",
                        f"{r['did']:+.4f}",
                        f"[{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]",
                    ])

                tbl = ax.table(
                    cellText=table_data,
                    colLabels=col_labels,
                    loc="center",
                    bbox=[0.0, 0.60, 1.0, 0.25],
                )
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(9)

                interp_lines = [
                    "How to read this table:",
                    "PPO−BC (treated/control): improvement from behavior cloning to PPO for each suite.",
                    "DiD: the *extra* improvement the treated suite gained from PPO relative to control.",
                    "     A positive, CI-excluding-zero DiD means PPO benefited the treated suite more,",
                    "     suggesting the sensor modality interacts with the RL fine-tuning signal.",
                    "Parallel-trends assumption: both suites would have improved equally without PPO.",
                    "     This holds if BC training quality is similar across suites (it is by design).",
                ]
                y = 0.57
                for i, line in enumerate(interp_lines):
                    weight = "bold" if i == 0 else "normal"
                    ax.text(0.08, y, line, ha="left", va="top", fontsize=8,
                            fontweight=weight, transform=ax.transAxes)
                    y -= 0.032

                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        log.info("PDF report saved to %s.", pdf_path)

    # -- Plotting -------------------------------------------------------------

    def _plot_pscore(
        self,
        ps_treated_before: np.ndarray,
        ps_control_before: np.ndarray,
        ps_treated_after: np.ndarray,
        ps_control_after: np.ndarray,
        name: str,
    ) -> None:
        """Save before/after propensity score distribution plots."""
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.hist(ps_treated_before, bins=20, alpha=0.6, label="Treated", color="steelblue")
        ax1.hist(ps_control_before, bins=20, alpha=0.6, label="Control", color="coral")
        ax1.set_title(f"{name} -- Before Matching")
        ax1.set_xlabel("Propensity Score")
        ax1.legend()

        ax2.hist(ps_treated_after, bins=20, alpha=0.6, label="Treated", color="steelblue")
        ax2.hist(ps_control_after, bins=20, alpha=0.6, label="Matched Control", color="coral")
        ax2.set_title(f"{name} -- After Matching")
        ax2.set_xlabel("Propensity Score")
        ax2.legend()

        fig.tight_layout()
        path = self.plots_dir / f"psm_{name}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        log.debug("Plot saved: %s", path)
