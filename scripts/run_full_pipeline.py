"""Full autonomous pipeline runner.

Runs all 9 PPO training combinations (3 sensor suites x 3 styles),
then evaluation for each suite, then causal analysis, then patches
the README with live metrics.

State is checkpointed to results/pipeline_state.json after every
completed phase so the script can resume safely if interrupted.

Usage:
    python scripts/run_full_pipeline.py
    python scripts/run_full_pipeline.py --resume   # skip already-done phases
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.agents.causal_agent import CausalAnalysisAgent
from src.agents.eval_agent import EvaluationAgent
from src.agents.ppo_agent import PPOAgent

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SENSOR_SUITES = ["single_cam", "multi_cam", "lidar"]
STYLES = ["chill", "standard", "hurry"]

BC_CHECKPOINTS = {
    "single_cam": PROJECT_ROOT / "models" / "BC_model_single_cam_best.pt",
    "multi_cam":  PROJECT_ROOT / "models" / "BC_model_multi_cam_best.pt",
    "lidar":      PROJECT_ROOT / "models" / "BC_model_lidar_best.pt",
}

RESULTS_DIR   = PROJECT_ROOT / "results"
STATE_PATH    = RESULTS_DIR / "pipeline_state.json"
LOG_PATH      = PROJECT_ROOT / "logs" / "pipeline_full.log"
README_PATH   = PROJECT_ROOT / "README.md"
EVAL_RESULTS  = RESULTS_DIR / "eval_results.json"

MAX_RETRIES = 3
RETRY_SLEEP = 30.0   # seconds between retries

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger("pipeline")

# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"ppo_completed": [], "eval_completed": [], "causal_done": False}


def save_state(state: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------

def run_with_retry(fn, label: str, max_retries: int = MAX_RETRIES):
    for attempt in range(1, max_retries + 1):
        try:
            log.info("[%s] Starting attempt %d/%d.", label, attempt, max_retries)
            result = fn()
            log.info("[%s] Completed successfully.", label)
            return result
        except Exception:
            log.error(
                "[%s] Attempt %d/%d failed:\n%s",
                label, attempt, max_retries, traceback.format_exc(),
            )
            if attempt < max_retries:
                log.info("[%s] Retrying in %.0fs...", label, RETRY_SLEEP)
                time.sleep(RETRY_SLEEP)
    log.error("[%s] All %d attempts exhausted. Skipping.", label, max_retries)
    return None


# ---------------------------------------------------------------------------
# Phase 1: PPO training
# ---------------------------------------------------------------------------

def run_ppo_phase(state: dict) -> dict:
    log.info("=" * 60)
    log.info("PHASE 1: PPO Training (9 runs)")
    log.info("=" * 60)

    for sensor_suite in SENSOR_SUITES:
        ckpt = BC_CHECKPOINTS[sensor_suite]
        if not ckpt.exists():
            log.error("BC checkpoint missing for %s: %s. Skipping suite.", sensor_suite, ckpt)
            continue

        for style in STYLES:
            label = f"{sensor_suite}-{style}"
            if label in state["ppo_completed"]:
                log.info("[PPO %s] Already done, skipping.", label)
                continue

            def train(ss=sensor_suite, st=style, c=ckpt):
                agent = PPOAgent(
                    bc_checkpoint=str(c),
                    style=st,
                    sensor_suite=ss,
                    save_dir=str(PROJECT_ROOT / "models"),
                    results_dir=str(RESULTS_DIR),
                )
                return agent.run()

            result = run_with_retry(train, label=f"PPO {label}")
            if result is not None:
                state["ppo_completed"].append(label)
                save_state(state)

    return state


# ---------------------------------------------------------------------------
# Phase 2: Evaluation
# ---------------------------------------------------------------------------

def run_eval_phase(state: dict) -> dict:
    log.info("=" * 60)
    log.info("PHASE 2: Evaluation (3 sensor suites)")
    log.info("=" * 60)

    for sensor_suite in SENSOR_SUITES:
        if sensor_suite in state["eval_completed"]:
            log.info("[Eval %s] Already done, skipping.", sensor_suite)
            continue

        # Check that at least one PPO model exists for this suite
        ppo_models = list((PROJECT_ROOT / "models").glob(f"ppo_*_{sensor_suite}_best.pt"))
        if not ppo_models:
            log.warning(
                "[Eval %s] No PPO models found. "
                "Running evaluation with BC model only.", sensor_suite,
            )

        def evaluate(ss=sensor_suite):
            agent = EvaluationAgent(
                sensor_suite=ss,
                models_dir=str(PROJECT_ROOT / "models"),
                results_dir=str(RESULTS_DIR),
                record_video=False,
            )
            return agent.run()

        result = run_with_retry(evaluate, label=f"Eval {sensor_suite}")
        if result is not None:
            state["eval_completed"].append(sensor_suite)
            save_state(state)

    return state


# ---------------------------------------------------------------------------
# Phase 3: Causal analysis
# ---------------------------------------------------------------------------

def run_causal_phase(state: dict) -> dict:
    log.info("=" * 60)
    log.info("PHASE 3: Causal Analysis")
    log.info("=" * 60)

    if state["causal_done"]:
        log.info("[Causal] Already done, skipping.")
        return state

    if not EVAL_RESULTS.exists():
        log.error("[Causal] eval_results.json not found. Cannot run causal analysis.")
        return state

    def causal():
        agent = CausalAnalysisAgent(
            results_dir=str(RESULTS_DIR),
            plots_dir=str(RESULTS_DIR / "causal_plots"),
        )
        return agent.run()

    result = run_with_retry(causal, label="Causal")
    if result is not None:
        state["causal_done"] = True
        save_state(state)

    return state


# ---------------------------------------------------------------------------
# Phase 4: Update README metrics
# ---------------------------------------------------------------------------

def update_readme_metrics() -> None:
    log.info("=" * 60)
    log.info("PHASE 4: Updating README metrics")
    log.info("=" * 60)

    if not EVAL_RESULTS.exists():
        log.error("eval_results.json not found. Cannot update README.")
        return

    with open(EVAL_RESULTS) as f:
        records = json.load(f)

    if not records:
        log.warning("eval_results.json is empty. Skipping README update.")
        return

    df = pd.DataFrame(records)

    # Aggregate by model type and driving style
    def agg_row(subset, label):
        if len(subset) == 0:
            return None
        return {
            "model": label,
            "route_completion": f"{subset['route_completion'].mean():.2f} +/- {subset['route_completion'].std():.2f}",
            "collision_rate":   f"{subset['collision_rate'].mean():.3f}",
            "lane_keeping":     f"{subset['lane_keeping_frac'].mean():.2f}",
            "avg_speed":        f"{subset['avg_speed_kmh'].mean():.1f}",
        }

    rows = []
    bc = agg_row(df[df["model_type"] == "bc"], "BC baseline")
    if bc:
        rows.append(bc)
    for style in STYLES:
        subset = df[(df["model_type"] == "ppo") & (df["driving_style"] == style)]
        row = agg_row(subset, f"PPO {style}")
        if row:
            rows.append(row)

    if not rows:
        log.warning("No aggregated rows to write. Skipping README update.")
        return

    # Build table
    header = (
        "| Model | Route Completion | Collision Rate | Lane Keeping | Avg Speed (km/h) |\n"
        "|-------|-----------------|----------------|-------------|------------------|\n"
    )
    table_lines = header
    for r in rows:
        table_lines += (
            f"| {r['model']} | {r['route_completion']} | "
            f"{r['collision_rate']} | {r['lane_keeping']} | {r['avg_speed']} |\n"
        )

    # Insert table into README before the qualitative descriptions
    readme = README_PATH.read_text(encoding="utf-8")

    metrics_block = f"\n### Results Table\n\n{table_lines}\n"

    # Replace existing results table if present, otherwise insert after section header
    if "### Results Table" in readme:
        readme = re.sub(
            r"### Results Table\n.*?(?=###|\Z)",
            metrics_block.lstrip("\n"),
            readme,
            flags=re.DOTALL,
        )
    else:
        readme = readme.replace(
            "## What the Car Learned",
            f"## What the Car Learned\n{metrics_block}",
        )

    README_PATH.write_text(readme, encoding="utf-8")
    log.info("README updated with live metrics.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full DriveNet pipeline.")
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip phases already recorded in pipeline_state.json.",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete pipeline_state.json and start fresh.",
    )
    args = parser.parse_args()

    if args.reset and STATE_PATH.exists():
        STATE_PATH.unlink()
        log.info("Pipeline state reset.")

    state = load_state() if args.resume or not args.reset else load_state()

    t0 = time.time()
    log.info("Pipeline started. State: %s", state)

    state = run_ppo_phase(state)
    state = run_eval_phase(state)
    state = run_causal_phase(state)
    update_readme_metrics()

    elapsed = (time.time() - t0) / 3600
    log.info("Pipeline complete in %.1f hours.", elapsed)
    log.info("PPO completed:  %s", state["ppo_completed"])
    log.info("Eval completed: %s", state["eval_completed"])
    log.info("Causal done:    %s", state["causal_done"])


if __name__ == "__main__":
    main()
