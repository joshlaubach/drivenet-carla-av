"""Autonomous DriveNet pipeline: collect → [BC→PPO→eval] × suite → causal.

Runs all three sensor suites (single_cam, multi_cam, lidar) in suite-first
order so each suite produces a complete baseline before the next starts.
Data collection runs once and is skipped automatically if the dataset already
meets the 95% frame threshold. Causal analysis runs once at the end after all
suites finish.

Usage
-----
Full run (auto-skips collection if data exists):
    py -3.11 scripts/run_pipeline.py

Force data re-collection:
    py -3.11 scripts/run_pipeline.py --force-collect

Skip collection unconditionally:
    py -3.11 scripts/run_pipeline.py --skip-collection

Run specific suites only:
    py -3.11 scripts/run_pipeline.py --suites single_cam lidar

Resume from a specific suite (skip earlier ones):
    py -3.11 scripts/run_pipeline.py --from-suite lidar

Re-run one suite without re-collecting or re-running causal:
    py -3.11 scripts/run_pipeline.py --suites multi_cam --skip-collection

PPO resume: if models/ppo_{style}_{suite}_resume.pt exists from a prior
interrupted run, PPOAgent loads it automatically. No flag needed.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_ALL_SUITES = ["single_cam", "multi_cam", "lidar"]
_PPO_STYLES = ["chill", "standard", "hurry"]
_TOWNS = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD"]

# 54 conditions × 300 frames; 95% threshold matches NB01 validation cell.
_FRAMES_PER_TOWN = 54 * 300
_COLLECTION_THRESHOLD = int(_FRAMES_PER_TOWN * 0.95)

log = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Collection check
# ---------------------------------------------------------------------------

def _count_town_frames(town_dir: Path) -> int:
    chunks = sorted(town_dir.glob("chunk_*.npz"))
    total = 0
    for path in chunks:
        with np.load(path, allow_pickle=True) as data:
            total += int(data["images"].shape[0])
    return total


def _collection_complete(data_dir: Path) -> bool:
    """Return True if all 6 towns have at least 95% of expected frames."""
    for town in _TOWNS:
        town_dir = data_dir / town
        if not town_dir.exists():
            return False
        if _count_town_frames(town_dir) < _COLLECTION_THRESHOLD:
            return False
    return True


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def run_collection(data_dir: Path) -> dict[str, Any]:
    """Run DataCollectionAgent for all towns via collect_one_town.py subprocess."""
    import subprocess

    script = PROJECT_ROOT / "scripts" / "collect_one_town.py"
    results: dict[str, int] = {}

    for town in _TOWNS:
        log.info("Collecting %s ...", town)
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, str(script), "--town", town, "--data-dir", str(data_dir)],
            cwd=str(PROJECT_ROOT),
        )
        elapsed = time.time() - t0
        chunks = sorted((data_dir / town).glob("chunk_*.npz"))
        results[town] = len(chunks)
        status = "OK" if proc.returncode in (0, 3) else f"exit {proc.returncode}"
        log.info("%s: %d chunks in %.1f min [%s]", town, len(chunks), elapsed / 60, status)

    return {"stage": "collection", "towns": results}


def run_bc(suite: str, data_dir: Path, models_dir: Path, results_dir: Path) -> dict[str, Any]:
    from src.agents.training_agent import BehaviorCloningAgent
    log.info("[%s] Starting BC training ...", suite)
    t0 = time.time()
    agent = BehaviorCloningAgent(
        sensor_suite=suite,
        data_dir=str(data_dir),
        save_dir=str(models_dir),
        results_dir=str(results_dir),
    )
    metrics = agent.run()
    elapsed = time.time() - t0
    log.info("[%s] BC complete in %.1f min. test_loss=%.6f", suite, elapsed / 60, metrics["test_loss"])
    return {"stage": "bc", "suite": suite, "elapsed_min": round(elapsed / 60, 1), **metrics}


def run_ppo(suite: str, models_dir: Path, results_dir: Path) -> dict[str, Any]:
    from src.agents.ppo_agent import PPOAgent
    style_results: dict[str, Any] = {}

    for style in _PPO_STYLES:
        bc_ckpt = models_dir / f"BC_model_{suite}_best.pt"
        if not bc_ckpt.exists():
            log.error("[%s/%s] BC checkpoint missing: %s — skipping PPO.", suite, style, bc_ckpt)
            style_results[style] = {"skipped": True, "reason": "bc_checkpoint_missing"}
            continue

        log.info("[%s/%s] Starting PPO training ...", suite, style)
        t0 = time.time()
        try:
            agent = PPOAgent(
                bc_checkpoint=str(bc_ckpt),
                style=style,
                sensor_suite=suite,
                save_dir=str(models_dir),
                results_dir=str(results_dir),
            )
            agent.run()
            elapsed = time.time() - t0
            log.info("[%s/%s] PPO complete in %.1f min.", suite, style, elapsed / 60)
            style_results[style] = {"elapsed_min": round(elapsed / 60, 1)}
        except Exception:
            elapsed = time.time() - t0
            log.exception("[%s/%s] PPO failed after %.1f min.", suite, style, elapsed / 60)
            style_results[style] = {"failed": True, "elapsed_min": round(elapsed / 60, 1)}

    return {"stage": "ppo", "suite": suite, "styles": style_results}


def run_eval(suite: str, models_dir: Path, results_dir: Path) -> dict[str, Any]:
    from src.agents.eval_agent import EvaluationAgent
    log.info("[%s] Starting evaluation ...", suite)
    t0 = time.time()
    agent = EvaluationAgent(
        sensor_suite=suite,
        models_dir=str(models_dir),
        results_dir=str(results_dir),
    )
    records = agent.run()
    elapsed = time.time() - t0
    log.info("[%s] Eval complete in %.1f min. %d episodes recorded.", suite, elapsed / 60, len(records))
    return {"stage": "eval", "suite": suite, "episodes": len(records), "elapsed_min": round(elapsed / 60, 1)}


def run_causal(results_dir: Path) -> dict[str, Any]:
    from src.agents.causal_agent import CausalAnalysisAgent
    log.info("Starting causal analysis ...")
    t0 = time.time()
    agent = CausalAnalysisAgent(results_dir=str(results_dir))
    agent.run()
    elapsed = time.time() - t0
    log.info("Causal analysis complete in %.1f min.", elapsed / 60)
    return {"stage": "causal", "elapsed_min": round(elapsed / 60, 1)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous DriveNet pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--suites", nargs="+", choices=_ALL_SUITES, default=_ALL_SUITES,
        metavar="SUITE",
        help="Sensor suites to train and evaluate (default: all three).",
    )
    parser.add_argument(
        "--from-suite", choices=_ALL_SUITES, default=None,
        metavar="SUITE",
        help="Skip all suites before this one.",
    )
    parser.add_argument(
        "--skip-collection", action="store_true",
        help="Skip data collection unconditionally.",
    )
    parser.add_argument(
        "--force-collect", action="store_true",
        help="Re-collect data even if the dataset already looks complete.",
    )
    parser.add_argument(
        "--data-dir", default=str(PROJECT_ROOT / "data"),
        help="Path to the data directory (default: data/).",
    )
    parser.add_argument(
        "--models-dir", default=str(PROJECT_ROOT / "models"),
        help="Path to the models directory (default: models/).",
    )
    parser.add_argument(
        "--results-dir", default=str(PROJECT_ROOT / "results"),
        help="Path to the results directory (default: results/).",
    )
    args = parser.parse_args()

    if args.skip_collection and args.force_collect:
        parser.error("--skip-collection and --force-collect are mutually exclusive.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(PROJECT_ROOT / "results" / "pipeline.log", mode="a"),
        ],
    )

    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    results_dir = Path(args.results_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Resolve suite list, applying --from-suite offset
    suites = args.suites
    if args.from_suite:
        if args.from_suite not in suites:
            suites = [args.from_suite]
        else:
            suites = suites[suites.index(args.from_suite):]

    pipeline_t0 = time.time()
    all_results: list[dict[str, Any]] = []
    suite_failures: list[str] = []

    log.info("=" * 70)
    log.info("DriveNet Pipeline")
    log.info("  Suites : %s", suites)
    log.info("  Data   : %s", data_dir)
    log.info("  Models : %s", models_dir)
    log.info("  Results: %s", results_dir)
    log.info("=" * 70)

    # -- Data collection -------------------------------------------------------
    if args.skip_collection:
        log.info("Collection: SKIPPED (--skip-collection).")
    elif args.force_collect:
        log.info("Collection: RUNNING (--force-collect).")
        result = run_collection(data_dir)
        all_results.append(result)
    else:
        log.info("Collection: checking dataset completeness ...")
        complete = _collection_complete(data_dir)
        if complete:
            log.info(
                "Collection: SKIPPED — all towns have >= %d frames (95%% threshold).",
                _COLLECTION_THRESHOLD,
            )
        else:
            missing = [
                t for t in _TOWNS
                if not (data_dir / t).exists()
                or _count_town_frames(data_dir / t) < _COLLECTION_THRESHOLD
            ]
            log.info("Collection: RUNNING — incomplete towns: %s", missing)
            result = run_collection(data_dir)
            all_results.append(result)

    # -- Suite loop ------------------------------------------------------------
    for suite in suites:
        log.info("")
        log.info("=" * 70)
        log.info("Suite: %s", suite)
        log.info("=" * 70)
        suite_t0 = time.time()
        suite_failed = False

        # BC
        try:
            result = run_bc(suite, data_dir, models_dir, results_dir)
            all_results.append(result)
        except Exception:
            log.exception("[%s] BC failed — skipping this suite.", suite)
            suite_failures.append(f"{suite}/bc")
            suite_failed = True

        # PPO (only if BC succeeded)
        if not suite_failed:
            try:
                result = run_ppo(suite, models_dir, results_dir)
                all_results.append(result)
                any_ppo_ok = any(
                    not v.get("failed") and not v.get("skipped")
                    for v in result["styles"].values()
                )
                if not any_ppo_ok:
                    log.warning("[%s] All PPO styles failed — skipping eval.", suite)
                    suite_failed = True
            except Exception:
                log.exception("[%s] PPO stage failed — skipping eval.", suite)
                suite_failures.append(f"{suite}/ppo")
                suite_failed = True

        # Eval (only if at least one PPO checkpoint exists)
        if not suite_failed:
            ppo_exists = any(
                (models_dir / f"ppo_{style}_{suite}_best.pt").exists()
                for style in _PPO_STYLES
            )
            if not ppo_exists:
                log.warning("[%s] No PPO checkpoints found — skipping eval.", suite)
            else:
                try:
                    result = run_eval(suite, models_dir, results_dir)
                    all_results.append(result)
                except Exception:
                    log.exception("[%s] Eval failed.", suite)
                    suite_failures.append(f"{suite}/eval")

        suite_elapsed = time.time() - suite_t0
        log.info("[%s] Suite complete in %.1f min.", suite, suite_elapsed / 60)

    # -- Causal analysis -------------------------------------------------------
    try:
        result = run_causal(results_dir)
        all_results.append(result)
    except Exception:
        log.exception("Causal analysis failed.")
        suite_failures.append("causal")

    # -- Summary ---------------------------------------------------------------
    total_elapsed = time.time() - pipeline_t0
    summary = {
        "total_elapsed_min": round(total_elapsed / 60, 1),
        "suites_requested": suites,
        "failures": suite_failures,
        "stages": all_results,
    }
    summary_path = results_dir / "pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info("")
    log.info("=" * 70)
    log.info("Pipeline complete in %.1f min.", total_elapsed / 60)
    if suite_failures:
        log.warning("Failures: %s", suite_failures)
        log.info("Re-run failed stages with:")
        for f in suite_failures:
            parts = f.split("/")
            suite_arg = f"--suites {parts[0]}" if len(parts) > 1 else ""
            log.info("  py -3.11 scripts/run_pipeline.py %s --skip-collection", suite_arg)
    else:
        log.info("All stages passed.")
    log.info("Summary: %s", summary_path)
    log.info("=" * 70)

    sys.exit(1 if suite_failures else 0)


if __name__ == "__main__":
    main()
