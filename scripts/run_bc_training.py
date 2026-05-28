"""Run BehaviorCloningAgent for all three sensor suites sequentially.

Usage:
    python scripts/run_bc_training.py
    python scripts/run_bc_training.py --suites single_cam multi_cam lidar
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.training_agent import BehaviorCloningAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s -- %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "logs" / "bc_training.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

SUITES = ["single_cam", "multi_cam", "lidar"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suites", nargs="+", default=SUITES, choices=SUITES)
    args = parser.parse_args()

    all_metrics: dict[str, dict] = {}
    for suite in args.suites:
        log.info("=" * 60)
        log.info("Starting BC training: suite=%s", suite)
        log.info("=" * 60)
        try:
            agent = BehaviorCloningAgent(
                sensor_suite=suite,
                data_dir=str(ROOT / "data" / suite),
                save_dir=str(ROOT / "models"),
                results_dir=str(ROOT / "results"),
            )
            metrics = agent.run()
            all_metrics[suite] = metrics
            log.info("Finished %s: %s", suite, metrics)
        except Exception:
            log.exception("Suite %s FAILED", suite)
            all_metrics[suite] = {"error": "FAILED"}

    summary_path = ROOT / "results" / "bc_all_metrics.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    log.info("All suites done. Summary -> %s", summary_path)

    for suite, m in all_metrics.items():
        if "error" in m:
            log.error("  %s: FAILED", suite)
        else:
            log.info("  %s: test_loss=%.6f  steer=%.4f  throttle=%.4f  brake=%.4f",
                     suite, m["test_loss"], m["test_steer_mse"],
                     m["test_throttle_mse"], m["test_brake_mse"])


if __name__ == "__main__":
    main()
