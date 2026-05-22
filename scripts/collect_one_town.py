"""Collect data for a single town in an isolated subprocess.

NB01 invokes this once per town so each collection runs in a fresh Python
interpreter.  This is required because libcarla's Boost.Asio streaming
threads outlive ``del client`` -- when a long-lived parent process drives
multiple towns, stale background threads from a prior town abort the next
town's load_world().  Subprocess isolation lets the OS reap those threads
on exit.

Usage:
    python scripts/collect_one_town.py --town Town01 [--data-dir data]
                                       [--frames-per-condition N]
                                       [--no-follow-car]

Exits 0 on success.  A non-zero exit is treated as an error by the caller,
but chunks already saved to disk remain valid -- the validation cells in
NB01 verify completeness independently of the subprocess return code.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_ALL_TOWNS = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--town", required=True, choices=_ALL_TOWNS)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--frames-per-condition", type=int, default=None,
                        help="Override frames_per_condition for smoke runs.")
    parser.add_argument("--no-follow-car", action="store_true",
                        help="Disable the follow car (faster smoke runs).")
    parser.add_argument("--startup-wait", type=float, default=60.0)
    parser.add_argument("--shutdown-wait", type=float, default=10.0)
    parser.add_argument("--viz", action="store_true",
                        help="Open live Tesla/Waymo-style visualization window.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        stream=sys.stdout,
    )

    from src.agents.collection_agent import DataCollectionAgent

    agent = DataCollectionAgent(
        town=args.town,
        data_dir=args.data_dir,
        frames_per_condition=args.frames_per_condition,
        enable_follow_car=not args.no_follow_car,
        enable_viz=args.viz,
    )

    try:
        results = agent.run_all_towns(
            towns=[args.town],
            startup_wait=args.startup_wait,
            shutdown_wait=args.shutdown_wait,
        )
    except Exception:
        logging.exception("Collection failed for %s", args.town)
        return 2

    chunks = results.get(args.town, 0)
    logging.info("Done -- %s saved %d chunks.", args.town, chunks)
    sys.stdout.flush()

    # Bypass interpreter shutdown so libcarla's lingering Boost.Asio threads
    # don't trigger a Windows Fast-Fail during atexit.  The data is already
    # on disk at this point.
    os._exit(0 if chunks > 0 else 3)


if __name__ == "__main__":
    sys.exit(main())
