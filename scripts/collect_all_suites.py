"""Collect expert data for all three sensor suites across all towns.

Loops through [single_cam, multi_cam, lidar] x [Town01..Town10HD] and calls
collect_one_town.py in a fresh subprocess for each (suite, town) pair.
Skips a pair if the target directory already has enough chunks (>= 7).
Retries up to MAX_RETRIES on non-zero exit.

Completion criteria: data/{suite}/{town}/chunk_*.npz with >= MIN_CHUNKS each.

Usage:
    py -3.11 scripts/collect_all_suites.py
    py -3.11 scripts/collect_all_suites.py --suites single_cam multi_cam
    py -3.11 scripts/collect_all_suites.py --towns Town01 Town02
    py -3.11 scripts/collect_all_suites.py --resume   # skip complete pairs
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_ALL_SUITES = ["single_cam", "multi_cam", "lidar"]
_ALL_TOWNS  = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD"]
_MIN_CHUNKS = 7       # 7 × 2400 = 16 800 frames > 95% of 16 200
_MAX_RETRIES = 3
_RETRY_SLEEP = 45.0   # seconds between retries

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "collect_all_suites.log",
                            mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger("collect_all_suites")


def _chunk_count(suite: str, town: str, data_dir: Path) -> int:
    d = data_dir / suite / town
    return len(list(d.glob("chunk_*.npz"))) if d.exists() else 0


def _kill_carla_ps() -> None:
    """Kill any running CARLA process via PowerShell Stop-Process.

    taskkill silently fails on RTX 5080 Blackwell; Stop-Process is reliable.
    """
    try:
        subprocess.run(
            [
                "powershell", "-NoProfile", "-Command",
                "Get-Process -Name 'CarlaUE4*' -ErrorAction SilentlyContinue"
                " | Stop-Process -Force",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def _collect(suite: str, town: str, data_dir: Path) -> bool:
    """Run collect_one_town.py for one (suite, town). Returns True on success."""
    # Belt-and-suspenders: ensure no stale CARLA before the subprocess starts
    log.info("[%s / %s] Killing any stale CARLA ...", suite, town)
    _kill_carla_ps()
    time.sleep(8.0)  # let the process exit and release port 2000

    script = PROJECT_ROOT / "scripts" / "collect_one_town.py"
    cmd = [
        sys.executable, str(script),
        "--town", town,
        "--data-dir", str(data_dir),
        "--sensor-suite", suite,
        "--startup-wait", "90",
        "--shutdown-wait", "20",
    ]
    log.info("[%s / %s] Running: %s", suite, town, " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    ok = proc.returncode in (0, 3)
    log.info("[%s / %s] Exit %d (%s)", suite, town, proc.returncode,
             "OK" if ok else "FAIL")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suites", nargs="+", default=_ALL_SUITES,
                        choices=_ALL_SUITES, metavar="SUITE")
    parser.add_argument("--towns", nargs="+", default=_ALL_TOWNS,
                        choices=_ALL_TOWNS, metavar="TOWN")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--resume", action="store_true",
                        help="Skip pairs that already have >= MIN_CHUNKS.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    (PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)

    pairs = [(s, t) for s in args.suites for t in args.towns]
    failed: list[tuple[str, str]] = []

    log.info("=" * 60)
    log.info("collect_all_suites: %d pairs to process", len(pairs))
    log.info("  Suites : %s", args.suites)
    log.info("  Towns  : %s", args.towns)
    log.info("  data   : %s", data_dir)
    log.info("=" * 60)

    for suite, town in pairs:
        label = f"{suite}/{town}"
        chunks = _chunk_count(suite, town, data_dir)

        if args.resume and chunks >= _MIN_CHUNKS:
            log.info("[%s] SKIP -- already %d chunks (>= %d).",
                     label, chunks, _MIN_CHUNKS)
            continue

        if chunks >= _MIN_CHUNKS:
            log.info("[%s] Already complete (%d chunks). Skipping.",
                     label, chunks)
            continue

        log.info("[%s] Starting collection (%d / %d chunks present).",
                 label, chunks, _MIN_CHUNKS)

        for attempt in range(1, _MAX_RETRIES + 1):
            if attempt > 1:
                log.warning("[%s] Retry %d/%d after %.0fs ...",
                            label, attempt, _MAX_RETRIES, _RETRY_SLEEP)
                time.sleep(_RETRY_SLEEP)
            _collect(suite, town, data_dir)
            # Check actual chunk count after each attempt — exit 0/3 alone is
            # not sufficient because CARLA can crash partway through, leaving
            # fewer than _MIN_CHUNKS even on a "successful" subprocess exit.
            if _chunk_count(suite, town, data_dir) >= _MIN_CHUNKS:
                break

        final_chunks = _chunk_count(suite, town, data_dir)
        if final_chunks >= _MIN_CHUNKS:
            log.info("[%s] Complete: %d chunks saved.", label, final_chunks)
        else:
            log.error("[%s] INCOMPLETE after %d attempts: only %d / %d chunks.",
                      label, attempt, final_chunks, _MIN_CHUNKS)
            failed.append((suite, town))

    log.info("=" * 60)
    if failed:
        log.error("FAILED pairs: %s", failed)
        log.info("Re-run: py -3.11 scripts/collect_all_suites.py --resume")
        sys.exit(1)
    else:
        log.info("All pairs complete.")
        sys.exit(0)


if __name__ == "__main__":
    main()
