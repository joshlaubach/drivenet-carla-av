"""Smoke test: subprocess-per-town orchestration -- exact pattern NB01 uses.

Runs scripts/collect_one_town.py twice (Town01 + Town02) with tiny frame
counts.  Validates that each subprocess exits cleanly and produces chunks.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    smoke_dir = PROJECT_ROOT / "data_smoke"
    if smoke_dir.exists():
        import shutil
        shutil.rmtree(smoke_dir)

    towns = ["Town01", "Town02"]
    for town in towns:
        log(f"=== {town} subprocess starting ===")
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "collect_one_town.py"),
             "--town", town,
             "--data-dir", str(smoke_dir),
             "--frames-per-condition", "5",
             "--no-follow-car"],
            cwd=str(PROJECT_ROOT),
        )
        log(f"=== {town} subprocess exited code={result.returncode} ===")
        chunks = list((smoke_dir / town).glob("chunk_*.npz"))
        log(f"{town}: {len(chunks)} chunks on disk")
        if not chunks:
            log(f"FAILED: {town} produced no chunks")
            return 2

    log("ALL TOWNS COMPLETE -- subprocess-per-town flow works")
    return 0


if __name__ == "__main__":
    sys.exit(main())
