"""One-time migration: normalise legacy chunk NPZ files to the current schema.

Chunks collected before the schema stabilised may be missing fields or have
under-dimensioned state vectors.  This script patches them in-place:

  - states: zero-pad to STATE_DIM (6) if shorter
  - style:  add column of "standard" strings if missing (expert autopilot data
            is always collected in standard style)

Run once before BC training.  Safe to re-run; already-correct chunks are
skipped after a shape check.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

STATE_DIM = 6
STYLE_DEFAULT = "standard"


def migrate_chunk(path: Path, dry_run: bool = False) -> str:
    """Return a status string: 'ok', 'patched', or 'error:<msg>'."""
    try:
        with np.load(path, allow_pickle=True) as f:
            data = {k: f[k] for k in f.files}
    except Exception as exc:
        return f"error:{exc}"

    changed = False

    # -- Fix state dimensions --------------------------------------------------
    states = data["states"]
    if states.shape[1] < STATE_DIM:
        pad = np.zeros((states.shape[0], STATE_DIM - states.shape[1]), dtype=np.float32)
        data["states"] = np.concatenate([states, pad], axis=1)
        changed = True
    elif states.shape[1] > STATE_DIM:
        data["states"] = states[:, :STATE_DIM]
        changed = True

    # -- Add missing style field -----------------------------------------------
    if "style" not in data:
        data["style"] = np.full(states.shape[0], STYLE_DEFAULT)
        changed = True

    if not changed:
        return "ok"

    if dry_run:
        return "would-patch"

    try:
        np.savez_compressed(path, **data)
    except Exception as exc:
        return f"error:{exc}"

    return "patched"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data",
                        help="Root data directory (default: data/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    chunks = sorted(data_dir.rglob("chunk_*.npz"))
    if not chunks:
        print(f"No chunk files found under {data_dir}.")
        return 1

    counts = {"ok": 0, "patched": 0, "would-patch": 0, "error": 0}
    for path in chunks:
        status = migrate_chunk(path, dry_run=args.dry_run)
        key = status.split(":")[0]
        counts[key] = counts.get(key, 0) + 1
        if status != "ok":
            print(f"  {status:14s}  {path.relative_to(data_dir)}")

    print(f"\nDone. {len(chunks)} chunks: "
          f"{counts['ok']} ok, "
          f"{counts.get('patched', 0) + counts.get('would-patch', 0)} patched, "
          f"{counts.get('error', 0)} errors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
