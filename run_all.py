"""
Run all 5 DriveNet notebooks in sequence.

Usage:
    py -3.11 run_all.py              # run NB02-NB05 (skip NB01 since data exists)
    py -3.11 run_all.py --all        # run NB01-NB05 (re-collect data, takes hours)
    py -3.11 run_all.py --from 3     # start from NB03

Prerequisites:
    - CARLA server running for NB01, NB03, NB04
    - Python 3.11 with torch, carla, gymnasium, etc.
"""
import argparse
import json
import os
import sys
import time
import traceback

import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "notebooks")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "notebooks", "executed")

NOTEBOOKS = [
    ("01_data_collection.ipynb",   "Data Collection",   True),   # needs CARLA
    ("02_behavior_cloning.ipynb",  "Behavior Cloning",  False),
    ("03_ppo_finetuning.ipynb",    "PPO Fine-Tuning",   True),   # needs CARLA
    ("04_evaluation.ipynb",        "Evaluation",         True),   # needs CARLA
    ("05_causal_analysis.ipynb",   "Causal Analysis",    False),
]

# Per-notebook timeout in seconds.  NB01 and NB03 interact with CARLA for
# hundreds of thousands of simulation steps; NB04 runs 162 evaluation episodes.
TIMEOUTS = {
    "01_data_collection.ipynb":  8 * 3600,   # 8 hours
    "02_behavior_cloning.ipynb": 2 * 3600,   # 2 hours
    "03_ppo_finetuning.ipynb":   8 * 3600,   # 8 hours
    "04_evaluation.ipynb":       4 * 3600,   # 4 hours
    "05_causal_analysis.ipynb":    600,       # 10 minutes
}


def install_kernel():
    """Ensure the python3 kernel points to this interpreter."""
    import jupyter_client
    ksm = jupyter_client.kernelspec.KernelSpecManager()
    try:
        spec = ksm.get_kernel_spec("python3")
        argv0 = spec.argv[0]
        # If the kernel already points to our interpreter, nothing to do
        if os.path.normcase(os.path.abspath(argv0)) == os.path.normcase(sys.executable):
            return
    except Exception:
        pass

    # Overwrite with our interpreter
    print(f"[setup] Registering python3 kernel -> {sys.executable}")
    os.system(f'"{sys.executable}" -m ipykernel install --user --name python3 '
              f'--display-name "Python 3.11 (DriveNet)"')


def run_notebook(filename, label, timeout):
    """Execute a single notebook. Returns (success, elapsed_seconds, error_msg)."""
    nb_path = os.path.join(NOTEBOOKS_DIR, filename)
    out_path = os.path.join(OUTPUT_DIR, filename)

    print(f"\n{'=' * 70}")
    print(f"  [{label}] {filename}")
    print(f"  Timeout: {timeout // 3600}h {(timeout % 3600) // 60}m")
    print(f"{'=' * 70}")

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": NOTEBOOKS_DIR}},
    )

    t0 = time.time()
    try:
        client.execute()
        elapsed = time.time() - t0
        print(f"\n  PASSED in {elapsed / 60:.1f} min")

        # Save executed notebook with outputs
        with open(out_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        print(f"  Saved: {out_path}")

        return True, elapsed, None

    except Exception as exc:
        elapsed = time.time() - t0
        err_msg = str(exc)

        # Try to find the failing cell
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == "code" and hasattr(cell, "outputs"):
                for output in cell.outputs:
                    if output.get("output_type") == "error":
                        err_msg = (
                            f"Cell {i}: {output.get('ename', '?')}: "
                            f"{output.get('evalue', '?')}"
                        )
                        break

        print(f"\n  FAILED after {elapsed / 60:.1f} min")
        print(f"  Error: {err_msg}")

        # Save partial output even on failure
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                nbformat.write(nb, f)
            print(f"  Partial output saved: {out_path}")
        except Exception:
            pass

        return False, elapsed, err_msg


def main():
    parser = argparse.ArgumentParser(description="Run DriveNet notebooks in sequence")
    parser.add_argument("--all", action="store_true",
                        help="Include NB01 (data collection, takes hours)")
    parser.add_argument("--from", dest="start_from", type=int, default=None,
                        help="Start from notebook N (1-5)")
    parser.add_argument("--only", type=int, default=None,
                        help="Run only notebook N (1-5)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Determine which notebooks to run
    start = 0
    if args.start_from is not None:
        start = args.start_from - 1
    elif not args.all:
        # Default: skip NB01 since data already exists
        start = 1

    if args.only is not None:
        indices = [args.only - 1]
    else:
        indices = list(range(start, len(NOTEBOOKS)))

    # Pre-flight checks
    print("=" * 70)
    print("  DriveNet Notebook Runner")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Executable: {sys.executable}")
    print(f"  Project: {PROJECT_ROOT}")
    print("=" * 70)

    import torch
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()

    carla_needed = any(NOTEBOOKS[i][2] for i in indices)
    if carla_needed:
        try:
            import carla
            client = carla.Client("localhost", 2000)
            client.set_timeout(10.0)
            ver = client.get_server_version()
            print(f"  CARLA server: {ver}")
        except Exception as exc:
            print(f"  ERROR: Cannot reach CARLA at localhost:2000")
            print(f"  Launch CarlaUE4-Win64-Shipping.exe before running.")
            print(f"  Detail: {exc}")
            sys.exit(1)

    to_run = [(i, NOTEBOOKS[i]) for i in indices]
    print(f"\n  Notebooks to run: {', '.join(nb[0] for _, nb in to_run)}")
    total_timeout = sum(TIMEOUTS[nb[0]] for _, nb in to_run)
    print(f"  Max total time: {total_timeout / 3600:.1f} hours")
    print()

    # Ensure kernel points to this Python
    install_kernel()

    # Run
    results = []
    overall_t0 = time.time()

    for idx, (filename, label, needs_carla) in to_run:
        timeout = TIMEOUTS[filename]
        success, elapsed, err = run_notebook(filename, label, timeout)
        results.append((filename, success, elapsed, err))

        if not success:
            print(f"\n  Stopping: {filename} failed. Fix the error and re-run with:")
            print(f"    py -3.11 run_all.py --from {idx + 1}")
            break

    # Summary
    overall_elapsed = time.time() - overall_t0
    print(f"\n\n{'=' * 70}")
    print(f"  SUMMARY  (total: {overall_elapsed / 60:.1f} min)")
    print(f"{'=' * 70}")
    for filename, success, elapsed, err in results:
        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {filename:40s} {elapsed / 60:>7.1f} min"
              + (f"  -- {err[:60]}" if err else ""))
    print()

    all_passed = all(s for _, s, _, _ in results)
    if all_passed:
        print("  All notebooks passed.")
        print(f"  Executed outputs: {OUTPUT_DIR}")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
