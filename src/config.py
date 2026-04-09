"""
Shared configuration loader for the DriveNet project.

All YAML config files live in configs/ at the project root.
Agents and notebooks call load_config(name) to retrieve parameters.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIGS_DIR = _PROJECT_ROOT / "configs"


def load_config(name: str) -> dict[str, Any]:
    """Load ``configs/{name}.yaml`` and return its contents as a dict.

    Parameters
    ----------
    name : str
        Config file stem (e.g. ``"ppo"`` loads ``configs/ppo.yaml``).

    Returns
    -------
    dict[str, Any]
        Parsed YAML contents.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    """
    path = _CONFIGS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. "
            f"Expected configs/{name}.yaml in the project root."
        )
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    log.debug("Loaded config '%s' from %s", name, path)
    return cfg


def require_keys(
    cfg: dict[str, Any],
    keys: list[str],
    config_name: str,
) -> None:
    """Validate that *cfg* contains every key in *keys*.

    Parameters
    ----------
    cfg : dict
        Config dict returned by :func:`load_config`.
    keys : list[str]
        Required top-level keys.
    config_name : str
        Human-readable config name for error messages.

    Raises
    ------
    KeyError
        If one or more required keys are missing.
    """
    missing = [k for k in keys if k not in cfg]
    if missing:
        raise KeyError(
            f"Config '{config_name}' is missing required keys: {missing}. "
            f"Check configs/{config_name}.yaml."
        )
