"""CARLA helper functions for weather, environment creation, and process lifecycle."""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import time
from pathlib import Path

import carla

from src.carla_env import CarlaEnv

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Process lifecycle
# ---------------------------------------------------------------------------

_DEFAULT_CARLA_EXE = Path(__file__).resolve().parent.parent / "CARLA_0.9.16" / "CarlaUE4.exe"


def launch_carla(
    carla_exe: Path | str | None = None,
) -> subprocess.Popen:
    """Launch a fresh CARLA server process and return the Popen handle.

    Uses -dx12 and DXGI_GPU_PREFERENCE=2 for RTX 5080 Blackwell stability.
    The caller is responsible for calling wait_for_carla() before connecting
    and kill_carla() after use.
    """
    exe = Path(carla_exe) if carla_exe is not None else _DEFAULT_CARLA_EXE
    if not exe.exists():
        raise FileNotFoundError(
            f"CARLA executable not found: {exe}. "
            "Place CARLA at CARLA_0.9.16/ or pass carla_exe= explicitly."
        )
    cmd = [
        str(exe),
        "-dx12", "-quality-level=Low", "-fps=20",
        "-benchmark", "-windowed", "-ResX=800", "-ResY=600",
        "-nosound", "-NoSplash",
    ]
    env_vars = dict(os.environ)
    env_vars["DXGI_GPU_PREFERENCE"] = "2"
    proc = subprocess.Popen(cmd, env=env_vars, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log.info("CARLA process started (PID %d).", proc.pid)
    return proc


def wait_for_carla(
    host: str = "localhost",
    port: int = 2000,
    max_wait: float = 60.0,
    poll_interval: float = 3.0,
    post_connect_sleep: float = 15.0,
) -> None:
    """Block until CARLA accepts TCP connections on host:port.

    post_connect_sleep gives the server time to finish world initialization
    after the TCP port becomes reachable (15 s is load-bearing on Blackwell).
    """
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2.0):
                log.info("CARLA is reachable on %s:%d.", host, port)
                time.sleep(post_connect_sleep)
                return
        except (ConnectionRefusedError, OSError):
            time.sleep(poll_interval)
    raise TimeoutError(
        f"CARLA did not become reachable on {host}:{port} within {max_wait:.0f}s."
    )


def kill_carla(proc: subprocess.Popen | None = None) -> None:
    """Terminate a CARLA process and sweep any stray CarlaUE4 instances.

    Uses PowerShell Stop-Process rather than taskkill /F /IM — the latter
    silently fails on DX12-protected handles and leaves zombies that conflict
    with the next CARLA launch on RTX 5080 Blackwell.
    """
    if proc is not None:
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except (subprocess.TimeoutExpired, OSError):
            proc.kill()
        log.info("Terminated CARLA PID %d.", proc.pid)
    try:
        subprocess.run(
            [
                "powershell", "-NonInteractive", "-Command",
                "Get-Process -Name 'CarlaUE4*' -ErrorAction SilentlyContinue"
                " | Stop-Process -Force -ErrorAction SilentlyContinue",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
    except Exception:
        pass


def make_weather(env: CarlaEnv, weather_name: str) -> None:
    """Apply a named weather preset to an existing CarlaEnv.

    Parameters
    ----------
    env : CarlaEnv
        Active environment instance.
    weather_name : str
        Name of a ``carla.WeatherParameters`` preset (e.g. ``"ClearNoon"``).
    """
    env.world.set_weather(getattr(carla.WeatherParameters, weather_name))


def make_env(
    town: str,
    weather_name: str,
    host: str = "localhost",
    port: int = 2000,
) -> CarlaEnv:
    """Create a CarlaEnv for *town* and set weather.

    Parameters
    ----------
    town : str
        CARLA town name (e.g. ``"Town03"``).
    weather_name : str
        Weather preset name.
    host, port : str, int
        CARLA server address.

    Returns
    -------
    CarlaEnv
    """
    env = CarlaEnv(host=host, port=port, town=town)
    make_weather(env, weather_name)
    return env
