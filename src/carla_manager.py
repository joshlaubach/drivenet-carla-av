"""Standalone CARLA server lifecycle manager.

Use as a context manager for automatic start/stop:

    with CARLAManager() as mgr:
        env = CarlaEnv(host=mgr.host, port=mgr.port, ...)

Or manually:

    mgr = CARLAManager()
    mgr.start()
    mgr.wait_ready()
    ...
    mgr.stop()

CARLAManager is intentionally decoupled from any training agent so it can be
used from scripts, notebooks, or future agents without pulling in PPO/BC deps.
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import time
from pathlib import Path

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CARLA_EXE = _PROJECT_ROOT / "CARLA_0.9.16" / "CarlaUE4.exe"
_STRAY_EXES = ("CarlaUE4-Win64-Shipping.exe", "CarlaUE4.exe")


class CARLAManager:
    """Manages the CARLA server process lifecycle.

    Parameters
    ----------
    exe_path : path-like, optional
        Path to CarlaUE4.exe. Defaults to CARLA_0.9.16/CarlaUE4.exe.
    host : str
        CARLA server host (default localhost).
    port : int
        CARLA server port (default 2000).
    quality : str
        Render quality level passed to -quality-level= flag (default Low).
    """

    def __init__(
        self,
        exe_path: str | Path | None = None,
        host: str = "localhost",
        port: int = 2000,
        quality: str = "Low",
    ) -> None:
        self.exe_path = Path(exe_path) if exe_path else _DEFAULT_CARLA_EXE
        self.host = host
        self.port = port
        self.quality = quality
        self._proc: subprocess.Popen | None = None

    # -- Lifecycle -------------------------------------------------------------

    def start(self) -> subprocess.Popen:
        """Launch a fresh CARLA server. Returns the Popen handle."""
        if not self.exe_path.exists():
            raise FileNotFoundError(
                f"CARLA executable not found: {self.exe_path}. "
                "Install CARLA 0.9.16 or pass exe_path= explicitly."
            )
        cmd = [
            str(self.exe_path),
            "-dx12",
            f"-quality-level={self.quality}",
            "-fps=20",
            "-benchmark",
            "-windowed",
            "-ResX=800",
            "-ResY=600",
            "-nosound",
            "-NoSplash",
        ]
        env_vars = dict(os.environ)
        env_vars["DXGI_GPU_PREFERENCE"] = "2"
        self._proc = subprocess.Popen(
            cmd,
            env=env_vars,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log.info("CARLA started (PID %d).", self._proc.pid)
        return self._proc

    def stop(self, proc: subprocess.Popen | None = None) -> None:
        """Terminate the managed process (or a given proc) and kill strays."""
        target = proc or self._proc
        if target is not None:
            try:
                target.terminate()
                target.wait(timeout=10)
                log.info("Terminated CARLA PID %d.", target.pid)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    target.kill()
                except OSError:
                    pass
        for exe in _STRAY_EXES:
            try:
                subprocess.run(
                    ["taskkill", "/F", "/IM", exe],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError:
                pass
        self._proc = None

    def restart(self, sleep_after: float = 6.0) -> subprocess.Popen:
        """Stop any running instance, sleep for cleanup, then start fresh."""
        self.stop()
        time.sleep(sleep_after)
        return self.start()

    def wait_ready(
        self,
        max_wait: float = 60.0,
        poll_interval: float = 3.0,
        settle: float = 5.0,
    ) -> None:
        """Block until CARLA accepts TCP connections on self.port.

        Parameters
        ----------
        max_wait : float
            Total seconds to wait before raising TimeoutError.
        poll_interval : float
            Seconds between connection attempts.
        settle : float
            Extra sleep after the port opens to let CARLA finish initializing.
        """
        deadline = time.time() + max_wait
        while time.time() < deadline:
            try:
                with socket.create_connection((self.host, self.port), timeout=2.0):
                    log.info("CARLA reachable on %s:%d.", self.host, self.port)
                    time.sleep(settle)
                    return
            except (ConnectionRefusedError, OSError):
                time.sleep(poll_interval)
        raise TimeoutError(
            f"CARLA not reachable on {self.host}:{self.port} "
            f"within {max_wait:.0f}s."
        )

    def is_alive(self) -> bool:
        """Return True if the managed process is still running."""
        return self._proc is not None and self._proc.poll() is None

    # -- Context manager -------------------------------------------------------

    def __enter__(self) -> "CARLAManager":
        self.start()
        self.wait_ready()
        return self

    def __exit__(self, *_) -> None:
        self.stop()
