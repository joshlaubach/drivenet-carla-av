#!/usr/bin/env python3
"""
CARLA startup diagnostic matrix for Town01 pre-collection crash triage.

This script runs startup-only and light sensor profiles repeatedly to isolate
where instability occurs before data collection begins.

What it measures per cycle:
- Launch success and process liveness
- TCP readiness and RPC readiness timing
- Optional map load behavior and verification retries
- Optional sync mode + light sensor callback load
- Graceful teardown success

Outputs:
- Console summary table
- JSON report in results/startup_diagnostics/
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from queue import Empty, Queue
from typing import Any

import psutil

CARLA_HOST = "localhost"
CARLA_PORT = 2000
DEFAULT_CARLA_EXE = r"C:\Users\joshu\Documents\Projects\CARLA AV\CARLA_0.9.16\CarlaUE4.exe"


@dataclass
class StartupProfile:
    name: str
    rhi_flag: str
    quality_level: str = "Low"
    fps: int = 20
    benchmark: bool = True
    windowed: bool = True
    res_x: int = 400
    res_y: int = 300
    nosound: bool = True
    startup_timeout_seconds: float = 120.0
    tcp_poll_seconds: float = 0.5
    rpc_retry_seconds: float = 1.0
    tcp_to_rpc_grace_seconds: float = 2.0
    map_strategy: str = "load_world"  # load_world | verify_current
    map_load_timeout_seconds: float = 120.0
    map_grace_seconds: float = 3.0
    map_verify_retries: int = 2
    target_town: str = "Town01"
    enable_sync_mode: bool = True
    fixed_delta_seconds: float = 0.05
    enable_sensor: bool = False
    sensor_ticks: int = 60
    camera_width: int = 320
    camera_height: int = 240
    queue_maxsize: int = 8
    destroy_timeout_seconds: float = 6.0
    teardown_sleep_seconds: float = 1.5


BASE_PROFILES: list[StartupProfile] = [
    StartupProfile(
        name="dx11_startup_only",
        rhi_flag="-dx11",
        map_strategy="load_world",
        enable_sync_mode=False,
        enable_sensor=False,
    ),
    StartupProfile(
        name="dx11_sync_only",
        rhi_flag="-dx11",
        map_strategy="load_world",
        enable_sync_mode=True,
        enable_sensor=False,
        fixed_delta_seconds=0.05,
    ),
    StartupProfile(
        name="dx11_sensor_light",
        rhi_flag="-dx11",
        map_strategy="load_world",
        enable_sync_mode=True,
        enable_sensor=True,
        sensor_ticks=120,
        camera_width=320,
        camera_height=240,
        queue_maxsize=12,
    ),
    StartupProfile(
        name="dx11_sensor_stress",
        rhi_flag="-dx11",
        map_strategy="load_world",
        enable_sync_mode=True,
        enable_sensor=True,
        sensor_ticks=240,
        camera_width=800,
        camera_height=600,
        queue_maxsize=32,
    ),
]

VULKAN_CONTROL_PROFILE = StartupProfile(
    name="vulkan_control",
    rhi_flag="-vulkan",
    map_strategy="load_world",
    enable_sync_mode=True,
    enable_sensor=False,
    sensor_ticks=0,
)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{_now()}] {message}", flush=True)


def port_accepting(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False


def kill_all_carla() -> None:
    candidates = [
        p
        for p in psutil.process_iter(["pid", "name"])
        if "CarlaUE4" in (p.info.get("name") or "")
    ]
    for proc in candidates:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(candidates, timeout=20)

    deadline = time.time() + 20
    while time.time() < deadline and port_accepting(CARLA_HOST, CARLA_PORT, timeout=0.25):
        time.sleep(0.25)


def build_launch_command(carla_exe: str, profile: StartupProfile) -> list[str]:
    cmd = [carla_exe, profile.rhi_flag]
    cmd.extend(["-quality-level=" + profile.quality_level, f"-fps={profile.fps}"])
    if profile.benchmark:
        cmd.append("-benchmark")
    if profile.windowed:
        cmd.extend(["-windowed", f"-ResX={profile.res_x}", f"-ResY={profile.res_y}"])
    if profile.nosound:
        cmd.append("-nosound")
    return cmd


def wait_for_server_ready(profile: StartupProfile):
    import carla

    t0 = time.time()
    deadline = t0 + profile.startup_timeout_seconds
    saw_tcp_open = False
    tcp_open_ts = None

    while time.time() < deadline:
        if not port_accepting(CARLA_HOST, CARLA_PORT, timeout=0.5):
            time.sleep(profile.tcp_poll_seconds)
            continue

        if not saw_tcp_open:
            saw_tcp_open = True
            tcp_open_ts = time.time()
            time.sleep(profile.tcp_to_rpc_grace_seconds)

        try:
            client = carla.Client(CARLA_HOST, CARLA_PORT)
            client.set_timeout(10.0)
            version = client.get_server_version()
            return client, version, t0, tcp_open_ts
        except Exception:
            time.sleep(profile.rpc_retry_seconds)

    raise RuntimeError(
        f"Server not RPC-ready within {profile.startup_timeout_seconds:.1f}s"
    )


def short_map_name(full_map_name: str) -> str:
    return full_map_name.split("/")[-1].replace("_Opt", "")


def load_or_verify_map(client, profile: StartupProfile) -> str:
    target = profile.target_town

    if profile.map_strategy == "load_world":
        client.set_timeout(profile.map_load_timeout_seconds)
        client.load_world(target)
        time.sleep(profile.map_grace_seconds)

    client.set_timeout(30.0)
    last_seen = "<unknown>"
    for _ in range(max(1, profile.map_verify_retries) + 1):
        world = client.get_world()
        loaded = short_map_name(world.get_map().name)
        last_seen = loaded
        if loaded == target:
            return loaded
        time.sleep(2.0)

    raise RuntimeError(
        f"Map verification failed: expected '{target}', got '{last_seen}'"
    )


def run_sync_sensor_probe(client, profile: StartupProfile) -> dict[str, Any]:
    import carla

    probe_result: dict[str, Any] = {
        "sync_enabled": False,
        "sensor_enabled": False,
        "ticks": 0,
        "frames_seen": 0,
        "cleanup_warnings": [],
    }

    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = bool(profile.enable_sync_mode)
    settings.fixed_delta_seconds = (
        float(profile.fixed_delta_seconds) if profile.enable_sync_mode else 0.0
    )
    world.apply_settings(settings)
    probe_result["sync_enabled"] = bool(profile.enable_sync_mode)

    if not profile.enable_sensor:
        if profile.enable_sync_mode:
            for _ in range(10):
                world.tick()
            probe_result["ticks"] = 10
        return probe_result

    probe_result["sensor_enabled"] = True
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points available for sensor probe")

    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[0])
    if vehicle is None:
        raise RuntimeError("Failed to spawn probe vehicle")

    camera = None
    q: Queue[Any] = Queue(maxsize=profile.queue_maxsize)
    try:
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(profile.camera_width))
        cam_bp.set_attribute("image_size_y", str(profile.camera_height))
        camera = world.spawn_actor(
            cam_bp,
            carla.Transform(carla.Location(x=1.5, z=2.4)),
            attach_to=vehicle,
        )
        camera.listen(lambda image: q.put(image) if not q.full() else None)

        if profile.enable_sync_mode:
            for _ in range(profile.sensor_ticks):
                world.tick()
                probe_result["ticks"] += 1
        else:
            time.sleep(max(1.0, profile.sensor_ticks * 0.02))

        probe_result["frames_seen"] = q.qsize()
    finally:
        try:
            if camera is not None:
                camera.stop()
        except Exception:
            pass

        if camera is not None:
            ok, err = destroy_actor_with_timeout(camera, profile.destroy_timeout_seconds)
            if not ok and err:
                probe_result["cleanup_warnings"].append(f"camera: {err}")

        if vehicle is not None:
            ok, err = destroy_actor_with_timeout(vehicle, profile.destroy_timeout_seconds)
            if not ok and err:
                probe_result["cleanup_warnings"].append(f"vehicle: {err}")

        while not q.empty():
            try:
                q.get_nowait()
            except Empty:
                break

    return probe_result


def destroy_actor_with_timeout(actor, timeout_seconds: float) -> tuple[bool, str | None]:
    """Best-effort actor destroy that cannot block the main diagnostics loop."""
    state: dict[str, Any] = {"done": False, "error": None}

    def _do_destroy() -> None:
        try:
            actor.destroy()
            state["done"] = True
        except Exception as exc:  # pragma: no cover - depends on simulator state
            state["error"] = f"{type(exc).__name__}: {exc}"

    thread = threading.Thread(target=_do_destroy, daemon=True)
    thread.start()
    thread.join(timeout=max(0.1, float(timeout_seconds)))

    if thread.is_alive():
        return False, f"destroy timed out after {timeout_seconds:.1f}s"
    if state["error"] is not None:
        return False, str(state["error"])
    return True, None


def safe_teardown(client, profile: StartupProfile) -> None:
    world = None
    try:
        world = client.get_world()
    except Exception:
        pass

    if world is not None:
        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = 0.0
            world.apply_settings(settings)
        except Exception:
            pass

    del client
    time.sleep(profile.teardown_sleep_seconds)


def run_cycle(carla_exe: str, profile: StartupProfile, cycle_id: int) -> dict[str, Any]:
    started = time.time()
    result: dict[str, Any] = {
        "profile": profile.name,
        "cycle": cycle_id,
        "status": "pass",
        "stage": "start",
        "duration_seconds": None,
        "server_version": None,
        "loaded_map": None,
        "tcp_open_seconds": None,
        "error": None,
        "probe": None,
        "launch_command": build_launch_command(carla_exe, profile),
    }

    proc = None
    client = None

    try:
        kill_all_carla()

        cmd = build_launch_command(carla_exe, profile)
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        result["stage"] = "launch"

        client, version, t0, tcp_open_ts = wait_for_server_ready(profile)
        result["stage"] = "connected"
        result["server_version"] = version
        if tcp_open_ts is not None:
            result["tcp_open_seconds"] = round(tcp_open_ts - t0, 3)

        result["loaded_map"] = load_or_verify_map(client, profile)
        result["stage"] = "map_verified"

        result["probe"] = run_sync_sensor_probe(client, profile)
        result["stage"] = "probe_done"

    except Exception as exc:
        result["status"] = "fail"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if client is not None:
            try:
                safe_teardown(client, profile)
            except Exception as exc:
                if result["status"] == "pass":
                    result["status"] = "fail"
                    result["error"] = f"TeardownError: {exc}"

        if proc is not None:
            try:
                proc.kill()
                proc.wait(timeout=20)
            except Exception:
                pass

        kill_all_carla()
        result["duration_seconds"] = round(time.time() - started, 3)

    return result


def summarize(results: list[dict[str, Any]]) -> None:
    log("\nSummary")
    log("profile                cycle   status   stage         map        tcp_s   dur_s")
    for row in results:
        profile = row["profile"][:20].ljust(20)
        cycle = str(row["cycle"]).rjust(5)
        status = row["status"].ljust(7)
        stage = (row.get("stage") or "-")[:12].ljust(12)
        loaded_map = (row.get("loaded_map") or "-")[:10].ljust(10)
        tcp_s = str(row.get("tcp_open_seconds") if row.get("tcp_open_seconds") is not None else "-").rjust(6)
        dur_s = str(row.get("duration_seconds") if row.get("duration_seconds") is not None else "-").rjust(6)
        log(f"{profile} {cycle}   {status}  {stage} {loaded_map} {tcp_s} {dur_s}")

    failures = [r for r in results if r["status"] != "pass"]
    if failures:
        log("\nFailures")
        for row in failures:
            log(
                f"- {row['profile']} cycle {row['cycle']} failed at {row.get('stage')}: {row.get('error')}"
            )


def save_report(results: list[dict[str, Any]], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"startup_diagnostic_report_{timestamp}.json")
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "host": CARLA_HOST,
        "port": CARLA_PORT,
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CARLA startup diagnostic matrix for Town01 crash triage"
    )
    parser.add_argument(
        "--carla-exe",
        default=DEFAULT_CARLA_EXE,
        help="Path to CarlaUE4 executable",
    )
    parser.add_argument(
        "--town",
        default="Town01",
        help="Target town for load/verify",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Cycles per profile",
    )
    parser.add_argument(
        "--include-vulkan",
        action="store_true",
        help="Include a Vulkan control profile for comparison",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("results", "startup_diagnostics"),
        help="Directory for JSON report",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not os.path.exists(args.carla_exe):
        log(f"ERROR: Carla executable not found: {args.carla_exe}")
        return 2

    profiles = [
        StartupProfile(**{**asdict(p), "target_town": args.town}) for p in BASE_PROFILES
    ]
    if args.include_vulkan:
        profiles.append(StartupProfile(**{**asdict(VULKAN_CONTROL_PROFILE), "target_town": args.town}))

    log("CARLA Startup Diagnostic Matrix")
    log(f"Executable: {args.carla_exe}")
    log(f"Target town: {args.town}")
    log(f"Repeat count: {args.repeat}")
    log(f"Profiles: {', '.join(p.name for p in profiles)}")

    results: list[dict[str, Any]] = []
    for profile in profiles:
        for cycle in range(1, args.repeat + 1):
            log(f"\n[RUN] profile={profile.name} cycle={cycle}/{args.repeat}")
            cycle_result = run_cycle(args.carla_exe, profile, cycle)
            results.append(cycle_result)
            status = cycle_result["status"].upper()
            log(
                f"[DONE] profile={profile.name} cycle={cycle} status={status} stage={cycle_result.get('stage')}"
            )

    summarize(results)
    report_path = save_report(results, args.output_dir)
    log(f"\nSaved report: {report_path}")

    failed = any(r["status"] != "pass" for r in results)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
