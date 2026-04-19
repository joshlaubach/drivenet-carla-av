#!/usr/bin/env python3
"""
CARLA libcarla Fast-Fail Crash Diagnostic  (exit 0xC0000409 = 3221226505)
=========================================================================
Isolates the exact cause of the Python kernel crash in restart_carla_and_reconnect().

Scenarios 1-2: safe (run in this process, CARLA is OFF).
Scenarios 3-7: dangerous (run in a subprocess; a crash only kills the child).

The test runner detects exit code 0xC0000409 and labels the scenario as the
confirmed root cause.

Usage
-----
    python tests/test_crash_scenarios.py            # run all 7 in order
    python tests/test_crash_scenarios.py -s 6       # run one scenario
    python tests/test_crash_scenarios.py --run 6    # internal: direct execution (subprocess use)

Results are printed to stdout and appended to crash_diagnostic.log.
"""

import argparse
import os
import socket as _socket
import subprocess
import sys
import time

import psutil

# --- Config -------------------------------------------------------------------

CARLA_HOST = "localhost"
CARLA_PORT = 2000
CARLA_EXE = r"C:\Users\joshu\Documents\Projects\CARLA AV\CARLA_0.9.16\CarlaUE4.exe"
TEST_MAP_PATH = "/Game/Carla/Maps/Town01"
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crash_diagnostic.log")

CRASH_EXIT = 3221226505  # 0xC0000409 STATUS_STACK_BUFFER_OVERRUN / Windows Fast-Fail


# --- Utilities ----------------------------------------------------------------

def log(msg):
    ts = time.strftime("%H:%M:%S.") + f"{int(time.time() * 1000) % 1000:03d}"
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def port_accepting(timeout=1.0):
    """Raw TCP probe -- no libcarla involved."""
    try:
        _socket.create_connection((CARLA_HOST, CARLA_PORT), timeout=timeout).close()
        return True
    except OSError:
        return False


def kill_all_carla():
    procs = [p for p in psutil.process_iter(["name"])
             if "CarlaUE4" in p.info.get("name", "")]
    for p in procs:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(procs, timeout=15)
    deadline = time.time() + 15
    while time.time() < deadline and port_accepting():
        time.sleep(1)
    time.sleep(2)


def start_carla():
    return subprocess.Popen(
        [CARLA_EXE, TEST_MAP_PATH, "-dx11", "-windowed", "-ResX=400", "-ResY=300"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# --- Scenarios ----------------------------------------------------------------

def scenario_1():
    """
    HYPOTHESIS: carla.Client() itself crashes when the port is CLOSED.

    If TRUE:  even a single client creation against a closed port kills the process.
    If FALSE: RuntimeError is raised cleanly -- the Client constructor is safe.

    CARLA must NOT be running. Safe to run in-process.
    """
    import carla
    log("S1 | Ensuring CARLA is not running...")
    kill_all_carla()
    assert not port_accepting(0.5), "S1 ABORTED: CARLA is still running"

    log("S1 | Creating carla.Client() against CLOSED port...")
    _c = carla.Client(CARLA_HOST, CARLA_PORT)
    _c.set_timeout(3.0)
    log("S1 | Calling get_server_version()...")
    try:
        ver = _c.get_server_version()
        log(f"S1 | UNEXPECTED SUCCESS -- version={ver!r}")
    except RuntimeError as e:
        log(f"S1 | PASS -- RuntimeError as expected: {e}")
    except Exception as e:
        log(f"S1 | UNEXPECTED {type(e).__name__}: {e}")
    del _c
    log("S1 | DONE -- constructor is safe against a closed port")


def scenario_2():
    """
    HYPOTHESIS: Accumulating carla.Client() objects (never del'd) exhausts
    Boost.Asio thread-pool resources and eventually triggers Fast-Fail.

    Creates 30 clients without releasing any, calling get_server_version() on each.
    If TRUE:  crash at some iteration number -- resource exhaustion confirmed.
    If FALSE: all 30 complete -- accumulation alone doesn't cause the crash.

    CARLA must NOT be running. Safe to run in-process.
    """
    import carla
    log("S2 | Ensuring CARLA is not running...")
    kill_all_carla()
    assert not port_accepting(0.5), "S2 ABORTED: CARLA is still running"

    accumulated = []
    for i in range(30):
        log(f"S2 | Creating client {i + 1}/30 (NOT releasing previous)...")
        _c = carla.Client(CARLA_HOST, CARLA_PORT)
        _c.set_timeout(1.0)
        try:
            _c.get_server_version()
        except Exception:
            pass
        accumulated.append(_c)

    log("S2 | All 30 clients created without crash -- releasing...")
    del accumulated
    log("S2 | DONE -- accumulation alone does NOT crash")


def scenario_3():
    """
    HYPOTHESIS: The original polling loop (create carla.Client() per iteration,
    no TCP probe, no explicit del) crashes because libcarla's background
    Boost.Asio thread hits a connection error before the port is open or right
    as it transitions CLOSED -> OPEN.

    Mirrors the original code exactly.
    If TRUE:  exits with code 3221226505 -- root cause confirmed.
    If FALSE: connects successfully -- crash had a different trigger.

    Starts CARLA. Run via subprocess.
    """
    import carla
    log("S3 | Killing any running CARLA...")
    kill_all_carla()

    log("S3 | Starting CARLA with Town01...")
    _proc = start_carla()
    t0 = time.time()

    log("S3 | Starting ORIGINAL polling loop (no TCP probe, no del)...")
    deadline = time.time() + 120
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        log(f"S3 | Attempt {attempt:03d}  t+{time.time()-t0:.1f}s | carla.Client()...")
        _c = carla.Client(CARLA_HOST, CARLA_PORT)
        _c.set_timeout(5.0)
        try:
            ver = _c.get_server_version()
            log(f"S3 | CONNECTED on attempt {attempt}: {ver}")
            del _c
            break
        except RuntimeError as e:
            # Deliberately NOT calling del _c -- mirrors original bug
            log(f"S3 | Attempt {attempt} | RuntimeError: {e}")
            time.sleep(3)

    _proc.kill()
    log("S3 | DONE -- loop completed without crash")


def scenario_4():
    """
    HYPOTHESIS: Adding `del _c` after each failed attempt (no TCP probe)
    prevents the crash by cleaning up the Boost.Asio io_context immediately.

    Same as S3 but with explicit `del _c` in the exception handler.
    If TRUE:  connects without crash -- del alone is the fix.
    If FALSE: still crashes -- TCP probing is also required.

    Starts CARLA. Run via subprocess.
    """
    import carla
    log("S4 | Killing any running CARLA...")
    kill_all_carla()

    log("S4 | Starting CARLA with Town01...")
    _proc = start_carla()
    t0 = time.time()

    log("S4 | Polling WITH explicit del, WITHOUT TCP probe...")
    deadline = time.time() + 120
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        _c = None
        log(f"S4 | Attempt {attempt:03d}  t+{time.time()-t0:.1f}s | carla.Client()...")
        try:
            _c = carla.Client(CARLA_HOST, CARLA_PORT)
            _c.set_timeout(5.0)
            ver = _c.get_server_version()
            log(f"S4 | CONNECTED on attempt {attempt}: {ver}")
            del _c
            break
        except Exception as e:
            log(f"S4 | Attempt {attempt} | {type(e).__name__}: {e}")
            del _c   # explicit cleanup -- this is the only difference from S3
            _c = None
            time.sleep(3)

    _proc.kill()
    log("S4 | DONE -- loop completed without crash")


def scenario_5():
    """
    HYPOTHESIS: Even with a TCP probe (so carla.Client() is only created once
    the port accepts), calling get_world().get_map() immediately after the TCP
    port opens crashes because UE4's map isn't fully initialized yet.

    Waits for TCP port, then connects and calls get_map() immediately (0s grace).
    Logs the exact delta from TCP-open to each call result.

    If TRUE:  crash inside get_map() -- a grace period is required.
    If FALSE: get_map() succeeds -- TCP probe alone is sufficient.

    Starts CARLA. Run via subprocess.
    """
    import carla
    log("S5 | Killing any running CARLA...")
    kill_all_carla()

    log("S5 | Starting CARLA with Town01...")
    _proc = start_carla()
    t_launch = time.time()

    log("S5 | Waiting for TCP port to open (raw socket probe)...")
    while not port_accepting():
        time.sleep(0.5)
    t_open = time.time()
    log(f"S5 | TCP port open at t+{t_open - t_launch:.1f}s -- connecting IMMEDIATELY (0s grace)...")

    _c = carla.Client(CARLA_HOST, CARLA_PORT)
    _c.set_timeout(10.0)

    try:
        ver = _c.get_server_version()
        log(f"S5 | get_server_version() OK: {ver!r}  dt={time.time()-t_open:.3f}s")

        world = _c.get_world()
        log(f"S5 | get_world() OK  dt={time.time()-t_open:.3f}s")

        m = world.get_map()
        log(f"S5 | get_map() OK: {m.name!r}  dt={time.time()-t_open:.3f}s")

    except Exception as e:
        log(f"S5 | EXCEPTION at dt={time.time()-t_open:.3f}s: {type(e).__name__}: {e}")
    finally:
        del _c
        _proc.kill()

    log("S5 | DONE")


def scenario_6():
    """
    HYPOTHESIS: Killing CARLA while a carla.Client() is actively connected
    (no graceful teardown) triggers a Fast-Fail crash in the Python process.

    This is the ACTUAL notebook scenario: collect_town() finishes with a live
    `client`, then restart_carla_and_reconnect() calls proc.kill() immediately
    without releasing the client object first.

    Procedure:
      1. Start CARLA, wait for TCP port, connect with carla.Client()
      2. Call proc.kill() WITHOUT deleting the client (original notebook bug)
      3. Wait to see if the Boost.Asio background thread Fast-Fails

    If TRUE:  exits 0xC0000409 -- kill-during-active-connection is root cause.
    If FALSE: survives -- crash has a different trigger.

    Starts CARLA. Run via subprocess.
    """
    import carla
    log("S6 | Killing any running CARLA...")
    kill_all_carla()

    log("S6 | Starting CARLA...")
    _proc = start_carla()
    t0 = time.time()

    log("S6 | Waiting for TCP port to open...")
    while not port_accepting():
        time.sleep(0.5)
    t_open = time.time()
    log(f"S6 | Port open at t+{t_open - t0:.1f}s -- connecting...")
    time.sleep(2)   # brief grace so RPC is ready

    _c = carla.Client(CARLA_HOST, CARLA_PORT)
    _c.set_timeout(10.0)
    ver = _c.get_server_version()
    log(f"S6 | Connected: {ver}  dt={time.time()-t_open:.3f}s")

    log("S6 | Killing CARLA WITHOUT releasing client (no del, no teardown)...")
    _proc.kill()
    _proc.wait()
    log("S6 | CARLA killed -- waiting 5s for Boost.Asio background thread to react...")
    time.sleep(5)
    log("S6 | SURVIVED -- kill-during-active-connection does NOT crash on this system")
    # (_c destructor runs at function exit)


def scenario_7():
    """
    HYPOTHESIS: Explicitly deleting carla.Client() (and sleeping briefly) BEFORE
    killing CARLA prevents the Fast-Fail crash by letting Boost.Asio drain.

    Same setup as S6 but with `del _c` + 0.5s sleep before proc.kill().
    This mirrors the fixed notebook behavior.

    If S6 crashes and S7 passes -> del-before-kill is the confirmed fix.
    If both pass -> root cause is elsewhere.

    Starts CARLA. Run via subprocess.
    """
    import carla
    log("S7 | Killing any running CARLA...")
    kill_all_carla()

    log("S7 | Starting CARLA...")
    _proc = start_carla()
    t0 = time.time()

    log("S7 | Waiting for TCP port to open...")
    while not port_accepting():
        time.sleep(0.5)
    t_open = time.time()
    log(f"S7 | Port open at t+{t_open - t0:.1f}s -- connecting...")
    time.sleep(2)   # brief grace so RPC is ready

    _c = carla.Client(CARLA_HOST, CARLA_PORT)
    _c.set_timeout(10.0)
    ver = _c.get_server_version()
    log(f"S7 | Connected: {ver}  dt={time.time()-t_open:.3f}s")

    log("S7 | Releasing client BEFORE killing CARLA (del + 0.5s sleep -- the fix)...")
    del _c
    time.sleep(0.5)

    log("S7 | Now killing CARLA...")
    _proc.kill()
    _proc.wait()
    log("S7 | Waiting 5s...")
    time.sleep(5)
    log("S7 | DONE -- survived with proper teardown")


def _wait_for_map(client, expected, timeout=120):
    """Block until get_map() returns `expected`, then return the world.
    On timeout, logs the actual map that was loaded for diagnostics."""
    deadline = time.time() + timeout
    last_loaded = "<unknown>"
    while time.time() < deadline:
        try:
            world = client.get_world()
            loaded = world.get_map().name.split("/")[-1].replace("_Opt", "")
            last_loaded = loaded
            if loaded == expected:
                return world
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(
        f"Map '{expected}' not ready within {timeout}s "
        f"(actual map after timeout: '{last_loaded}')"
    )


def _start_carla_load_world(world_name="Town01"):
    """
    Start CARLA with -dx11 only (no CLI map arg -- they are silently ignored),
    connect, and use load_world() to load the target map.
    Returns (_proc, _c, world).

    S8/S9 used start_carla() which passes a CLI map arg that CARLA ignores.
    _wait_for_map('Town01') then timed out after 120s (CARLA loaded Town10HD_Opt).
    Both scenarios exited with RuntimeError (code 1), never reaching the sensor
    or kill steps. This helper fixes that so S11-S14 actually exercise sensors.
    """
    import carla
    kill_all_carla()
    _proc = subprocess.Popen(
        [CARLA_EXE, "-dx11", "-windowed", "-ResX=400", "-ResY=300"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    while not port_accepting():
        time.sleep(0.5)
    time.sleep(3)
    _c = carla.Client(CARLA_HOST, CARLA_PORT)
    _c.set_timeout(60.0)
    _c.load_world(world_name)
    time.sleep(3)
    world = _c.get_world()
    loaded = world.get_map().name.split("/")[-1].replace("_Opt", "")
    log(f"_start_carla_load_world | map: {loaded}")
    return _proc, _c, world


def _spawn_sensor_setup(world):
    """
    Spawn a vehicle + RGB camera in sync mode. Returns (vehicle, camera).
    Caller must tick the world first (for sync mode) after this returns.
    """
    import carla

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter("vehicle.*")[0]
    spawn_points = world.get_map().get_spawn_points()
    vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])

    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "320")
    cam_bp.set_attribute("image_size_y", "240")
    cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)

    frame_count = [0]

    def on_image(__):
        frame_count[0] += 1

    camera.listen(on_image)
    return vehicle, camera, frame_count


def scenario_8():
    """
    Most realistic simulation of the actual notebook failure mode.

    Mimics the state AFTER collect_town() finishes:
      - CARLA running with a map loaded
      - synchronous_mode = True
      - A vehicle is spawned
      - A camera sensor has an active listen() callback
      - Global client is live

    Then restart_carla_and_reconnect() kills CARLA WITHOUT stopping the
    sensor or releasing the client first (original notebook behavior).

    If TRUE:  exits 0xC0000409 -- sensor callback teardown is root cause.
    If FALSE: survives -- crash has a yet-unidentified trigger.

    Starts CARLA. Run via subprocess.
    """
    import carla
    log("S8 | Killing any running CARLA...")
    kill_all_carla()

    log("S8 | Starting CARLA with Town01...")
    _proc = start_carla()

    log("S8 | Waiting for TCP port...")
    while not port_accepting():
        time.sleep(0.5)
    time.sleep(2)

    _c = carla.Client(CARLA_HOST, CARLA_PORT)
    _c.set_timeout(30.0)
    ver = _c.get_server_version()
    log(f"S8 | Connected: {ver}")

    log("S8 | Waiting for Town01 map to finish loading...")
    world = _wait_for_map(_c, "Town01")
    log("S8 | Town01 ready -- spawning vehicle + camera sensor in sync mode...")

    vehicle, camera, frame_count = _spawn_sensor_setup(world)
    log(f"S8 | Vehicle {vehicle.id} + camera {camera.id} spawned, callbacks active")

    log("S8 | Running 20 ticks to generate callback traffic...")
    for _ in range(20):
        world.tick()
    log(f"S8 | {frame_count[0]} frames received via callback")

    log("S8 | Killing CARLA WITHOUT stopping sensor or releasing client (no teardown)...")
    _proc.kill()
    _proc.wait()
    log("S8 | CARLA killed -- waiting 5s for Boost.Asio / callback thread to react...")
    time.sleep(5)
    log("S8 | SURVIVED -- sensor callbacks + kill does NOT crash on this system")


def scenario_9():
    """
    Same setup as S8 (sync mode + vehicle + camera callback) but with proper
    teardown before killing CARLA:
      1. camera.stop()          -- flush pending callbacks, stop the data stream
      2. settings.synchronous_mode = False  -- leave world clean
      3. del camera, del vehicle, del client
      4. time.sleep(0.5)        -- let io_context drain
      5. proc.kill()

    This mirrors the FIXED notebook behavior.

    If S8 crashes and S9 passes -> sensor callback teardown is the confirmed fix.
    If both pass -> crash has a different trigger.

    Starts CARLA. Run via subprocess.
    """
    import carla
    log("S9 | Killing any running CARLA...")
    kill_all_carla()

    log("S9 | Starting CARLA with Town01...")
    _proc = start_carla()

    log("S9 | Waiting for TCP port...")
    while not port_accepting():
        time.sleep(0.5)
    time.sleep(2)

    _c = carla.Client(CARLA_HOST, CARLA_PORT)
    _c.set_timeout(30.0)
    ver = _c.get_server_version()
    log(f"S9 | Connected: {ver}")

    log("S9 | Waiting for Town01 map to finish loading...")
    world = _wait_for_map(_c, "Town01")
    log("S9 | Town01 ready -- spawning vehicle + camera sensor in sync mode...")

    vehicle, camera, frame_count = _spawn_sensor_setup(world)
    log(f"S9 | Vehicle {vehicle.id} + camera {camera.id} spawned, callbacks active")

    log("S9 | Running 20 ticks to generate callback traffic...")
    for _ in range(20):
        world.tick()
    log(f"S9 | {frame_count[0]} frames received via callback")

    log("S9 | Proper teardown: camera.stop() -> reset sync -> del actors -> del client...")
    try:
        camera.stop()
        time.sleep(0.1)
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        vehicle.destroy()
    except Exception as e:
        log(f"S9 | Teardown warning: {e}")

    del camera
    del vehicle
    del _c
    time.sleep(0.5)

    log("S9 | Killing CARLA after proper teardown...")
    _proc.kill()
    _proc.wait()
    log("S9 | Waiting 5s...")
    time.sleep(5)
    log("S9 | DONE -- survived with sensor teardown")


def scenario_10():
    """
    HYPOTHESIS: client.load_world("Town01") works when CARLA is launched with
    -dx11, even on Blackwell (RTX 5080).

    Background: load_world() was previously documented to crash on Blackwell
    (Vulkan null-pointer). Since we now launch with -dx11 (DirectX 11), the
    Vulkan code path is bypassed and load_world() may succeed.

    This matters because the CLI map argument (/Game/Carla/Maps/Town01) does
    NOT actually load Town01 -- CARLA always restores its cached map (Town10HD_Opt).
    load_world() is the only API to change the active map.

    Procedure:
      1. Kill CARLA, start fresh with -dx11 only (no map arg)
      2. Connect -- CARLA will load its cached map (Town10HD_Opt)
      3. Call client.load_world("Town01")
      4. Reconnect, call get_map() to confirm Town01 is active

    If TRUE (survives + map=Town01): load_world() works with -dx11; this is
        the correct fix for restart_carla_and_reconnect().
    If CRASHED (0xC0000409):        load_world() still crashes on Blackwell
        even with -dx11; a different map-loading strategy is needed.
    If RuntimeError:                load_world() raised Python exception;
        check message for diagnosis.

    Starts CARLA. Run via subprocess.
    """
    import carla
    log("S10 | Killing any running CARLA...")
    kill_all_carla()

    log("S10 | Starting CARLA with -dx11 only (no map arg)...")
    _proc = subprocess.Popen(
        [CARLA_EXE, "-dx11", "-windowed", "-ResX=400", "-ResY=300"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    log("S10 | Waiting for TCP port...")
    while not port_accepting():
        time.sleep(0.5)
    time.sleep(3)   # extra grace for RPC to be ready

    _c = carla.Client(CARLA_HOST, CARLA_PORT)
    _c.set_timeout(30.0)
    ver = _c.get_server_version()
    initial_map = _c.get_world().get_map().name.split("/")[-1].replace("_Opt", "")
    log(f"S10 | Connected: {ver}, initial map: {initial_map}")

    log("S10 | Calling client.load_world('Town01') with -dx11 ...")
    try:
        _c.load_world("Town01")
        log("S10 | load_world() returned without exception")
    except RuntimeError as e:
        log(f"S10 | load_world() raised RuntimeError: {e}")
        _proc.kill()
        return

    # After load_world() the client may be disconnected -- reconnect
    time.sleep(3)
    try:
        new_map = _c.get_world().get_map().name.split("/")[-1].replace("_Opt", "")
        log(f"S10 | Map after load_world(): '{new_map}'")
        if new_map == "Town01":
            log("S10 | SUCCESS -- load_world() loaded Town01 correctly!")
        else:
            log(f"S10 | WARNING -- map is '{new_map}', not Town01 (load may still be in progress)")
    except Exception as e:
        log(f"S10 | get_map() after load_world() raised: {e}")

    _proc.kill()
    log("S10 | DONE")


def scenario_11():
    """
    FIXED S8: sensors + sync mode + 200 ticks + kill WITHOUT teardown.
    Uses load_world() so Town01 actually loads (CLI map arg was silently ignored;
    S8 raised RuntimeError from _wait_for_map and never reached the sensors).

    If TRUE  (crash): kill-during-active-sensor-callbacks is root cause.
    If FALSE (pass):  crash has a different trigger.

    Starts CARLA. Run via subprocess.
    """
    log("S11 | Killing any running CARLA...")
    _proc, _c, world = _start_carla_load_world("Town01")

    log("S11 | Spawning vehicle + camera (counter callback, sync mode, 200 ticks)...")
    vehicle, camera, frame_count = _spawn_sensor_setup(world)
    log(f"S11 | Vehicle {vehicle.id} + camera {camera.id} spawned")

    log("S11 | Running 200 ticks in sync mode...")
    for i in range(200):
        world.tick()
        if i % 20 == 0:
            log(f"S11 | tick {i+1}/200 (frames so far: {frame_count[0]})")
    log(f"S11 | {frame_count[0]} frames received -- killing CARLA WITHOUT teardown...")
    _proc.kill()
    _proc.wait()
    time.sleep(5)
    log("S11 | SURVIVED -- sensors+sync+kill does NOT crash on this system")


def scenario_12():
    """
    FIXED S9: sensors + sync mode + 200 ticks + PROPER teardown before kill.
    Uses load_world() so Town01 actually loads.

    If S11 crashes and S12 passes -> sensor callback teardown is the fix.
    If both pass -> crash has a different trigger.

    Starts CARLA. Run via subprocess.
    """
    log("S12 | Killing any running CARLA...")
    _proc, _c, world = _start_carla_load_world("Town01")

    log("S12 | Spawning vehicle + camera (counter callback, sync mode, 200 ticks)...")
    vehicle, camera, frame_count = _spawn_sensor_setup(world)
    log(f"S12 | Vehicle {vehicle.id} + camera {camera.id} spawned")

    log("S12 | Running 200 ticks in sync mode...")
    for _ in range(200):
        world.tick()
    log(f"S12 | {frame_count[0]} frames -- starting proper teardown...")

    try:
        camera.stop()
        time.sleep(0.1)
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = 0.0
        world.apply_settings(settings)
        camera.destroy()
        vehicle.destroy()
    except Exception as e:
        log(f"S12 | Teardown warning: {e}")

    del camera, vehicle, _c
    time.sleep(1.5)

    log("S12 | Killing CARLA after teardown...")
    _proc.kill()
    _proc.wait()
    time.sleep(5)
    log("S12 | DONE -- survived with proper teardown")


def scenario_13():
    """
    Most notebook-accurate: camera uses Queue.put callback (not just a counter).
    200 ticks in sync mode. Kill WITHOUT teardown.

    Key difference from S11: callback is `lambda img: q.put(img)` -- exactly as
    in the notebook. carla.Image objects accumulate in q, backed by libcarla's
    shared buffer pool, and are NOT drained before kill.

    If S13 crashes where S11 passed -> Queue.put / carla.Image retention is the
    root cause of the notebook crash.

    Starts CARLA. Run via subprocess.
    """
    from queue import Queue

    import carla
    log("S13 | Killing any running CARLA...")
    _proc, _c, world = _start_carla_load_world("Town01")

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])

    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "320")
    cam_bp.set_attribute("image_size_y", "240")
    camera = world.spawn_actor(cam_bp,
                               carla.Transform(carla.Location(x=1.5, z=2.4)),
                               attach_to=vehicle)

    q = Queue()
    camera.listen(lambda img: q.put(img))
    log(f"S13 | Vehicle {vehicle.id} + camera {camera.id} with Queue.put callback")

    log("S13 | Running 200 ticks...")
    for _ in range(200):
        world.tick()
    log(f"S13 | Queue size: {q.qsize()} images -- killing WITHOUT teardown...")
    _proc.kill()
    _proc.wait()
    time.sleep(5)
    log("S13 | SURVIVED")


def scenario_14():
    """
    Camera + Queue.put callback + EXACT fixed teardown sequence (matches
    the notebook fixes applied 2026-03-28):

      1. world.apply_settings(sync=False)   -- disable sync first
      2. time.sleep(1.0)                    -- let async callbacks drain into queue
      3. camera.stop()                      -- stop callback stream
      4. camera.destroy() / vehicle.destroy()
      5. drain queue                        -- clear carla.Image objects
      6. del _c                             -- release client
      7. time.sleep(1.5)                    -- drain libcarla background threads
      8. kill CARLA

    If S13 crashes and S14 passes -> the fixed teardown sequence is confirmed correct.
    If both crash -> teardown order doesn't matter; crash is in a different path.

    Starts CARLA. Run via subprocess.
    """
    from queue import Empty, Queue

    import carla
    log("S14 | Killing any running CARLA...")
    _proc, _c, world = _start_carla_load_world("Town01")

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])

    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "320")
    cam_bp.set_attribute("image_size_y", "240")
    camera = world.spawn_actor(cam_bp,
                               carla.Transform(carla.Location(x=1.5, z=2.4)),
                               attach_to=vehicle)

    q = Queue()
    camera.listen(lambda img: q.put(img))
    log(f"S14 | Vehicle {vehicle.id} + camera {camera.id} with Queue.put callback")

    log("S14 | Running 200 ticks...")
    for _ in range(200):
        world.tick()
    log(f"S14 | Queue size: {q.qsize()} -- starting FIXED teardown sequence...")

    # Step 1: disable sync mode first
    try:
        s = world.get_settings()
        s.synchronous_mode = False
        s.fixed_delta_seconds = 0.0
        world.apply_settings(s)
        log("S14 | sync=False applied")
    except Exception as e:
        log(f"S14 | sync=False warning: {e}")

    # Step 2: let async callbacks drain
    time.sleep(1.0)
    log(f"S14 | Queue size after 1s async: {q.qsize()}")

    # Step 3: stop sensor (disconnect callback)
    try:
        camera.stop()
        log("S14 | camera.stop() OK")
    except Exception as e:
        log(f"S14 | camera.stop() warning: {e}")

    # Step 4: destroy actors
    try:
        camera.destroy()
        vehicle.destroy()
        log("S14 | actors destroyed")
    except Exception as e:
        log(f"S14 | destroy warning: {e}")

    # Step 5: drain carla.Image objects from queue
    drained = 0
    while not q.empty():
        try:
            q.get_nowait()
            drained += 1
        except Empty:
            break
    log(f"S14 | drained {drained} images from queue")

    # Step 6: release client
    del _c
    # Step 7: let libcarla background threads wind down
    time.sleep(1.5)

    log("S14 | Killing CARLA...")
    _proc.kill()
    _proc.wait()
    time.sleep(5)
    log("S14 | DONE -- survived with fixed teardown")


def scenario_15():
    """
    S11 + pre-camera stabilization ticks.

    Hypothesis: S11 hangs because CARLA is still loading background resources
    (nav meshes, etc.) when the camera tries to render on tick 2.  Ticks 1-N
    without any camera let CARLA finish initialization; the camera is spawned
    AFTER those ticks and registered AFTER they complete.

    If TRUE  (pass): pre-camera stabilization ticks fix the rendering deadlock.
    If FALSE (hang): the deadlock is not timing-related; a different fix is needed.

    Starts CARLA. Run via subprocess.
    """
    import carla
    log("S15 | Killing any running CARLA...")
    _proc, _c, world = _start_carla_load_world("Town01")

    # Enable sync mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
    log(f"S15 | Vehicle {vehicle.id} spawned -- running 10 PRE-CAMERA stabilization ticks...")

    # Let CARLA finish background initialization before spawning sensors
    for i in range(10):
        world.tick()
        if i % 5 == 0:
            log(f"S15 | pre-camera tick {i+1}/10")

    # NOW spawn camera and register callback
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "320")
    cam_bp.set_attribute("image_size_y", "240")
    camera = world.spawn_actor(cam_bp,
                               carla.Transform(carla.Location(x=1.5, z=2.4)),
                               attach_to=vehicle)
    frame_count = [0]
    camera.listen(lambda __: frame_count.__setitem__(0, frame_count[0] + 1))
    log(f"S15 | Camera {camera.id} spawned with callback -- running 200 ticks...")

    for i in range(200):
        world.tick()
        if i % 20 == 0:
            log(f"S15 | tick {i+1}/200 (frames: {frame_count[0]})")

    log(f"S15 | {frame_count[0]} frames -- killing CARLA WITHOUT teardown...")
    _proc.kill()
    _proc.wait()
    time.sleep(5)
    log("S15 | SURVIVED -- pre-camera stabilization ticks fix the deadlock")


def scenario_16():
    """
    S16: Spawn camera WITHOUT listen(), do one sync tick (pipeline init),
    THEN attach listen(), run 200 ticks.

    Hypothesis: the tick-2 deadlock occurs because CARLA tries to initialize
    the camera rendering pipeline AND deliver a frame to Python's callback
    simultaneously while nav mesh loading is in progress. Separating pipeline
    init (tick without callback) from callback registration may avoid the race.
    """
    import carla

    log("S16 | Killing any running CARLA...")
    _proc, _c, world = _start_carla_load_world("Town01")

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
    log(f"S16 | Vehicle {vehicle.id} spawned")

    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "320")
    cam_bp.set_attribute("image_size_y", "240")
    camera = world.spawn_actor(
        cam_bp,
        carla.Transform(carla.Location(x=1.5, z=2.4)),
        attach_to=vehicle,
    )
    log(f"S16 | Camera {camera.id} spawned WITHOUT listen() -- doing 1 warmup tick...")
    world.tick()  # let CARLA initialize the camera rendering pipeline before callback
    log("S16 | Warmup tick done -- attaching listen() callback...")
    frame_count = [0]
    camera.listen(lambda __: frame_count.__setitem__(0, frame_count[0] + 1))
    log("S16 | Running 200 ticks with callback active...")

    for i in range(200):
        world.tick()
        if i % 20 == 0:
            log(f"S16 | tick {i+1}/200 (frames: {frame_count[0]})")

    log(f"S16 | {frame_count[0]} frames -- killing CARLA WITHOUT teardown...")
    _proc.kill()
    _proc.wait()
    time.sleep(5)
    log("S16 | SURVIVED -- deferred listen() avoids the camera pipeline deadlock")


def scenario_17():
    """
    S17: Spawn vehicle+camera+listen() in ASYNC mode, sleep 5s to let the
    camera pipeline warm up naturally, then enable sync mode and tick 200 times.

    Hypothesis: in async mode CARLA renders the camera on its own schedule
    without deadlocking. Switching to sync mode AFTER the pipeline is warm
    avoids the initialization race on tick 2.
    """
    import carla

    log("S17 | Killing any running CARLA...")
    _proc, _c, world = _start_carla_load_world("Town01")

    # Intentionally stay in async mode during sensor spawn
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
    log(f"S17 | Vehicle {vehicle.id} spawned (async mode)")

    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "320")
    cam_bp.set_attribute("image_size_y", "240")
    camera = world.spawn_actor(
        cam_bp,
        carla.Transform(carla.Location(x=1.5, z=2.4)),
        attach_to=vehicle,
    )
    frame_count = [0]
    camera.listen(lambda __: frame_count.__setitem__(0, frame_count[0] + 1))
    log(f"S17 | Camera {camera.id} spawned with callback in ASYNC mode -- sleeping 5s...")
    time.sleep(5.0)
    log(f"S17 | Frames received in async: {frame_count[0]} -- enabling sync mode...")

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    log("S17 | Sync mode enabled -- running 200 ticks...")

    for i in range(200):
        world.tick()
        if i % 20 == 0:
            log(f"S17 | tick {i+1}/200 (frames: {frame_count[0]})")

    log(f"S17 | {frame_count[0]} frames -- killing CARLA WITHOUT teardown...")
    _proc.kill()
    _proc.wait()
    time.sleep(5)
    log("S17 | SURVIVED -- async warmup before sync avoids the camera pipeline deadlock")


def scenario_18():
    """
    S18: Async warmup with 300s timeout and adaptive wait.

    S17 showed that `world.get_settings()` (RPC) hangs immediately after the camera
    starts rendering in async mode -- nav mesh loading (triggered by the camera) blocks
    ALL RPC calls for 60+ seconds.

    Fix: set client timeout to 300s and wait at least 30s (adaptive) until CARLA's
    frame delivery rate stabilizes before calling any RPC.
    """
    import carla

    log("S18 | Killing any running CARLA...")
    _proc, _c, world = _start_carla_load_world("Town01")

    # Extend timeout so nav mesh / camera init cannot time out
    _c.set_timeout(300.0)
    world = _c.get_world()  # refresh world reference with new timeout

    # Spawn in async mode (sync mode NOT enabled yet)
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
    log(f"S18 | Vehicle {vehicle.id} spawned (async mode, timeout=300s)")

    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "320")
    cam_bp.set_attribute("image_size_y", "240")
    camera = world.spawn_actor(
        cam_bp,
        carla.Transform(carla.Location(x=1.5, z=2.4)),
        attach_to=vehicle,
    )
    frame_count = [0]
    camera.listen(lambda __: frame_count.__setitem__(0, frame_count[0] + 1))
    log("S18 | Camera spawned with callback -- waiting for async frame rate to stabilize...")

    # Adaptive wait: sample every 2s, require 30s minimum and >=5 fps before proceeding
    MIN_WAIT_SECONDS = 30.0
    MAX_WAIT_SECONDS = 120.0
    t0 = time.time()
    prev_count = 0
    ready = False
    while time.time() - t0 < MAX_WAIT_SECONDS:
        time.sleep(2.0)
        current = frame_count[0]
        delta = current - prev_count  # frames in last 2s
        elapsed = time.time() - t0
        fps_2s = delta / 2.0
        log(f"S18 | t={elapsed:.1f}s  total_frames={current}  +{delta} in 2s ({fps_2s:.1f} fps)")
        prev_count = current
        if elapsed >= MIN_WAIT_SECONDS and fps_2s >= 5.0:
            log("S18 | Frame rate >=5 fps after 30s min wait -- CARLA init complete")
            ready = True
            break

    if not ready:
        log(f"S18 | WARNING: {frame_count[0]} total frames after {MAX_WAIT_SECONDS:.0f}s -- proceeding anyway")

    log(f"S18 | Enabling sync mode (total async frames: {frame_count[0]})...")
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    log("S18 | Sync mode enabled -- running 200 ticks...")

    for i in range(200):
        world.tick()
        if i % 20 == 0:
            log(f"S18 | tick {i+1}/200 (frames: {frame_count[0]})")

    log(f"S18 | {frame_count[0]} frames -- killing CARLA WITHOUT teardown...")
    _proc.kill()
    _proc.wait()
    time.sleep(5)
    log("S18 | SURVIVED -- async warmup + wait-for-stable-fps fixes the deadlock")


def scenario_19():
    """
    S19: Launch CARLA with -RenderOffScreen instead of -windowed.

    S11-S18 all show that CARLA's dx11 camera rendering pipeline deadlocks
    permanently (rendering thread frozen, no RPC response) after ~5 camera
    frames on RTX 5080 Blackwell. The deadlock is in CARLA's windowed dx11
    viewport code path.

    -RenderOffScreen bypasses the windowed viewport entirely -- UE4 renders
    sensors to an offscreen d3d buffer without a window swapchain, which
    should avoid the deadlock. Camera sensors still produce frames.
    """
    import carla

    log("S19 | Killing any running CARLA...")
    kill_all_carla()

    proc = subprocess.Popen(
        [CARLA_EXE, "-dx11", "-RenderOffScreen"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    while not port_accepting():
        time.sleep(0.5)
    time.sleep(3)

    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(120.0)
    client.load_world("Town01")
    time.sleep(3)
    world = client.get_world()
    loaded = world.get_map().name.split("/")[-1].replace("_Opt", "")
    log(f"S19 | Map: {loaded}")

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])

    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "320")
    cam_bp.set_attribute("image_size_y", "240")
    camera = world.spawn_actor(
        cam_bp,
        carla.Transform(carla.Location(x=1.5, z=2.4)),
        attach_to=vehicle,
    )
    frame_count = [0]
    camera.listen(lambda __: frame_count.__setitem__(0, frame_count[0] + 1))
    log(f"S19 | Vehicle {vehicle.id} + Camera {camera.id} -- running 200 ticks...")

    for i in range(200):
        world.tick()
        if i % 20 == 0:
            log(f"S19 | tick {i+1}/200 (frames: {frame_count[0]})")

    log(f"S19 | {frame_count[0]} frames -- killing CARLA WITHOUT teardown...")
    proc.kill()
    proc.wait()
    time.sleep(5)
    log("S19 | SURVIVED -- -RenderOffScreen avoids the dx11 camera rendering deadlock")


def scenario_20():
    """
    S20: Launch CARLA with -dx12 (D3D12) instead of -dx11.

    D3D12 has a completely different threading model from D3D11. If the deadlock
    is caused by a D3D11 driver bug or resource contention on RTX 5080 Blackwell,
    D3D12 may avoid it entirely. UE4.26 (which CARLA 0.9.16 uses) supports D3D12
    on Windows via the -dx12 flag.
    """
    import carla

    log("S20 | Killing any running CARLA...")
    kill_all_carla()

    proc = subprocess.Popen(
        [CARLA_EXE, "-dx12", "-windowed", "-ResX=400", "-ResY=300"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    while not port_accepting():
        time.sleep(0.5)
    time.sleep(3)

    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(120.0)
    client.load_world("Town01")
    time.sleep(3)
    world = client.get_world()
    loaded = world.get_map().name.split("/")[-1].replace("_Opt", "")
    log(f"S20 | Map: {loaded}")

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])

    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "320")
    cam_bp.set_attribute("image_size_y", "240")
    camera = world.spawn_actor(
        cam_bp,
        carla.Transform(carla.Location(x=1.5, z=2.4)),
        attach_to=vehicle,
    )
    frame_count = [0]
    camera.listen(lambda __: frame_count.__setitem__(0, frame_count[0] + 1))
    log(f"S20 | Vehicle {vehicle.id} + Camera {camera.id} -- running 200 ticks with -dx12...")

    for i in range(200):
        world.tick()
        if i % 20 == 0:
            log(f"S20 | tick {i+1}/200 (frames: {frame_count[0]})")

    log(f"S20 | {frame_count[0]} frames -- killing CARLA WITHOUT teardown...")
    proc.kill()
    proc.wait()
    time.sleep(5)
    log("S20 | SURVIVED -- D3D12 avoids the RTX 5080 Blackwell camera rendering deadlock")


def scenario_21():
    """
    S21: Use sensor.camera.semantic_segmentation instead of sensor.camera.rgb.

    Semantic segmentation uses a simpler single-pass stencil shader instead of
    the full PBR rendering pipeline. If the deadlock is in the full PBR/GBuffer
    pass (specific to sensor.camera.rgb), semantic_segmentation might survive.
    """
    import carla

    log("S21 | Killing any running CARLA...")
    _proc, _c, world = _start_carla_load_world("Town01")

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])

    cam_bp = bp_lib.find("sensor.camera.semantic_segmentation")
    cam_bp.set_attribute("image_size_x", "320")
    cam_bp.set_attribute("image_size_y", "240")
    camera = world.spawn_actor(
        cam_bp,
        carla.Transform(carla.Location(x=1.5, z=2.4)),
        attach_to=vehicle,
    )
    frame_count = [0]
    camera.listen(lambda __: frame_count.__setitem__(0, frame_count[0] + 1))
    log(f"S21 | Vehicle {vehicle.id} + SemanticSeg camera {camera.id} -- running 200 ticks...")

    for i in range(200):
        world.tick()
        if i % 20 == 0:
            log(f"S21 | tick {i+1}/200 (frames: {frame_count[0]})")

    log(f"S21 | {frame_count[0]} frames -- killing CARLA WITHOUT teardown...")
    _proc.kill()
    _proc.wait()
    time.sleep(5)
    log("S21 | SURVIVED -- semantic_segmentation camera avoids the RGB shader deadlock")


# --- Test Runner --------------------------------------------------------------

SCENARIOS = {
    1: ("carla.Client() on CLOSED port -- does constructor crash?",                scenario_1,  False),
    2: ("30x carla.Client() accumulation (no del) -- resource overflow?",          scenario_2,  False),
    3: ("Original loop: no TCP probe, no del -- reproduces crash?",                scenario_3,  True),
    4: ("del without TCP probe -- is del alone sufficient?",                       scenario_4,  True),
    5: ("TCP probe + immediate get_map() (0s grace) -- does get_map() crash?",     scenario_5,  True),
    6: ("Kill CARLA with active connection, NO teardown -- Fast-Fail crash?",      scenario_6,  True),
    7: ("Kill CARLA with active connection, del+sleep first -- survives?",         scenario_7,  True),
    8: ("Sensors+sync+kill, NO teardown [BROKEN: Town01 never loaded]",           scenario_8,  True),
    9: ("Sensors+sync+kill, teardown    [BROKEN: Town01 never loaded]",           scenario_9,  True),
   10: ("load_world('Town01') with -dx11 -- works on Blackwell?",                  scenario_10, True),
   11: ("FIXED S8: sensors+sync+200ticks+kill, NO teardown (load_world)",         scenario_11, True),
   12: ("FIXED S9: sensors+sync+200ticks+kill, proper teardown (load_world)",     scenario_12, True),
   13: ("Queue.put callback+200ticks+kill, NO teardown -- notebook-accurate",      scenario_13, True),
   14: ("Queue.put callback+200ticks+FIXED teardown -- tests notebook fix",        scenario_14, True),
   15: ("Pre-camera stabilization ticks -- does camera work after 10 no-cam ticks?", scenario_15, True),
   16: ("Deferred listen() -- 1 warmup tick without callback, then attach+200 ticks", scenario_16, True),
   17: ("Async warmup -- spawn camera+listen() in async, sleep 5s, then sync+200 ticks", scenario_17, True),
   18: ("Async warmup 300s timeout -- wait until frame rate stabilizes, then sync+200 ticks", scenario_18, True),
   19: ("-RenderOffScreen launch -- does dx11 camera deadlock disappear without a window?", scenario_19, True),
   20: ("-dx12 launch + RGB camera -- D3D12 avoids Blackwell dx11 deadlock?",             scenario_20, True),
   21: ("semantic_segmentation camera (dx11) -- simpler shader avoids RGB deadlock?",     scenario_21, True),
}


def run_in_subprocess(n):
    """Run scenario N in a child process. Returns (survived: bool, exit_code: int)."""
    result = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--run", str(n)],
        timeout=600,
    )
    rc = result.returncode
    if rc == CRASH_EXIT:
        log(f">>> SCENARIO {n} CRASHED (exit {rc} = 0xC0000409)  <- ROOT CAUSE CONFIRMED")
        return False, rc
    elif rc == 0:
        log(f">>> Scenario {n} completed without crash (exit 0)")
        return True, rc
    else:
        log(f">>> Scenario {n} exited with code {rc}")
        return True, rc


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--scenario", type=int, help="Run a single scenario (1-14)")
    parser.add_argument("--run", type=int, help=argparse.SUPPRESS)  # internal subprocess use
    args = parser.parse_args()

    # Internal: direct scenario execution (called from run_in_subprocess)
    if args.run is not None:
        n = args.run
        _, fn, _ = SCENARIOS[n]
        fn()
        sys.exit(0)

    # Determine which scenarios to run
    to_run = [args.scenario] if args.scenario else list(SCENARIOS.keys())

    log(f"\n{'='*70}")
    log(f"CARLA CRASH DIAGNOSTIC -- {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Log file: {LOG_FILE}")
    log(f"{'='*70}\n")

    results = {}
    for n in to_run:
        desc, fn, dangerous = SCENARIOS[n]
        log(f"\n{'-'*70}")
        log(f"SCENARIO {n}: {desc}")
        log(f"Mode: {'subprocess (might crash)' if dangerous else 'in-process (safe)'}")
        log(f"{'-'*70}")

        if dangerous:
            survived, rc = run_in_subprocess(n)
            results[n] = survived
            # Clean up CARLA left behind by a crashed subprocess
            kill_all_carla()
        else:
            fn()
            results[n] = True

    # Summary
    log(f"\n{'='*70}")
    log("SUMMARY")
    log(f"{'='*70}")
    for n, survived in results.items():
        desc, _, dangerous = SCENARIOS[n]
        status = "PASSED" if survived else "CRASHED <- root cause"
        log(f"  S{n}: {status:30s}  {desc}")

    # Interpretation guide
    log(f"\n{'-'*70}")
    log("INTERPRETATION:")
    log("  S1 or S2 crash -> fundamental libcarla issue, independent of server state")
    log("  S3 crashes, S4 passes -> explicit `del _c` after each attempt is the fix")
    log("  S3 crashes, S4 crashes, S5 passes -> TCP probe alone is the fix")
    log("  S3 crashes, S4 crashes, S5 crashes -> grace period after TCP open is needed")
    log("  S6 crashes, S7 passes -> kill-during-active-connection is root cause;")
    log("                          del-before-kill (the notebook fix) is confirmed correct")
    log("  S6 passes -> crash has a different trigger not yet identified")
    log("  NOTE: S8/S9 are INVALID -- _wait_for_map('Town01') always timed out because")
    log("        CARLA ignores the CLI map arg and loads Town10HD. Both exited with")
    log("        RuntimeError (code 1), never reaching the sensor or kill steps.")
    log("        Use S11-S14 instead.")
    log("  S11 crashes, S12 passes -> kill-during-sensor-callback is root cause (counter cb)")
    log("  S11 passes, S13 crashes -> Queue.put / carla.Image retention causes the crash")
    log("  S13 crashes, S14 passes -> the fixed teardown sequence (drain+stop+del) is correct")
    log("  S13 crashes, S14 crashes -> teardown order is not the issue; further investigation needed")
    log("  All 11-14 pass -> crash is caused by something specific to Jupyter or longer runs")
    log("  S11 hangs, S16 passes -> deferred listen() (warmup tick before callback) is the fix")
    log("  S11 hangs, S16 hangs, S17 passes -> async warmup before sync mode is the fix")
    log("  S11 hangs, S16 hangs, S17 hangs -> RTX 5080 dx11 camera deadlock needs different workaround")
    log("  S17 hangs, S18 passes -> nav mesh loading blocks RPC; must wait for init to complete before sync")
    log("  S18 hangs, S19 passes -> dx11 viewport pipeline causes deadlock; -RenderOffScreen is the fix")
    log(f"{'='*70}\n")


if __name__ == "__main__":
    main()
