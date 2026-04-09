"""CARLA connectivity tests -- skipped if CARLA server is not running."""

import socket

import pytest

_CARLA_HOST = "localhost"
_CARLA_PORT = 2000


def _carla_reachable() -> bool:
    """Check if CARLA server is listening on the default port."""
    try:
        with socket.create_connection((_CARLA_HOST, _CARLA_PORT), timeout=2):
            return True
    except (ConnectionRefusedError, OSError):
        return False


_skip_no_carla = pytest.mark.skipif(
    not _carla_reachable(),
    reason="CARLA server not running on localhost:2000",
)


@_skip_no_carla
def test_carla_client_connects() -> None:
    """Verify CARLA client can connect, get world, and tick once."""
    import carla

    client = carla.Client(_CARLA_HOST, _CARLA_PORT)
    client.set_timeout(10.0)
    world = client.get_world()
    assert world is not None
    world.tick()


@_skip_no_carla
def test_sync_mode() -> None:
    """Verify synchronous mode can be toggled."""
    import carla

    client = carla.Client(_CARLA_HOST, _CARLA_PORT)
    client.set_timeout(10.0)
    world = client.get_world()

    settings = world.get_settings()
    original_sync = settings.synchronous_mode

    # Toggle and restore
    settings.synchronous_mode = not original_sync
    world.apply_settings(settings)
    assert world.get_settings().synchronous_mode == (not original_sync)

    settings.synchronous_mode = original_sync
    world.apply_settings(settings)
