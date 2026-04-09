"""CARLA helper functions for weather and environment creation."""

from __future__ import annotations

import carla

from src.carla_env import CarlaEnv


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
