"""Utilities for accessing the :mod:`simulation_core` environment objects."""
from __future__ import annotations

import importlib
import importlib.resources
import importlib.util
from typing import Optional

__all__ = [
    "SimCoreEnvironment",
    "LiDARSensor",
    "hex_to_rgb",
    "switch_map_type",
    "AD2pos",
    "default_config_path",
]

_SPEC = importlib.util.find_spec("simulation_core.environment")
if _SPEC is None:
    _SPEC = importlib.util.find_spec("simulation_core.simulation_core.environment")
    if _SPEC is None:
        raise ModuleNotFoundError("Unable to locate simulation_core.environment")
    _environment_module = importlib.import_module(
        "simulation_core.simulation_core.environment"
    )
else:
    _environment_module = importlib.import_module("simulation_core.environment")

SimCoreEnvironment = _environment_module.SimCoreEnvironment

_sensor_module = importlib.import_module(f"{_environment_module.__package__}.sensor")
_utils_module = importlib.import_module(f"{_environment_module.__package__}.utils")

LiDARSensor = _sensor_module.LiDARSensor
hex_to_rgb = _utils_module.hex_to_rgb
switch_map_type = _utils_module.switch_map_type
AD2pos = _utils_module.AD2pos


def default_config_path() -> str:
    """Return the default configuration file shipped with :mod:`simulation_core`."""

    return str(
        importlib.resources.files("simulation_core").parent
        / "config"
        / "simulator_config.yaml"
    )
