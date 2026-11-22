"""Utilities to access the :mod:`simulation_core` environment modules."""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType

__all__ = [
    "SimCoreEnvironment",
    "LiDARSensor",
    "utils_module",
]


def _load_environment_module() -> ModuleType:
    """Load the :mod:`simulation_core.environment` module."""

    spec = importlib.util.find_spec("simulation_core.environment")
    if spec is not None:
        return importlib.import_module("simulation_core.environment")

    spec = importlib.util.find_spec("simulation_core.simulation_core.environment")
    if spec is not None:
        return importlib.import_module("simulation_core.simulation_core.environment")

    raise ModuleNotFoundError("Unable to locate simulation_core.environment")


_environment_module = _load_environment_module()
SimCoreEnvironment = _environment_module.SimCoreEnvironment

_sensor_module = importlib.import_module(f"{_environment_module.__package__}.sensor")
LiDARSensor = _sensor_module.LiDARSensor

utils_module = importlib.import_module(f"{_environment_module.__package__}.utils")

