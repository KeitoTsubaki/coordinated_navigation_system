"""Public API for the multi-robot navigation package."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "MultiRobotSimulator",
    "PolicyController",
    "RobotState",
    "FormationPlan",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - exercised via imports
    if name == "PolicyController":
        return import_module(".policy2_newobs", __name__).PolicyController
    if name == "MultiRobotSimulator":
        return import_module(".simulator_v5", __name__).MultiRobotSimulator
    if name in {"RobotState", "FormationPlan"}:
        types_module = import_module(".types", __name__)
        return getattr(types_module, name)
    raise AttributeError(name)


def __dir__() -> list[str]:  # pragma: no cover - matches __all__
    return sorted(__all__)

