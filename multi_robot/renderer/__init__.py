"""Modular pygame renderer used for the multi robot experiments."""
from __future__ import annotations

import sys
from types import ModuleType

from .core import MultiRobotSimulator, demo
from .state import RobotState
from .cli import main

__all__ = ["MultiRobotSimulator", "RobotState", "demo", "main"]


def _ensure_module_entry_point() -> None:
    """Register a ``multi_robot.renderer.__main__`` module if missing.

    Some installation workflows (notably on Windows) can end up with the package
    data excluding :mod:`multi_robot.renderer.__main__`.  This helper registers a
    lightweight shim so that ``python -m multi_robot.renderer`` still resolves to
    :func:`main` even when the dedicated module file is not present.
    """

    module_name = f"{__name__}.__main__"
    if module_name in sys.modules:
        return

    shim = ModuleType(module_name, "Compatibility shim for the renderer CLI")
    shim.main = main  # type: ignore[attr-defined]
    sys.modules[module_name] = shim


_ensure_module_entry_point()
