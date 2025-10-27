"""Helpers for the renderer's formation-aware velocity adjustment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Tuple


def _clip(value: float, lower: float, upper: float) -> float:
    """Lightweight alternative to :func:`numpy.clip`."""

    return min(max(value, lower), upper)


@dataclass(frozen=True)
class SpeedAdjustmentParameters:
    """Container for the formation-aware velocity adjustment gains."""

    eta_theta: float = 0.5
    eta_d: float = 2.0
    k_d: float = 30
    omega_fixed: float = 0.4 * math.pi
    t_df: float = 0.5

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, float]) -> "SpeedAdjustmentParameters":
        """Create parameters from a mapping with optional keys.

        Missing keys fall back to the dataclass defaults, allowing callers to
        provide partial configuration dictionaries. Legacy keys (``k_theta``,
        ``p_theta`` and ``p_d``) remain supported for backwards compatibility.
        """

        legacy_k_theta = float(mapping.get("k_theta", cls.omega_fixed))
        return cls(
            eta_theta=float(
                mapping.get("eta_theta", mapping.get("p_theta", cls.eta_theta))
            ),
            eta_d=float(mapping.get("eta_d", mapping.get("p_d", cls.eta_d))),
            k_d=float(mapping.get("k_d", cls.k_d)),
            omega_fixed=float(
                mapping.get(
                    "omega_fixed",
                    mapping.get("omega_gain", legacy_k_theta),
                )
            ),
            t_df=float(mapping.get("T_df", mapping.get("t_df", cls.t_df))),
        )


def adjust_velocity(
    v_e: float,
    omega_e: float,
    d_f: float,
    d_o: float,
    theta_f: float,
    eta_theta: float,
    eta_d: float,
    k_d: float,
    omega_fixed: float,
    t_df: float,
    v_max: float,
    omega_max: float,
) -> Tuple[float, float]:
    """Apply formation-aware scaling to the executable velocity command.

    The attenuation terms follow the formation-control design provided by the
    operators:

    * ``l_theta = ((-cos(theta_f) + 1) / 2) ** eta_theta``
    * ``l_d = min(((1 / d_f - 1 / T_df) ** eta_d) / k_d, 1)`` when ``d_f < T_df``
    * ``l_t = max(l_theta - l_d, 0)``

    The helper then computes ``v_a = K[v_e * (1 - l_t)]`` and
    ``omega_a = K[omega_e + min(|d_o|, 1) * (omega_f - omega_e)]`` with ``K``
    implemented via simple saturation to the platform limits ``[0, V_MAX]`` and
    ``[-OMEGA_MAX, OMEGA_MAX]``. ``omega_f`` uses a fixed magnitude (defaulting
    to ``0.4π``) and follows the sign of ``theta_f`` so the robot turns at a
    consistent speed toward the formation target regardless of the heading
    error.
    """

    base = (-math.cos(theta_f) + 1.0) / 2.0
    base = _clip(base, 0.0, 1.0)
    eta_theta = max(eta_theta, 0.0)
    l_theta = float(base**eta_theta) if eta_theta != 0.0 else 1.0

    if d_f < t_df and d_f > 0.0 and t_df > 0.0:
        inv_distance = 1.0 / max(d_f, 1e-6)
        inv_threshold = 1.0 / t_df
        distance_term = max(inv_distance - inv_threshold, 0.0)
        eta_d = max(eta_d, 0.0)
        if eta_d == 0.0:
            distance_power = 1.0
        else:
            distance_power = distance_term**eta_d
        divisor = k_d if k_d > 0.0 else 1.0
        l_d = min(distance_power / divisor, 1.0)
    else:
        l_d = 0.0

    l_t = max(l_theta - l_d, 0.0)

    v_a = float(_clip(v_e * (1.0 - l_t), 0.0, v_max))

    omega_magnitude = abs(omega_fixed)
    if omega_magnitude == 0.0 or theta_f == 0.0:
        omega_f = 0.0
    else:
        omega_f = math.copysign(omega_magnitude, theta_f)
    omega_f = float(_clip(omega_f, -omega_max, omega_max))
    # print(omega_f)
    blend = min(abs(2 * d_o), 1.0)
    print(blend)
    omega_target = omega_e + blend * (omega_f - omega_e)
    omega_a = float(_clip(omega_target, -omega_max, omega_max))
    return v_a, omega_a


__all__ = ["SpeedAdjustmentParameters", "adjust_velocity"]

