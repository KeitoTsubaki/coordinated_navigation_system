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
    blend = min(abs(d_o), 1.0)
    omega_target = omega_e + blend * (omega_f - omega_e)
    omega_a = float(_clip(omega_target, -omega_max, omega_max))
    return v_a, omega_a


def plot_speed_adjustment_heatmap(
    theta_range: tuple[float, float] = (-math.pi, math.pi),
    distance_range: tuple[float, float] = (0.0, 2.0),
    *,
    samples_theta: int = 181,
    samples_distance: int = 100,
    parameters: SpeedAdjustmentParameters | Mapping[str, float] | None = None,
    v_e: float = 1.0,
    omega_e: float = 0.0,
    d_o: float = 0.0,
    v_max: float = 1.0,
    omega_max: float = math.pi,
):
    """Visualise the attenuation applied by :func:`adjust_velocity`.

    The helper samples the provided ``theta`` and ``distance`` ranges on a
    regular grid and computes the forward speed ratio ``v_a / v_max`` for each
    point. The resulting array is rendered as a heatmap using matplotlib where
    warm colours indicate regions with higher executable velocity and cooler
    colours highlight stronger attenuation.

    Parameters
    ----------
    theta_range:
        Inclusive range of heading errors (in radians) used for the horizontal
        axis of the heatmap. Defaults to ``[-π, π]``.
    distance_range:
        Inclusive range of formation distance errors (in metres) used for the
        vertical axis of the heatmap.
    samples_theta / samples_distance:
        Number of discrete samples for the respective axes. A minimum of two
        samples is enforced so the grid has meaningful area.
    parameters:
        Optional :class:`SpeedAdjustmentParameters` instance or mapping used to
        configure the constraint gains. When omitted, the defaults from the
        dataclass are applied.
    v_e, omega_e, d_o, v_max, omega_max:
        Additional arguments forwarded to :func:`adjust_velocity` when
        computing the attenuation field.

    Returns
    -------
    tuple(matplotlib.figure.Figure, matplotlib.axes.Axes)
        The figure and axis containing the rendered heatmap. The function keeps
        matplotlib imports local so that the visualisation dependencies remain
        optional for consumers who only require the numerical helper.
    """

    import matplotlib.pyplot as plt

    if samples_theta < 2 or samples_distance < 2:
        raise ValueError("At least two samples are required for each axis.")

    if parameters is None:
        params = SpeedAdjustmentParameters()
    elif isinstance(parameters, SpeedAdjustmentParameters):
        params = parameters
    else:
        params = SpeedAdjustmentParameters.from_mapping(parameters)

    theta_min, theta_max = theta_range
    distance_min, distance_max = distance_range

    theta_step = (theta_max - theta_min) / (samples_theta - 1)
    distance_step = (distance_max - distance_min) / (samples_distance - 1)

    theta_values = [theta_min + idx * theta_step for idx in range(samples_theta)]
    distance_values = [
        distance_min + idx * distance_step for idx in range(samples_distance)
    ]

    attenuation_grid = []
    vmax_safe = v_max if v_max > 0.0 else 1.0

    for distance in distance_values:
        row = []
        for theta in theta_values:
            v_adjusted, _ = adjust_velocity(
                v_e=v_e,
                omega_e=omega_e,
                d_f=distance,
                d_o=d_o,
                theta_f=theta,
                eta_theta=params.eta_theta,
                eta_d=params.eta_d,
                k_d=params.k_d,
                omega_fixed=params.omega_fixed,
                t_df=params.t_df,
                v_max=v_max,
                omega_max=omega_max,
            )
            row.append(v_adjusted / vmax_safe)
        attenuation_grid.append(row)

    fig, ax = plt.subplots()
    heatmap = ax.imshow(
        attenuation_grid,
        origin="lower",
        extent=(theta_min, theta_max, distance_min, distance_max),
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )

    ax.set_xlabel("Heading error θ_f [rad]")
    ax.set_ylabel("Formation distance d_f [m]")
    ax.set_title("Speed adjustment attenuation (v_a / v_max)")
    fig.colorbar(heatmap, ax=ax, label="Speed ratio")

    return fig, ax


__all__ = [
    "SpeedAdjustmentParameters",
    "adjust_velocity",
    "plot_speed_adjustment_heatmap",
]

