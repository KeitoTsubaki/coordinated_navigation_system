"""Helpers for the renderer's formation-aware velocity adjustment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Tuple


def _clip(value: float, lower: float, upper: float) -> float:
    """Lightweight alternative to :func:`numpy.clip`."""

    return min(max(value, lower), upper)


def _compute_attenuation_terms(
    *,
    theta_f: float,
    d_f: float,
    eta_theta: float,
    eta_d: float,
    k_d: float,
    t_df: float,
) -> tuple[float, float, float]:
    """Return the attenuation components ``(l_theta, l_d, l_t)``.

    The helper encapsulates the shared maths between :func:`adjust_velocity` and
    :func:`plot_speed_adjustment_heatmap`. Keeping the implementation in a
    single place avoids subtle inconsistencies when tweaking the formation
    control gains or the mathematical safeguards around the edge cases (for
    instance zero distances or invalid configuration values).
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
    return l_theta, l_d, l_t


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
    target_is_inner: bool,
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
    ``omega_a = K[omega_e + min(multiplier * |d_o|, 1) * (omega_f - omega_e)]``
    with ``K``
    implemented via simple saturation to the platform limits ``[0, V_MAX]`` and
    ``[-OMEGA_MAX, OMEGA_MAX]``. ``omega_f`` uses a fixed magnitude (defaulting
    to ``0.4π``) and follows the sign of ``theta_f`` so the robot turns at a
    consistent speed toward the formation target regardless of the heading
    error. ``target_is_inner`` toggles the lateral blending gain between inner
    and outer targets, tightening the turn when the assignment lies within the
    formation hull—typically when the target is at least as close to the
    formation centre as the robot itself.
    """

    _, _, l_t = _compute_attenuation_terms(
        theta_f=theta_f,
        d_f=d_f,
        eta_theta=eta_theta,
        eta_d=eta_d,
        k_d=k_d,
        t_df=t_df,
    )

    v_a = float(_clip(v_e * (1.0 - l_t), 0.0, v_max))

    omega_magnitude = abs(omega_fixed)
    if omega_magnitude == 0.0 or theta_f == 0.0:
        omega_f = 0.0
    else:
        omega_f = math.copysign(omega_magnitude, theta_f)
    omega_f = float(_clip(omega_f, -omega_max, omega_max))
    # print(omega_f)
    multiplier = 3.0 if target_is_inner else 2.0
    blend = min(abs(multiplier * d_o), 1.0)
    # print(blend)
    omega_target = omega_e + blend * (omega_f - omega_e)
    omega_a = float(_clip(omega_target, -omega_max, omega_max))
    return v_a, omega_a


def plot_speed_adjustment_heatmap(
    x_range: tuple[float, float] = (-2.0, 2.0),
    y_range: tuple[float, float] = (-2.0, 2.0),
    *,
    samples_x: int = 200,
    samples_y: int = 200,
    parameters: SpeedAdjustmentParameters | Mapping[str, float] | None = None,
):
    """Visualise the attenuation applied by :func:`adjust_velocity`.

    The helper samples the provided ``x`` and ``y`` ranges on a regular grid and
    evaluates the attenuation term ``l_t`` derived from :func:`adjust_velocity`
    at each location. The resulting array is rendered as a heatmap using
    matplotlib where warm colours indicate strong reduction of the commanded
    speed and cooler regions correspond to negligible attenuation.

    Parameters
    ----------
    x_range, y_range:
        Inclusive ranges of the sampled workspace (in metres) used for the
        horizontal and vertical axes of the heatmap respectively.
    samples_x / samples_y:
        Number of discrete samples for the respective axes. A minimum of two
        samples is enforced so the grid has meaningful area.
    parameters:
        Optional :class:`SpeedAdjustmentParameters` instance or mapping used to
        configure the constraint gains. When omitted, the defaults from the
        dataclass are applied.

    Returns
    -------
    tuple(matplotlib.figure.Figure, matplotlib.axes.Axes)
        The figure and axis containing the rendered heatmap. The function keeps
        matplotlib imports local so that the visualisation dependencies remain
        optional for consumers who only require the numerical helper.
    """

    import matplotlib.pyplot as plt

    if samples_x < 2 or samples_y < 2:
        raise ValueError("At least two samples are required for each axis.")

    if parameters is None:
        params = SpeedAdjustmentParameters()
    elif isinstance(parameters, SpeedAdjustmentParameters):
        params = parameters
    else:
        params = SpeedAdjustmentParameters.from_mapping(parameters)

    x_min, x_max = x_range
    y_min, y_max = y_range

    x_step = (x_max - x_min) / (samples_x - 1)
    y_step = (y_max - y_min) / (samples_y - 1)

    x_values = [x_min + idx * x_step for idx in range(samples_x)]
    y_values = [y_min + idx * y_step for idx in range(samples_y)]

    attenuation_grid = []

    for y in y_values:
        row = []
        for x in x_values:
            theta_f = math.atan2(y, x)
            d_f = math.hypot(x, y)
            _, _, l_t = _compute_attenuation_terms(
                theta_f=theta_f,
                d_f=d_f,
                eta_theta=params.eta_theta,
                eta_d=params.eta_d,
                k_d=params.k_d,
                t_df=params.t_df,
            )
            row.append(l_t)
        attenuation_grid.append(row)

    fig, ax = plt.subplots()
    heatmap = ax.imshow(
        attenuation_grid,
        origin="lower",
        extent=(x_min, x_max, y_min, y_max),
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )

    ax.set_xlabel("x position [m]")
    ax.set_ylabel("y position [m]")
    ax.set_title("Attenuation term l_t over the workspace")
    fig.colorbar(heatmap, ax=ax, label="l_t")

    plt.show()

    return fig, ax


if __name__ == "__main__":
    plot_speed_adjustment_heatmap()

__all__ = [
    "SpeedAdjustmentParameters",
    "adjust_velocity",
    "plot_speed_adjustment_heatmap",
]

