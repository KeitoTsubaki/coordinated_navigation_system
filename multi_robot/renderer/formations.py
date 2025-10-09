"""Robot placement helpers for spawning and regrouping."""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from .state import Position, RobotState


def recenter_robots(simulator: "MultiRobotSimulator") -> None:
    """Move all robots to fresh spawn points and clear their history."""

    if not simulator.robot_states:
        return

    start_spawn_area, target_spawn_area = get_spawn_areas(simulator)

    spacing = simulator.env.ROBOT_RADIUS_PIXELS * 2.5
    robot_positions = generate_randomized_formation_positions(
        simulator, start_spawn_area, len(simulator.robot_states), spacing
    )

    for robot, position in zip(simulator.robot_states, robot_positions):
        robot.position = (float(position[0]), float(position[1]))
        robot.theta = 0.0
        robot.path.clear()
        robot.sensor_data = []
        robot.closest_obstacle = (float("inf"), 0.0)
        robot.last_update_time = None
        robot.velocity = 0.0

    assign_random_targets(simulator, target_spawn_area)


def get_spawn_areas(simulator: "MultiRobotSimulator") -> Tuple[Optional[Sequence[float]], Optional[Sequence[float]]]:
    """Return spawn areas defined by the current map generator."""

    screen_center_xy = (simulator.env.SCREEN_CENTER_X, simulator.env.SCREEN_CENTER_Y)
    screen_size_wh = (simulator.env.SCREEN_W, simulator.env.SCREEN_H)
    start_spawn_area, target_spawn_area = simulator.env.map_creator.initialize_map_parameters(
        simulator.env.map_type,
        screen_center_xy,
        screen_size_wh,
        simulator.env.ROBOT_RADIUS_PIXELS,
    )
    return start_spawn_area, target_spawn_area


def assign_random_targets(
    simulator: "MultiRobotSimulator", target_spawn_area: Optional[Sequence[float]]
) -> None:
    """Assign formation positions inside ``target_spawn_area`` as targets."""

    if not simulator.robot_states:
        return

    spacing = simulator.env.ROBOT_RADIUS_PIXELS * 2.5
    target_positions = generate_randomized_formation_positions(
        simulator, target_spawn_area, len(simulator.robot_states), spacing
    )

    if not target_positions:
        return

    for robot, target in zip(simulator.robot_states, target_positions):
        robot.target = target


def compute_formation_layout(
    simulator: "MultiRobotSimulator",
    count: int,
    spawn_area: Optional[Sequence[float]],
    spacing: float,
) -> Tuple[int, int]:
    """Return rows and columns for a rectangular formation."""

    if count <= 0:
        return 0, 0

    if spawn_area is None:
        columns = max(1, math.ceil(math.sqrt(count)))
        rows = math.ceil(count / columns)
    else:
        min_x, max_x, min_y, max_y = spawn_area
        available_width = max(0.0, max_x - min_x)
        available_height = max(0.0, max_y - min_y)

        max_columns = max(1, int(available_width // spacing) + 1)
        max_rows = max(1, int(available_height // spacing) + 1)

        columns = min(count, max_columns)
        rows = math.ceil(count / columns)

        while rows > max_rows and columns > 1:
            columns -= 1
            rows = math.ceil(count / columns)

        rows = max(rows, 1)

    return rows, columns


def sample_formation_center(
    simulator: "MultiRobotSimulator",
    spawn_area: Optional[Sequence[float]],
    spacing: float,
    rows: int,
    columns: int,
) -> Position:
    """Sample a centre point for a potential formation within the spawn area."""

    if spawn_area is None:
        min_x = float(simulator.env.ROBOT_RADIUS_PIXELS)
        max_x = float(simulator.env.SCREEN_W - simulator.env.ROBOT_RADIUS_PIXELS)
        min_y = float(simulator.env.ROBOT_RADIUS_PIXELS)
        max_y = float(simulator.env.SCREEN_H - simulator.env.ROBOT_RADIUS_PIXELS)
    else:
        min_x, max_x, min_y, max_y = spawn_area

    half_width = spacing * max(columns - 1, 0) / 2.0
    half_height = spacing * max(rows - 1, 0) / 2.0

    margin_x = half_width + simulator.robot_frame_radius
    margin_y = half_height + simulator.robot_frame_radius

    allowed_min_x = min_x + margin_x
    allowed_max_x = max_x - margin_x
    allowed_min_y = min_y + margin_y
    allowed_max_y = max_y - margin_y

    if allowed_min_x > allowed_max_x:
        center_x = (min_x + max_x) / 2.0
    else:
        center_x = float(simulator.env.np_random.uniform(allowed_min_x, allowed_max_x))

    if allowed_min_y > allowed_max_y:
        center_y = (min_y + max_y) / 2.0
    else:
        center_y = float(simulator.env.np_random.uniform(allowed_min_y, allowed_max_y))

    return center_x, center_y


def generate_randomized_formation_positions(
    simulator: "MultiRobotSimulator",
    spawn_area: Optional[Sequence[float]],
    count: int,
    spacing: float,
) -> List[Position]:
    """Sample formation positions, retrying until a collision free set is found."""

    if count <= 0:
        return []

    rows, columns = compute_formation_layout(simulator, count, spawn_area, spacing)
    if rows == 0 or columns == 0:
        return []

    max_attempts = 32 if spawn_area is not None else 8
    fallback_positions: Optional[List[Position]] = None

    for _ in range(max_attempts):
        center = sample_formation_center(simulator, spawn_area, spacing, rows, columns)
        positions = generate_formation_positions(
            simulator,
            spawn_area,
            count,
            spacing,
            rows=rows,
            columns=columns,
            center=center,
            enforce_bounds=False,
        )

        if are_spawn_positions_valid(simulator, positions, spawn_area):
            return positions

        if fallback_positions is None:
            fallback_positions = positions

    enforced_positions = generate_formation_positions(
        simulator,
        spawn_area,
        count,
        spacing,
        rows=rows,
        columns=columns,
        center=None,
        enforce_bounds=True,
    )

    if are_spawn_positions_valid(simulator, enforced_positions, spawn_area):
        return enforced_positions

    if fallback_positions is not None:
        return fallback_positions

    return enforced_positions


def generate_formation_positions(
    simulator: "MultiRobotSimulator",
    spawn_area: Optional[Sequence[float]],
    count: int,
    spacing: float,
    *,
    rows: Optional[int] = None,
    columns: Optional[int] = None,
    center: Optional[Position] = None,
    enforce_bounds: bool = True,
) -> List[Position]:
    """Create a rectangular grid of formation positions."""

    if count <= 0:
        return []

    if spawn_area is None:
        min_x = float(simulator.env.SCREEN_CENTER_X)
        max_x = min_x
        min_y = float(simulator.env.SCREEN_CENTER_Y)
        max_y = min_y
    else:
        min_x, max_x, min_y, max_y = spawn_area

    if rows is None or columns is None:
        rows, columns = compute_formation_layout(simulator, count, spawn_area, spacing)

    if center is None:
        if spawn_area is None:
            center_x = simulator.env.SCREEN_CENTER_X
            center_y = simulator.env.SCREEN_CENTER_Y
        else:
            center_x = (min_x + max_x) / 2.0
            center_y = (min_y + max_y) / 2.0
    else:
        center_x, center_y = center

    formation_positions: List[Position] = []
    row_offsets = [(row - (rows - 1) / 2.0) * spacing for row in range(rows)]
    column_offsets = [(col - (columns - 1) / 2.0) * spacing for col in range(columns)]

    idx = 0
    for row_offset in row_offsets:
        for col_offset in column_offsets:
            if idx >= count:
                break
            x = center_x + col_offset
            y = center_y + row_offset

            if enforce_bounds and spawn_area is not None:
                x = min(
                    max(x, min_x + simulator.robot_frame_radius),
                    max_x - simulator.robot_frame_radius,
                )
                y = min(
                    max(y, min_y + simulator.robot_frame_radius),
                    max_y - simulator.robot_frame_radius,
                )

            formation_positions.append((float(x), float(y)))
            idx += 1
        if idx >= count:
            break

    return formation_positions


def are_spawn_positions_valid(
    simulator: "MultiRobotSimulator",
    positions: Sequence[Position],
    spawn_area: Optional[Sequence[float]],
) -> bool:
    """Return ``True`` if ``positions`` lie within ``spawn_area`` and avoid collisions."""

    if not positions:
        return True

    if spawn_area is None:
        min_x = float(simulator.env.ROBOT_RADIUS_PIXELS)
        max_x = float(simulator.env.SCREEN_W - simulator.env.ROBOT_RADIUS_PIXELS)
        min_y = float(simulator.env.ROBOT_RADIUS_PIXELS)
        max_y = float(simulator.env.SCREEN_H - simulator.env.ROBOT_RADIUS_PIXELS)
    else:
        min_x, max_x, min_y, max_y = spawn_area

    for x, y in positions:
        if not (min_x <= x <= max_x and min_y <= y <= max_y):
            return False
        if simulator.env.is_collision((x, y)):
            return False

    return True


__all__ = [
    "recenter_robots",
    "get_spawn_areas",
    "assign_random_targets",
    "compute_formation_layout",
    "sample_formation_center",
    "generate_randomized_formation_positions",
    "generate_formation_positions",
    "are_spawn_positions_valid",
]
