"""Formation planning and visualisation helpers."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pygame
from scipy.optimize import linear_sum_assignment, minimize, minimize_scalar

from .types import Color, FormationPlan, Position
from .utils import brighten_color, with_alpha

if False:  # pragma: no cover - type checking helper
    from .simulator import MultiRobotSimulator


@dataclass(frozen=True)
class _RobotPlanCandidate:
    """Intermediate representation for per-robot formation options."""

    theta: float
    target_index: int
    cost: float
    target: Position


class FormationManager:
    """Encapsulates formation optimisation and related rendering."""

    def __init__(self, simulator: "MultiRobotSimulator") -> None:
        self.simulator = simulator
        self.current_plan: Optional[FormationPlan] = None
        self.plan_candidates: Dict[int, FormationPlan] = {}
        self._direct_targets: Dict[int, Optional[Position]] = {}
        self._direct_target_orientations: Dict[int, Optional[float]] = {}
        self._formation_targets: Dict[int, Position] = {}
        self._per_robot_candidates: Dict[int, List["_RobotPlanCandidate"]] = {}
        self.follow_formation_targets: bool = True
        self._enforce_vertical_constraint: bool = bool(
            getattr(simulator, "formation_vertical_constraint_enabled", True)
        )
        simulator.formation_vertical_constraint_enabled = self._enforce_vertical_constraint
        self._centroid_sampling_interval = max(
            0.0, float(getattr(simulator, "formation_centroid_sampling_interval", 0.0))
        )
        self._centroid_look_back_steps = max(
            0, int(getattr(simulator, "formation_centroid_look_back_steps", 0))
        )
        self._centroid_min_displacement_pixels = max(
            0.0,
            float(
                getattr(
                    simulator,
                    "formation_centroid_min_displacement_pixels",
                    getattr(simulator.env, "FORMATION_MIN_DISPLACEMENT_PIXELS", 0.0),
                )
            ),
        )
        self._centroid_min_displacement_m = max(
            0.0,
            float(
                getattr(
                    simulator,
                    "formation_centroid_min_displacement_m",
                    getattr(simulator.env, "FORMATION_MIN_DISPLACEMENT_M", 0.0),
                )
            ),
        )
        history_limit = max(1, self._centroid_look_back_steps + 1)
        self._centroid_history: Deque[Tuple[float, Position]] = deque(maxlen=history_limit)
        self._last_centroid_sample_time: Optional[float] = None
        self._last_centroid_progress: Optional[Tuple[float, float, float]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def _reset_plan_state(self) -> None:
        """Clear plan-related caches without disturbing centroid tracking."""

        self.current_plan = None
        self.plan_candidates.clear()
        self._formation_targets.clear()
        self._per_robot_candidates.clear()

    def clear(self) -> None:
        self._reset_plan_state()
        self._centroid_history.clear()
        self._last_centroid_sample_time = None
        self._last_centroid_progress = None

    def _record_centroid(self, center: Position) -> None:
        if not math.isfinite(center[0]) or not math.isfinite(center[1]):
            return

        now = pygame.time.get_ticks() / 1000.0
        interval = self._centroid_sampling_interval
        last_time = self._last_centroid_sample_time

        should_record = False
        if interval <= 0.0:
            should_record = True
        elif last_time is None:
            should_record = True
        elif now - last_time >= interval:
            should_record = True

        if not should_record:
            return

        self._centroid_history.append((now, (float(center[0]), float(center[1]))))
        self._last_centroid_sample_time = now

    def _centroid_progress_vector(self) -> Optional[Tuple[float, float, float]]:
        steps = self._centroid_look_back_steps
        cached_progress = self._last_centroid_progress
        if steps <= 0:
            return cached_progress

        if len(self._centroid_history) <= steps:
            return cached_progress

        _, current_pos = self._centroid_history[-1]
        _, lookback_pos = self._centroid_history[-1 - steps]
        dx = float(current_pos[0] - lookback_pos[0])
        dy = float(current_pos[1] - lookback_pos[1])
        displacement = math.hypot(dx, dy)

        if displacement < self._centroid_min_displacement_pixels:
            return cached_progress

        progress_vector = (dx, dy, displacement)
        self._last_centroid_progress = progress_vector
        return progress_vector

    def update_visualisation(self) -> None:
        simulator = self.simulator
        if not simulator.robot_states:
            self.clear()
            return

        if not self.follow_formation_targets:
            self._enforce_direct_targets()
            self._formation_targets.clear()
        self.plan_candidates.clear()
        self._per_robot_candidates.clear()
        selected_plan: Optional[FormationPlan] = None

        for ego_index in range(len(simulator.robot_states)):
            try:
                plan = self.optimize(ego_index=ego_index, apply=False)
            except ValueError:
                continue

            self.plan_candidates[ego_index] = plan
            if ego_index == simulator.selected_robot_index:
                selected_plan = plan

        if not self.plan_candidates:
            self.current_plan = None
            if self.follow_formation_targets:
                self._formation_targets.clear()
                self._per_robot_candidates.clear()
                self._enforce_direct_targets()
            return

        if selected_plan is not None:
            self.current_plan = selected_plan
        else:
            self.current_plan = next(iter(self.plan_candidates.values()))

        aggregated_candidates = self._aggregate_plan_candidates(
            self.plan_candidates.values()
        )
        self._per_robot_candidates = aggregated_candidates

        if self.follow_formation_targets:
            if aggregated_candidates:
                applied = self._apply_plan_targets(candidates=aggregated_candidates)
                if not applied:
                    self._enforce_direct_targets()
            else:
                self._formation_targets.clear()
                self._enforce_direct_targets()

    def draw(self) -> None:
        if not self.follow_formation_targets:
            # When robots are travelling directly towards their individual
            # navigation goals, omit the formation overlay so that only the
            # direct targets remain visible.
            self._draw_manual_target_overlay()
            return

        if not self.plan_candidates:
            self._draw_manual_target_overlay()
            return

        simulator = self.simulator
        layer = simulator.env.robot_and_lidar_layer
        target_radius = int(max(4, simulator.robot_radius_pixels * 0.4))
        font = pygame.font.SysFont("arial", 14)
        small_font = pygame.font.SysFont("consolas", 12)

        highlight_queue: List[Tuple[int, FormationPlan, Color]] = []

        for ego_index in sorted(self.plan_candidates):
            plan = self.plan_candidates[ego_index]
            if not plan.targets or not simulator.robot_states:
                continue
            if len(plan.targets) != len(simulator.robot_states):
                continue

            if 0 <= ego_index < len(simulator.robot_states):
                base_color = simulator.robot_states[ego_index].color
            else:
                base_color = (72, 61, 139)

            highlight = plan is self.current_plan
            if highlight:
                highlight_queue.append((ego_index, plan, base_color))
                continue

            self._draw_single_plan(
                plan,
                ego_index=ego_index,
                target_radius=target_radius,
                base_color=base_color,
                highlight=False,
                font=font,
                small_font=small_font,
            )

        for ego_index, plan, base_color in highlight_queue:
            self._draw_single_plan(
                plan,
                ego_index=ego_index,
                target_radius=target_radius,
                base_color=base_color,
                highlight=True,
                font=font,
                small_font=small_font,
            )

        self._draw_centroid_history()

    def draw_parallel_overlay(self) -> None:
        simulator = self.simulator
        if len(simulator.robot_states) < 2:
            return

        center_color = (255, 140, 0)
        text_color = (25, 25, 25)
        font = pygame.font.SysFont("arial", 14)

        sorted_states = sorted(simulator.robot_states, key=lambda robot: robot.position[0])
        positions = [robot.position for robot in sorted_states]

        center_x = sum(position[0] for position in positions) / len(positions)
        center_y = sum(position[1] for position in positions) / len(positions)
        pygame.draw.circle(
            simulator.env.robot_and_lidar_layer,
            center_color,
            (int(center_x), int(center_y)),
            6,
        )

        debug_text = simulator.show_debug_text

        for first, second in zip(sorted_states[:-1], sorted_states[1:]):
            start = first.position
            end = second.position

            dx = end[0] - start[0]
            dy = end[1] - start[1]
            distance_pixels = math.hypot(dx, dy)
            if distance_pixels == 0:
                continue

            midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
            normal_x = -dy / distance_pixels
            normal_y = dx / distance_pixels

            if not debug_text:
                continue

            distance_m = distance_pixels / simulator.env.PIXELS_PER_METER
            distance_label = font.render(f"{distance_m:.2f}m", True, text_color)

            angle_rad = math.atan2(-(dy), dx)
            angle_deg = (math.degrees(angle_rad) + 360.0) % 360.0
            angle_label = font.render(f"{angle_deg:.1f}°", True, text_color)

            offset = 12
            distance_position = (
                int(midpoint[0] + normal_x * offset - distance_label.get_width() / 2),
                int(midpoint[1] + normal_y * offset - distance_label.get_height() / 2),
            )
            angle_position = (
                int(midpoint[0] - normal_x * offset - angle_label.get_width() / 2),
                int(midpoint[1] - normal_y * offset - angle_label.get_height() / 2),
            )

            simulator.env.robot_and_lidar_layer.blit(distance_label, distance_position)
            simulator.env.robot_and_lidar_layer.blit(angle_label, angle_position)

    def optimize(
        self,
        formation_type: Optional[str] = None,
        *,
        weights: Optional[Sequence[float]] = None,
        ego_index: Optional[int] = None,
        spacing: Optional[float] = None,
        apply: bool = True,
    ) -> FormationPlan:
        simulator = self.simulator
        if not simulator.robot_states:
            raise ValueError("No robots available for formation optimisation")

        if formation_type is None:
            formation_type = simulator.default_formation_type

        if spacing is None:
            spacing = simulator.default_formation_spacing

        center_x = sum(robot.position[0] for robot in simulator.robot_states) / len(
            simulator.robot_states
        )
        center_y = sum(robot.position[1] for robot in simulator.robot_states) / len(
            simulator.robot_states
        )
        current_center = (center_x, center_y)
        center = self._refine_formation_center(
            current_center, formation_type, spacing
        )
        self._record_centroid(current_center)

        weights = weights or (0.5, 0.5, 0.5, 0.3, 1.0)
        ego_index = ego_index if ego_index is not None else simulator.selected_robot_index
        if ego_index is not None and not (0 <= ego_index < len(simulator.robot_states)):
            raise IndexError("ego_index is out of range")

        angle_min, angle_max = self._formation_angle_bounds(formation_type)
        penalty = 1e6
        self._prune_direct_target_cache()
        best_plan: Dict[str, Optional[FormationPlan]] = {"plan": None}

        def objective(theta: float) -> float:
            plan = self._evaluate_plan(
                center,
                theta,
                formation_type,
                spacing,
                weights,
                ego_index,
                penalty,
            )
            if plan is None:
                return penalty
            if best_plan["plan"] is None or plan.cost < best_plan["plan"].cost:
                best_plan["plan"] = plan
            return plan.cost

        result = minimize_scalar(objective, bounds=(angle_min, angle_max), method="bounded")
        objective(float(result.x))
        objective(angle_min)
        objective(angle_max)

        plan = best_plan["plan"]
        if plan is None:
            self._reset_plan_state()
            raise ValueError("Unable to find a feasible formation configuration")

        if apply:
            plans_for_selection: List[FormationPlan]
            if ego_index is not None:
                self.plan_candidates[ego_index] = plan
                plans_for_selection = list(self.plan_candidates.values())
            else:
                plans_for_selection = list(self.plan_candidates.values()) + [plan]

            if self.follow_formation_targets:
                candidates = self._aggregate_plan_candidates(plans_for_selection)
                applied = self._apply_plan_targets(candidates=candidates)
                if not applied:
                    self._enforce_direct_targets()
            else:
                self._enforce_direct_targets()
            self.current_plan = plan

        return plan

    def _evaluate_center_candidate(
        self,
        candidate: Position,
        *,
        current_center: Position,
        team_radius: float,
        search_radius: float,
    ) -> Tuple[float, bool]:
        """Return the cost for *candidate* and whether it is feasible."""

        simulator = self.simulator
        dx = float(candidate[0] - current_center[0])
        dy = float(candidate[1] - current_center[1])
        displacement = math.hypot(dx, dy)
        if displacement > search_radius + 1e-6:
            return float("inf"), False

        # print("distance: ", simulator._distance_to_nearest_obstacle_map(candidate), "team_radius: ", team_radius)
        clearance = simulator._distance_to_nearest_obstacle_map(candidate) - team_radius
        if clearance <= 0.0:
            return float("inf"), False
        
        clearance_weight = 1.0
        displacement_weight = 1.0

        # scaled_clearance_cost = clearance_weight / max(clearance, 1e-6)
        scaled_clearance_cost = clearance_weight * max(3 - clearance, 0)
        scaled_displacement_cost = displacement_weight * (
            displacement / max(search_radius, 1e-6)
        )
        total_cost = scaled_clearance_cost + scaled_displacement_cost
        # total_cost = scaled_clearance_cost
        return total_cost, True

    def _refine_formation_center(
        self, current_center: Position, formation_type: str, spacing: float
    ) -> Position:
        """Search for a collision-aware formation center near *current_center*."""

        simulator = self.simulator
        search_radius = float(getattr(simulator, "robot_radius_pixels", 0.0))
        if search_radius <= 0.0:
            return current_center

        robot_count = len(simulator.robot_states)
        offsets = self._formation_offsets(robot_count, formation_type, spacing)
        if not offsets:
            return current_center

        max_offset_radius = max(math.hypot(x, y) for x, y in offsets) if offsets else 0.0
        # frame_radius = float(getattr(simulator, "robot_frame_radius", 0.0))
        # team_radius = max_offset_radius + frame_radius
        robot_radius_m = float(getattr(simulator.env, "ROBOT_RADIUS_M", 0.0))
        team_radius = simulator.default_formation_spacing_m/2  + robot_radius_m
        if team_radius <= 0.0:
            team_radius = max_offset_radius
        if team_radius <= 0.0:
            return current_center

        best_center: Optional[Position] = None
        best_cost = float("inf")

        def consider(candidate: Position) -> None:
            nonlocal best_center, best_cost
            cost, feasible = self._evaluate_center_candidate(
                candidate,
                current_center=current_center,
                team_radius=team_radius,
                search_radius=search_radius,
            )
            # print("feasible: ", feasible, "cost: ", cost)
            if feasible and cost < best_cost:
                best_center = (float(candidate[0]), float(candidate[1]))
                best_cost = cost

        consider(current_center)

        radial_steps = max(1, int(getattr(simulator, "formation_center_radial_steps", 4)))
        angular_steps = max(6, int(getattr(simulator, "formation_center_angular_steps", 16)))

        for radial_index in range(1, radial_steps + 1):
            radius = search_radius * (radial_index / radial_steps)
            for angular_index in range(angular_steps):
                angle = 2.0 * math.pi * (angular_index / angular_steps)
                candidate = (
                    current_center[0] + radius * math.cos(angle),
                    current_center[1] + radius * math.sin(angle),
                )
                consider(candidate)

        if best_center is None:
            # print("best centroid is None")
            return current_center

        large_penalty = 1e6

        def objective(delta: np.ndarray) -> float:
            candidate = (
                current_center[0] + float(delta[0]),
                current_center[1] + float(delta[1]),
            )
            cost, feasible = self._evaluate_center_candidate(
                candidate,
                current_center=current_center,
                team_radius=team_radius,
                search_radius=search_radius,
            )
            return cost if feasible else large_penalty

        initial_delta = np.array(
            [best_center[0] - current_center[0], best_center[1] - current_center[1]],
            dtype=float,
        )

        constraints = (
            {
                "type": "ineq",
                "fun": lambda delta: search_radius**2
                - (delta[0] ** 2 + delta[1] ** 2),
            },
        )

        try:
            result = minimize(
                objective,
                initial_delta,
                method="SLSQP",
                constraints=constraints,
            )
        except Exception:
            result = None

        if result is not None and getattr(result, "success", False):
            refined_candidate = (
                current_center[0] + float(result.x[0]),
                current_center[1] + float(result.x[1]),
            )
            cost, feasible = self._evaluate_center_candidate(
                refined_candidate,
                current_center=current_center,
                team_radius=team_radius,
                search_radius=search_radius,
            )
            if feasible and cost < best_cost:
                best_center = refined_candidate
                best_cost = cost

        if best_center is None:
            return current_center
        # print("best centor is found")
        return best_center

    def register_direct_target(
        self,
        robot_index: int,
        target: Optional[Position],
        target_theta: Optional[float] = None,
    ) -> None:
        if target is None:
            self._direct_targets[robot_index] = None
            self._direct_target_orientations[robot_index] = None
            return

        self._direct_targets[robot_index] = (float(target[0]), float(target[1]))
        self._direct_target_orientations[robot_index] = (
            None if target_theta is None else float(target_theta)
        )
        self._formation_targets.pop(robot_index, None)

    def _prune_direct_target_cache(self) -> None:
        robot_count = len(self.simulator.robot_states)
        stale_indices = [index for index in self._direct_targets if index >= robot_count]
        for index in stale_indices:
            del self._direct_targets[index]
            if index in self._direct_target_orientations:
                del self._direct_target_orientations[index]

    def _prune_formation_target_cache(self) -> None:
        """Drop cached formation targets for robots that no longer exist."""

        robot_count = len(self.simulator.robot_states)
        stale_indices = [index for index in self._formation_targets if index >= robot_count]
        for index in stale_indices:
            del self._formation_targets[index]

    def _enforce_direct_targets(self) -> None:
        self._prune_direct_target_cache()
        self._prune_formation_target_cache()
        self._formation_targets.clear()
        for index, robot in enumerate(self.simulator.robot_states):
            if index not in self._direct_targets:
                continue

            direct_target = self._direct_targets[index]

            if direct_target is None:
                if robot.target is not None:
                    robot.target = None
                robot.target_theta = None
                continue

            if robot.target is None or (
                abs(robot.target[0] - direct_target[0]) > 1e-6
                or abs(robot.target[1] - direct_target[1]) > 1e-6
            ):
                robot.target = (float(direct_target[0]), float(direct_target[1]))
            orientation = self._direct_target_orientations.get(index)
            robot.target_theta = None if orientation is None else float(orientation)

    def toggle_follow_formation(self) -> bool:
        self.follow_formation_targets = not self.follow_formation_targets
        if self.follow_formation_targets:
            self.update_visualisation()
            if self.current_plan is None:
                self.follow_formation_targets = False
                self._enforce_direct_targets()
        else:
            self._enforce_direct_targets()
        return self.follow_formation_targets

    @property
    def vertical_constraint_enabled(self) -> bool:
        return self._enforce_vertical_constraint

    def set_vertical_constraint(self, enabled: bool) -> None:
        enabled_flag = bool(enabled)
        if self._enforce_vertical_constraint == enabled_flag:
            return

        self._enforce_vertical_constraint = enabled_flag
        self.simulator.formation_vertical_constraint_enabled = enabled_flag
        self.update_visualisation()

    def toggle_vertical_constraint(self) -> bool:
        self.set_vertical_constraint(not self._enforce_vertical_constraint)
        return self._enforce_vertical_constraint

    def _apply_plan_targets(
        self,
        plan: Optional[FormationPlan] = None,
        *,
        candidates: Optional[Dict[int, List[_RobotPlanCandidate]]] = None,
    ) -> bool:
        """Update ``_formation_targets`` using the best per-robot candidates.

        When *plan* is provided it will be merged into *candidates* (or the
        existing :attr:`plan_candidates`) before selecting targets.  The
        routine returns ``True`` when at least one assignment survived the
        clearance and visibility checks.
        """

        aggregated = candidates
        if aggregated is None:
            plans = list(self.plan_candidates.values())
            if plan is not None and plan not in plans:
                plans.append(plan)
            aggregated = self._aggregate_plan_candidates(plans)
        elif plan is not None:
            extra = self._aggregate_plan_candidates([plan])
            if extra:
                merged: Dict[int, List[_RobotPlanCandidate]] = {
                    index: list(items) for index, items in aggregated.items()
                }
                for index, items in extra.items():
                    merged.setdefault(index, []).extend(items)
                aggregated = merged

        self._per_robot_candidates = aggregated

        selected_targets = self._select_best_targets_for_robot(aggregated)
        if not selected_targets:
            self._formation_targets.clear()
            return False

        self._formation_targets = selected_targets
        # Ensure we never return a cached assignment for a robot that has since been removed.
        self._prune_formation_target_cache()
        return True

    def _aggregate_plan_candidates(
        self, plans: Iterable[FormationPlan]
    ) -> Dict[int, List[_RobotPlanCandidate]]:
        """Group candidate targets by robot using optimisation outputs."""

        aggregated: Dict[int, List[_RobotPlanCandidate]] = {}
        robot_count = len(self.simulator.robot_states)

        for plan in plans:
            if not plan.targets or not plan.assignment:
                continue

            limit = min(robot_count, len(plan.assignment))
            for robot_index in range(limit):
                target_index = plan.assignment[robot_index]
                if not (0 <= target_index < len(plan.targets)):
                    continue

                cost_value = plan.per_robot_costs.get(robot_index)
                if cost_value is None:
                    continue
                cost = float(cost_value)
                if not math.isfinite(cost):
                    continue

                target = plan.targets[target_index]
                candidate = _RobotPlanCandidate(
                    theta=float(plan.theta),
                    target_index=int(target_index),
                    cost=cost,
                    target=(float(target[0]), float(target[1])),
                )
                aggregated.setdefault(robot_index, []).append(candidate)

        return aggregated

    def _select_best_targets_for_robot(
        self, candidates: Dict[int, List[_RobotPlanCandidate]]
    ) -> Dict[int, Position]:
        """Return the best feasible target for each robot if available."""

        simulator = self.simulator
        selected: Dict[int, Position] = {}

        for robot_index, robot in enumerate(simulator.robot_states):
            feasible_candidates: List[_RobotPlanCandidate] = []
            for candidate in candidates.get(robot_index, []):
                if simulator._robot_line_blocked(robot, candidate.target):
                    continue

                clearance = simulator._robot_clearance(robot, candidate.target)
                if clearance <= 0.0:
                    continue

                feasible_candidates.append(candidate)

            if not feasible_candidates:
                continue

            best_candidate = feasible_candidates[0]
            for candidate in feasible_candidates[1:]:
                if candidate.cost < best_candidate.cost - 1e-9:
                    best_candidate = candidate
                    continue

                if math.isclose(candidate.cost, best_candidate.cost, rel_tol=1e-9, abs_tol=1e-9):
                    if candidate.theta < best_candidate.theta:
                        best_candidate = candidate

            selected[robot_index] = (
                float(best_candidate.target[0]),
                float(best_candidate.target[1]),
            )

        return selected

    def get_navigation_target(
        self,
        robot_index: int,
        *,
        fallback: Optional[Position] = None,
        prefer_formation: bool = True,
    ) -> Optional[Position]:
        """Return the navigation target for ``robot_index``.

        Parameters
        ----------
        robot_index:
            Index of the robot whose target should be queried.
        fallback:
            Direct per-robot navigation goal to fall back to when no formation
            target is available. If omitted, the current ``RobotState.target``
            is used when possible.
        prefer_formation:
            When ``True`` (the default) and follow mode is enabled, prefer the
            computed formation target over the direct navigation goal. Passing
            ``False`` forces the method to return the direct navigation goal
            even while formation following is active.
        """
        self._prune_formation_target_cache()
        if fallback is None and 0 <= robot_index < len(self.simulator.robot_states):
            fallback = self.simulator.robot_states[robot_index].target

        if prefer_formation and self.follow_formation_targets:
            if robot_index in self._formation_targets:
                target = self._formation_targets[robot_index]
                return (float(target[0]), float(target[1]))

        if fallback is None:
            return None

        return (float(fallback[0]), float(fallback[1]))

    def get_formation_target(self, robot_index: int) -> Optional[Position]:
        """Return the assigned formation target for ``robot_index`` if available."""

        # Keep cached assignments in sync with the current roster before reading from it.
        self._prune_formation_target_cache()
        target = self._formation_targets.get(robot_index)
        if target is None:
            return None

        return (float(target[0]), float(target[1]))

    def generate_randomized_positions(
        self,
        spawn_area: Optional[Sequence[float]],
        count: int,
        spacing: float,
    ) -> List[Position]:
        if count <= 0:
            return []

        rows, columns = self._compute_layout(count, spawn_area, spacing)
        if rows == 0 or columns == 0:
            return []

        max_attempts = 32 if spawn_area is not None else 8
        fallback_positions: Optional[List[Position]] = None

        for _ in range(max_attempts):
            center = self._sample_center(spawn_area, spacing, rows, columns)
            positions = self._generate_positions(
                spawn_area,
                count,
                spacing,
                rows=rows,
                columns=columns,
                center=center,
                enforce_bounds=False,
            )

            if self._are_spawn_positions_valid(positions, spawn_area):
                return positions

            if fallback_positions is None:
                fallback_positions = positions

        enforced_positions = self._generate_positions(
            spawn_area,
            count,
            spacing,
            rows=rows,
            columns=columns,
            center=None,
            enforce_bounds=True,
        )

        if self._are_spawn_positions_valid(enforced_positions, spawn_area):
            return enforced_positions

        if fallback_positions is not None:
            return fallback_positions

        return enforced_positions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _draw_single_plan(
        self,
        plan: FormationPlan,
        *,
        ego_index: int,
        target_radius: int,
        base_color: Color,
        highlight: bool,
        font: "pygame.font.Font",
        small_font: "pygame.font.Font",
    ) -> None:
        simulator = self.simulator
        layer = simulator.env.robot_and_lidar_layer

        connection_color = (
            base_color if highlight else with_alpha(base_color, 0.5)
        )
        target_color = (
            brighten_color(base_color)
            if highlight
            else with_alpha(brighten_color(base_color), 0.5)
        )
        center_color = (
            brighten_color(base_color)
            if highlight
            else with_alpha(brighten_color(base_color), 0.7)
        )
        text_color = (30, 30, 30) if highlight else (90, 90, 90)

        center_x, center_y = plan.center
        orientation_length = max(simulator.env.ROBOT_RADIUS_PIXELS * 3.0, plan.spacing)
        orientation_dx = math.cos(plan.theta) * orientation_length
        orientation_dy = math.sin(plan.theta) * orientation_length
        orientation_end = (center_x + orientation_dx, center_y + orientation_dy)

        radius = 6 if highlight else 4
        pygame.draw.circle(layer, center_color, (int(center_x), int(center_y)), radius)

        if highlight:
            label_text = f"θ={math.degrees(plan.theta):.1f}°"
        else:
            label_text = f"R{ego_index}: {math.degrees(plan.theta):.1f}°"
        angle_label = font.render(label_text, True, text_color)

        debug_text = simulator.show_debug_text

        for target_index, position in enumerate(plan.targets):
            draw_kwargs = {"width": 0 if highlight else 1}
            pygame.draw.circle(
                layer,
                target_color,
                (int(position[0]), int(position[1])),
                target_radius,
                **draw_kwargs,
            )

            if highlight and debug_text:
                label = small_font.render(f"T{target_index}", True, text_color)
                label_pos = (
                    int(position[0] - label.get_width() / 2),
                    int(position[1] - label.get_height() / 2),
                )
                layer.blit(label, label_pos)

        if not plan.assignment:
            return

        for robot_index, target_index in enumerate(plan.assignment):
            if not (0 <= target_index < len(plan.targets)):
                continue
            if robot_index >= len(simulator.robot_states):
                continue

            robot = simulator.robot_states[robot_index]
            target_position = plan.targets[target_index]
            pygame.draw.line(
                layer,
                connection_color,
                (int(robot.position[0]), int(robot.position[1])),
                (int(target_position[0]), int(target_position[1])),
                width=2 if highlight else 1,
            )

            if not highlight:
                continue

            midpoint = (
                (robot.position[0] + target_position[0]) / 2.0,
                (robot.position[1] + target_position[1]) / 2.0,
            )
            cost_value = plan.per_robot_costs.get(robot_index)
            if cost_value is not None and debug_text:
                cost_label = small_font.render(f"{cost_value:.2f}", True, text_color)
                layer.blit(
                    cost_label,
                    (
                        int(midpoint[0] - cost_label.get_width() / 2),
                        int(midpoint[1] - cost_label.get_height() / 2),
                    ),
                )

        marker_radius = max(2, target_radius - 2)
        robots_with_targets = []
        for index, robot in enumerate(simulator.robot_states):
            target = self.get_navigation_target(index, fallback=robot.target)
            if target is not None:
                robots_with_targets.append((index, robot, target))
        if not robots_with_targets:
            return

        label_targets = highlight and debug_text

        for index, robot, target in robots_with_targets:
            vivid_color = brighten_color(robot.color)
            if highlight:
                line_color = vivid_color
                marker_color = vivid_color
                line_width = 2
            else:
                line_color = with_alpha(vivid_color, 0.6)
                marker_color = with_alpha(vivid_color, 0.6)
                line_width = 1

            self._draw_robot_target_indicator(
                layer,
                robot,
                target,
                robot_index=index,
                line_color=line_color,
                marker_color=marker_color,
                line_width=line_width,
                marker_radius=marker_radius,
                show_label=label_targets,
                text_color=text_color,
                font=small_font,
            )

    def _draw_manual_target_overlay(self) -> None:
        simulator = self.simulator
        robots_with_targets = []
        for index, robot in enumerate(simulator.robot_states):
            target = self.get_navigation_target(index, fallback=robot.target)
            if target is not None:
                robots_with_targets.append((index, robot, target))
        if not robots_with_targets:
            self._draw_centroid_history()
            return

        layer = simulator.env.robot_and_lidar_layer
        text_color = (30, 30, 30)
        small_font = pygame.font.SysFont("consolas", 12)

        marker_radius = max(2, int(simulator.robot_radius_pixels * 0.35))
        debug_text = simulator.show_debug_text

        for index, robot, target in robots_with_targets:
            vivid_color = brighten_color(robot.color)
            line_color = with_alpha(vivid_color, 0.85)
            marker_color = vivid_color

            self._draw_robot_target_indicator(
                layer,
                robot,
                target,
                robot_index=index,
                line_color=line_color,
                marker_color=marker_color,
                line_width=2,
                marker_radius=marker_radius,
                show_label=debug_text,
                text_color=text_color,
                font=small_font,
            )

        self._draw_centroid_history()

    def _draw_centroid_history(self) -> None:
        if len(self._centroid_history) < 2:
            return

        simulator = self.simulator
        layer = simulator.env.robot_and_lidar_layer

        trail_color = with_alpha((65, 105, 225), 0.8)
        head_color = brighten_color((65, 105, 225))

        points = []
        latest_position: Optional[Position] = None
        for _timestamp, position in self._centroid_history:
            latest_position = position
            points.append((int(round(position[0])), int(round(position[1]))))

        if len(points) < 2 or latest_position is None:
            return

        pygame.draw.lines(layer, trail_color, False, points, width=2)

        start_x = float(latest_position[0])
        start_y = float(latest_position[1])
        start_point = (int(round(start_x)), int(round(start_y)))
        pygame.draw.circle(layer, head_color, start_point, 4, width=1)

        progress = self._centroid_progress_vector()
        if progress is None:
            return

        dx, dy, displacement = progress
        if displacement <= 0.0:
            return

        angle = math.atan2(dy, dx)
        arrow_length = max(20.0, min(80.0, displacement * 1.5))
        end_x = start_x + math.cos(angle) * arrow_length
        end_y = start_y + math.sin(angle) * arrow_length
        end_point = (int(round(end_x)), int(round(end_y)))

        direction_color = with_alpha((30, 144, 255), 0.95)
        pygame.draw.line(layer, direction_color, start_point, end_point, width=3)

        head_length = max(6.0, min(arrow_length * 0.35, 14.0))
        head_angle = math.radians(25)

        left_angle = angle + math.pi - head_angle
        right_angle = angle + math.pi + head_angle

        left_point = (
            int(round(end_x + math.cos(left_angle) * head_length)),
            int(round(end_y + math.sin(left_angle) * head_length)),
        )
        right_point = (
            int(round(end_x + math.cos(right_angle) * head_length)),
            int(round(end_y + math.sin(right_angle) * head_length)),
        )

        pygame.draw.line(layer, direction_color, end_point, left_point, width=2)
        pygame.draw.line(layer, direction_color, end_point, right_point, width=2)

        if not simulator.show_debug_text:
            return

        pixels_per_meter = getattr(simulator.env, "PIXELS_PER_METER", 1.0) or 1.0
        distance_m = displacement / float(pixels_per_meter)
        angle_deg = (math.degrees(angle) + 360.0) % 360.0

        font = pygame.font.SysFont("consolas", 12)
        label = font.render(f"{distance_m:.2f}m @ {angle_deg:.0f}°", True, (25, 25, 25))

        label_offset = 10
        label_position = (
            int(round(end_x + label_offset)),
            int(round(end_y + label_offset)),
        )
        layer.blit(label, label_position)

    def _draw_robot_target_indicator(
        self,
        layer: "pygame.Surface",
        robot,
        target: Optional[Position],
        *,
        robot_index: int,
        line_color: Color,
        marker_color: Color,
        line_width: int,
        marker_radius: int,
        show_label: bool,
        text_color: Color,
        font: "pygame.font.Font",
    ) -> None:
        if target is None:
            return

        start = (float(robot.position[0]), float(robot.position[1]))
        end = (float(target[0]), float(target[1]))

        marker_radius = max(1, int(marker_radius))
        line_width = max(1, int(line_width))

        self._draw_dashed_line(
            layer,
            line_color,
            start,
            end,
            width=line_width,
        )

        pygame.draw.circle(
            layer,
            marker_color,
            (int(end[0]), int(end[1])),
            marker_radius,
            width=1,
        )

        if not show_label:
            return

        midpoint = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
        distance_pixels = math.hypot(end[0] - start[0], end[1] - start[1])
        distance_m = distance_pixels / self.simulator.env.PIXELS_PER_METER
        label = font.render(
            f"R{robot_index} → target ({distance_m:.2f}m)", True, text_color
        )
        layer.blit(
            label,
            (
                int(midpoint[0] - label.get_width() / 2),
                int(midpoint[1] - label.get_height() / 2),
            ),
        )

    @staticmethod
    def _draw_dashed_line(
        surface: "pygame.Surface",
        color: Color,
        start: Position,
        end: Position,
        *,
        width: int = 1,
        dash_length: float = 8.0,
        gap_length: float = 4.0,
    ) -> None:
        dash_length = max(1.0, float(dash_length))
        gap_length = max(1.0, float(gap_length))

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.hypot(dx, dy)
        if distance < 1e-2:
            # Too short to meaningfully draw a dashed line; draw a single segment.
            pygame.draw.line(
                surface,
                color,
                (int(round(start[0])), int(round(start[1]))),
                (int(round(end[0])), int(round(end[1]))),
                max(1, int(width)),
            )
            return

        direction = (dx / distance, dy / distance)
        progress = 0.0
        width = max(1, int(width))

        while progress < distance:
            dash_end = min(progress + dash_length, distance)
            start_pos = (
                start[0] + direction[0] * progress,
                start[1] + direction[1] * progress,
            )
            end_pos = (
                start[0] + direction[0] * dash_end,
                start[1] + direction[1] * dash_end,
            )
            pygame.draw.line(
                surface,
                color,
                (int(round(start_pos[0])), int(round(start_pos[1]))),
                (int(round(end_pos[0])), int(round(end_pos[1]))),
                width,
            )
            progress += dash_length + gap_length

    def _formation_offsets(
        self, count: int, formation_type: str, spacing: float
    ) -> List[Tuple[float, float]]:
        if count <= 0:
            return []

        formation_type = formation_type.lower()
        offsets: List[Tuple[float, float]] = []

        if formation_type in {"line", "parallel"}:
            center_index = (count - 1) / 2.0
            offsets = [((i - center_index) * spacing, 0.0) for i in range(count)]
        elif formation_type == "triangle":
            remaining = count
            row = 1
            spacing_y = spacing * math.sin(math.pi / 3)
            while remaining > 0:
                row_count = min(row, remaining)
                x_start = -(row_count - 1) / 2.0 * spacing
                y_offset = (row - 1) * spacing_y
                for i in range(row_count):
                    offsets.append((x_start + i * spacing, y_offset))
                remaining -= row_count
                row += 1
        elif formation_type in {"square", "grid"}:
            rows, columns = self._compute_layout(count, None, spacing)
            for row_index in range(rows):
                y_offset = (row_index - (rows - 1) / 2.0) * spacing
                for column_index in range(columns):
                    if len(offsets) >= count:
                        break
                    x_offset = (column_index - (columns - 1) / 2.0) * spacing
                    offsets.append((x_offset, y_offset))
                if len(offsets) >= count:
                    break
        else:
            raise ValueError(f"Unsupported formation type: {formation_type}")

        if not offsets:
            return []

        mean_x = sum(x for x, _ in offsets) / len(offsets)
        mean_y = sum(y for _, y in offsets) / len(offsets)
        return [(x - mean_x, y - mean_y) for x, y in offsets]

    @staticmethod
    def _formation_angle_bounds(formation_type: str) -> Tuple[float, float]:
        formation_type = formation_type.lower()
        if formation_type in {"line", "parallel"}:
            return 0.0, math.pi
        if formation_type == "triangle":
            return 0.0, 2.0 * math.pi / 3.0
        if formation_type in {"square", "grid"}:
            return 0.0, math.pi / 2.0
        raise ValueError(f"Unsupported formation type: {formation_type}")

    def _generate_targets(
        self,
        center: Position,
        theta: float,
        formation_type: str,
        spacing: float,
    ) -> List[Position]:
        simulator = self.simulator
        offsets = self._formation_offsets(
            len(simulator.robot_states), formation_type, spacing
        )
        if not offsets:
            return []

        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        targets: List[Position] = []
        for offset_x, offset_y in offsets:
            rotated_x = offset_x * cos_theta - offset_y * sin_theta
            rotated_y = offset_x * sin_theta + offset_y * cos_theta
            targets.append((center[0] + rotated_x, center[1] + rotated_y))
        return targets

    def _evaluate_plan(
        self,
        center: Position,
        theta: float,
        formation_type: str,
        spacing: float,
        weights: Sequence[float],
        ego_index: Optional[int],
        penalty: float,
    ) -> Optional[FormationPlan]:
        simulator = self.simulator
        targets = self._generate_targets(center, theta, formation_type, spacing)
        robot_count = len(simulator.robot_states)
        if robot_count == 0 or not targets:
            return None

        weights = tuple(weights)
        if len(weights) != 5:
            raise ValueError("weights must contain five elements")

        formation_axis_direction = (math.cos(theta), math.sin(theta))
        progress_state = self._centroid_progress_vector()
        angle_difference = math.pi / 2.0
        if progress_state is not None:
            progress_dx, progress_dy, progress_displacement = progress_state
            if progress_displacement > 0.0:
                movement_direction_x = progress_dx / progress_displacement
                movement_direction_y = progress_dy / progress_displacement
                dot_product = (
                    movement_direction_x * formation_axis_direction[0]
                    + movement_direction_y * formation_axis_direction[1]
                )
                clamped_dot_product = max(-1.0, min(1.0, dot_product))
                angle_difference = math.acos(clamped_dot_product)

        # c5 = 2.0 * abs(angle_difference - math.pi / 2.0) / math.pi
        # c5 = abs(np.cos(angle_difference))
        c5 = (np.cos(2 * angle_difference) + 1) / 2.0


        cost_matrix = np.full((robot_count, robot_count), penalty, dtype=float)
        per_robot_costs: Dict[int, float] = {}

        clearance_tolerance = 1e-3  # Keep in sync with simulator._robot_line_blocked

        for robot_index, robot in enumerate(simulator.robot_states):
            for target_index, target in enumerate(targets):
                # C1
                clearance = simulator._robot_clearance(robot, target)
                if clearance <= clearance_tolerance:
                    # The robot either perceives an obstacle at the target or has
                    # no observations for that direction; treat it as unsafe.
                    continue
                
                # C2
                if ego_index is None or robot_index == ego_index:
                    if simulator._robot_line_blocked(robot, target):
                        continue

                distance_to_obstacle = clearance
                dx = target[0] - robot.position[0]
                dy = target[1] - robot.position[1]
                distance_pixels = math.hypot(dx, dy)
                distance_robot_target = distance_pixels / simulator.env.PIXELS_PER_METER

                previous_target = self.get_navigation_target(
                    robot_index, fallback=robot.target
                )
                if previous_target is not None:
                    previous_dx = target[0] - previous_target[0]
                    previous_dy = target[1] - previous_target[1]
                    distance_previous = (
                        math.hypot(previous_dx, previous_dy)
                        / simulator.env.PIXELS_PER_METER
                    )
                else:
                    distance_previous = 0.0

                heading_vector = (-math.sin(robot.theta), -math.cos(robot.theta))
                heading_norm = math.hypot(*heading_vector) or 1.0
                target_norm = distance_pixels or 1.0
                dot_product = heading_vector[0] * dx + heading_vector[1] * dy
                cos_angle = max(-1.0, min(1.0, dot_product / (heading_norm * target_norm)))
                relative_angle = math.acos(cos_angle)

                c1 = max(2.0 - distance_to_obstacle, 0.0)
                c2 = distance_robot_target
                c3 = distance_previous
                c4 = relative_angle / math.pi

                total_cost = (
                    weights[0] * c1
                    + weights[1] * c2
                    + weights[2] * c3
                    + weights[3] * c4
                    + weights[4] * c5
                )

                cost_matrix[robot_index, target_index] = total_cost

        if np.all(cost_matrix >= penalty):
            return None

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assignment = [0] * robot_count
        total_cost = 0.0
        for r, c in zip(row_ind, col_ind):
            cost = cost_matrix[r, c]
            if cost >= penalty:
                return None
            assignment[r] = c
            per_robot_costs[r] = cost
            total_cost += cost

        return FormationPlan(
            theta=theta,
            targets=targets,
            assignment=assignment,
            cost=total_cost,
            per_robot_costs=per_robot_costs,
            formation_type=formation_type,
            spacing=spacing,
            center=center,
        )

    def _compute_layout(
        self,
        count: int,
        spawn_area: Optional[Sequence[float]],
        spacing: float,
    ) -> Tuple[int, int]:
        simulator = self.simulator
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

    def _sample_center(
        self,
        spawn_area: Optional[Sequence[float]],
        spacing: float,
        rows: int,
        columns: int,
    ) -> Position:
        simulator = self.simulator
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
            center_x = float(
                simulator.env.np_random.uniform(allowed_min_x, allowed_max_x)
            )

        if allowed_min_y > allowed_max_y:
            center_y = (min_y + max_y) / 2.0
        else:
            center_y = float(
                simulator.env.np_random.uniform(allowed_min_y, allowed_max_y)
            )

        return center_x, center_y

    def _generate_positions(
        self,
        spawn_area: Optional[Sequence[float]],
        count: int,
        spacing: float,
        *,
        rows: Optional[int] = None,
        columns: Optional[int] = None,
        center: Optional[Position] = None,
        enforce_bounds: bool = True,
    ) -> List[Position]:
        simulator = self.simulator
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
            rows, columns = self._compute_layout(count, spawn_area, spacing)

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
        row_offsets = [
            (row - (rows - 1) / 2.0) * spacing for row in range(rows)
        ]
        column_offsets = [
            (col - (columns - 1) / 2.0) * spacing for col in range(columns)
        ]

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

    def _are_spawn_positions_valid(
        self, positions: Sequence[Position], spawn_area: Optional[Sequence[float]]
    ) -> bool:
        simulator = self.simulator
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

