"""Formation planning with forward-centroid search and prioritised sharing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pygame
from scipy.optimize import linear_sum_assignment, minimize, minimize_scalar

from .formation_sidebyside_4 import FormationManager as SideBySide4Manager
from .types import FormationPlan, Position
from .utils import brighten_color, with_alpha

if False:  # pragma: no cover - type checking helper
    from .simulator_v5 import MultiRobotSimulator


@dataclass(frozen=True)
class _PrioritisedRobotPlanCandidate:
    """Robot specific candidate enriched with a sharing priority."""

    theta: float
    target_index: int
    cost: float
    target: Position
    priority: float
    source_index: Optional[int]


class FormationManager(SideBySide4Manager):
    """Formation manager that follows the side-by-side V5 specification.

    The implementation keeps the rendering and policy integration offered by
    :class:`~multi_robot.renderer.formation_sidebyside_4.FormationManager` but
    swaps out the optimisation pipeline so that it follows the algorithm
    described in the accompanying research memo.  The main changes are:

    * Step 1 introduces a forward-searching centroid optimisation.
    * Step 2 evaluates formation targets with cost components C2–C5 only.
    * Step 3 prioritises and shares formation proposals between robot pairs.
    * Step 4 relies on the base policy integration once the formation targets
      are finalised.
    """

    def __init__(self, simulator: "MultiRobotSimulator") -> None:
        super().__init__(simulator)
        self._plan_priorities: Dict[int, float] = {}
        self._ranked_targets: Dict[int, List[Tuple[Optional[Position], float]]] = {}

    # ------------------------------------------------------------------
    # Helpers overriding the base implementation
    # ------------------------------------------------------------------
    def _reset_plan_state(self) -> None:  # type: ignore[override]
        super()._reset_plan_state()
        self._plan_priorities.clear()
        self._ranked_targets.clear()

    def draw(self) -> None:  # type: ignore[override]
        super().draw()

        if not self.follow_formation_targets:
            return

        if not self._ranked_targets:
            return

        simulator = self.simulator
        if not simulator.robot_states:
            return

        layer = simulator.env.robot_and_lidar_layer
        font = pygame.font.SysFont("consolas", 12)
        primary_width = max(1, int(round(simulator.robot_radius_pixels * 0.2)))
        secondary_width = max(1, int(round(simulator.robot_radius_pixels * 0.1)))

        for robot_index, robot in enumerate(simulator.robot_states):
            ranked_candidates = self._ranked_targets.get(robot_index)
            if not ranked_candidates:
                continue

            base_color = robot.color
            robot_position = (int(robot.position[0]), int(robot.position[1]))

            for rank, (target, priority) in enumerate(ranked_candidates):
                if target is None:
                    continue

                target_position = (int(target[0]), int(target[1]))

                if rank == 0:
                    color = brighten_color(base_color)
                    radius = max(4, int(round(simulator.robot_radius_pixels * 0.45)))
                    line_width = primary_width
                    circle_width = 0
                else:
                    color = with_alpha(base_color, 0.65)
                    radius = max(3, int(round(simulator.robot_radius_pixels * 0.35)))
                    line_width = secondary_width
                    circle_width = 1

                pygame.draw.line(layer, color, robot_position, target_position, line_width)
                pygame.draw.circle(layer, color, target_position, radius, circle_width)

                if priority > 0.0:
                    label = font.render(f"{priority:.2f}", True, (30, 30, 30))
                    label_rect = label.get_rect()
                    label_rect.midleft = (target_position[0] + radius + 4, target_position[1])
                    layer.blit(label, label_rect)

    # ------------------------------------------------------------------
    # Step 1 – centroid search
    # ------------------------------------------------------------------
    def _search_forward_centroid(
        self, current_center: Position, formation_type: str, spacing: float
    ) -> Optional[Position]:
        simulator = self.simulator
        robot_states = simulator.robot_states
        if not robot_states:
            return None

        offsets = self._formation_offsets(len(robot_states), formation_type, spacing)
        if not offsets:
            return None

        max_offset_radius = max(math.hypot(x, y) for x, y in offsets) if offsets else 0.0
        frame_radius = float(getattr(simulator, "robot_frame_radius", 0.0))
        pixels_per_meter = float(getattr(simulator.env, "PIXELS_PER_METER", 1.0) or 1.0)
        team_radius_pixels = max_offset_radius + frame_radius
        team_radius = team_radius_pixels / max(pixels_per_meter, 1e-6)
        if team_radius <= 0.0:
            return None

        forward_direction = self._estimate_forward_direction(current_center)
        if forward_direction is None:
            forward_direction = (0.0, -1.0)
        front_robot = max(
            robot_states,
            key=lambda robot: (
                (robot.position[0] - current_center[0]) * forward_direction[0]
                + (robot.position[1] - current_center[1]) * forward_direction[1]
            )
        )
        origin = front_robot.position

        circle_center = (float(current_center[0]), float(current_center[1]))
        circle_radius = team_radius_pixels
        if circle_radius <= 0.0:
            return None

        goal_position = self._determine_navigation_goal()
        current_goal_distance = None
        if goal_position is not None:
            current_goal_distance = math.hypot(
                current_center[0] - goal_position[0],
                current_center[1] - goal_position[1],
            )

        best_candidate: Optional[Position] = None
        best_cost = float("inf")

        def evaluate(
            candidate: Position,
            result: Optional[Tuple[float, bool]] = None,
        ) -> None:
            nonlocal best_candidate, best_cost
            dx_circle = candidate[0] - circle_center[0]
            dy_circle = candidate[1] - circle_center[1]
            if dx_circle * dx_circle + dy_circle * dy_circle > circle_radius * circle_radius:
                return
            if result is None:
                cost, feasible = self._evaluate_centroid_candidate(
                    candidate,
                    origin=origin,
                    team_radius=team_radius,
                    current_center=current_center,
                    goal_position=goal_position,
                    current_goal_distance=current_goal_distance,
                )
            else:
                cost, feasible = result
            if feasible and cost < best_cost:
                best_candidate = (float(candidate[0]), float(candidate[1]))
                best_cost = cost

        # Always consider the current center as a fallback option.
        evaluate(current_center)

        large_penalty = 1e6

        circle_bounds = (
            (circle_center[0] - circle_radius, circle_center[0] + circle_radius),
            (circle_center[1] - circle_radius, circle_center[1] + circle_radius),
        )

        def penalised_objective(point: Sequence[float]) -> float:
            candidate = (float(point[0]), float(point[1]))
            dx_circle = candidate[0] - circle_center[0]
            dy_circle = candidate[1] - circle_center[1]
            if dx_circle * dx_circle + dy_circle * dy_circle > circle_radius * circle_radius:
                return large_penalty

            result = self._evaluate_centroid_candidate(
                candidate,
                origin=origin,
                team_radius=team_radius,
                current_center=current_center,
                goal_position=goal_position,
                current_goal_distance=current_goal_distance,
            )
            cost, feasible = result
            if feasible:
                evaluate(candidate, result)
                return cost
            return large_penalty

        bounds = (
            circle_bounds[0],
            circle_bounds[1],
        )

        initial_guesses: List[Tuple[float, float]] = [
            circle_center,
        ]

        if forward_direction is not None:
            forward_guess = (
                circle_center[0] + forward_direction[0] * circle_radius * 0.8,
                circle_center[1] + forward_direction[1] * circle_radius * 0.8,
            )
            initial_guesses.append(forward_guess)

        initial_guesses.extend(
            (
                circle_center[0] + circle_radius * 0.6 * math.cos(angle),
                circle_center[1] + circle_radius * 0.6 * math.sin(angle),
            )
            for angle in np.linspace(0.0, 2.0 * math.pi, num=6, endpoint=False)
        )

        for guess in initial_guesses:
            evaluate(guess)
            try:
                result = minimize(
                    penalised_objective,
                    guess,
                    method="L-BFGS-B",
                    bounds=bounds,
                )
            except Exception:
                continue

            if result is not None and getattr(result, "success", False):
                candidate = (float(result.x[0]), float(result.x[1]))
                evaluate(
                    candidate,
                    self._evaluate_centroid_candidate(
                        candidate,
                        origin=origin,
                        team_radius=team_radius,
                        current_center=current_center,
                        goal_position=goal_position,
                        current_goal_distance=current_goal_distance,
                    ),
                )

        for angle in np.linspace(0.0, 2.0 * math.pi, num=8, endpoint=False):
            candidate = (
                circle_center[0] + circle_radius * math.cos(angle),
                circle_center[1] + circle_radius * math.sin(angle),
            )
            evaluate(candidate)

        return best_candidate

    def _estimate_forward_direction(
        self, current_center: Position
    ) -> Optional[Tuple[float, float]]:
        progress_state = self._centroid_progress_vector()
        if progress_state is not None:
            progress_dx, progress_dy, displacement = progress_state
            if displacement > 1e-6:
                scale = 1.0 / displacement
                return (float(progress_dx * scale), float(progress_dy * scale))

        goal_position = self._determine_navigation_goal()
        if goal_position is not None:
            dx = goal_position[0] - current_center[0]
            dy = goal_position[1] - current_center[1]
            length = math.hypot(dx, dy)
            if length > 1e-6:
                inv = 1.0 / length
                return (float(dx * inv), float(dy * inv))

        # Default forward direction: upward on the simulator screen.
        return (0.0, -1.0)

    def _determine_navigation_goal(self) -> Optional[Position]:
        """Return the final navigation target shared by the team when known."""

        # Prefer explicitly assigned direct navigation goals when the operators
        # have provided them. These correspond to the final navigation targets
        # that the formation is ultimately expected to reach.
        self._prune_direct_target_cache()
        for target in self._direct_targets.values():
            if target is not None:
                return (float(target[0]), float(target[1]))

        # Fall back to any per-robot goal available via the simulator state.
        for robot in self.simulator.robot_states:
            if robot.target is not None:
                return (float(robot.target[0]), float(robot.target[1]))

        # As a last resort rely on the environment level target.  This matches
        # the global navigation goal configured for the map.
        env_goal = getattr(self.simulator.env, "target_position", None)
        if env_goal is not None:
            return (float(env_goal[0]), float(env_goal[1]))

        return None

    def _evaluate_centroid_candidate(
        self,
        candidate: Position,
        *,
        origin: Position,
        team_radius: float,
        current_center: Position,
        goal_position: Optional[Position],
        current_goal_distance: Optional[float],
    ) -> Tuple[float, bool]:
        simulator = self.simulator
        x, y = candidate
        if not (0.0 <= x < simulator.env.SCREEN_W and 0.0 <= y < simulator.env.SCREEN_H):
            return float("inf"), False

        clearance = simulator._distance_to_nearest_obstacle_map(candidate) - team_radius
        if clearance <= 0.0:
            return float("inf"), False

        clearance_cost = 1.0 / max(clearance, 1e-6)

        pixels_per_meter = max(simulator.env.PIXELS_PER_METER, 1e-6)

        if self.previous_center is not None:
            previous_dx = candidate[0] - self.previous_center[0]
            previous_dy = candidate[1] - self.previous_center[1]
            previous_distance_m = math.hypot(previous_dx, previous_dy) / pixels_per_meter
        else:
            previous_distance_m = 0.0

        if (
            goal_position is not None
            and current_goal_distance is not None
            and team_radius > 1e-6
        ):
            current_goal_distance_m = current_goal_distance / pixels_per_meter
            candidate_goal_distance_m = math.hypot(
                candidate[0] - goal_position[0],
                candidate[1] - goal_position[1],
            ) / pixels_per_meter
            raw_progress = (current_goal_distance_m - candidate_goal_distance_m) / team_radius
            normalised_progress = min(1.0, max(0.0, raw_progress))
            goal_progress_ratio = 1.0 - normalised_progress
        else:
            goal_progress_ratio = 0.0

        # displacement_from_origin = math.hypot(candidate[0] - origin[0], candidate[1] - origin[1])
        # displacement_cost = displacement_from_origin / max(
        #     simulator.env.PIXELS_PER_METER, 1e-6
        # )


        total_cost = (
            0.5 * clearance_cost
            + 0.3 * previous_distance_m
            + 1.0 * goal_progress_ratio
            # + 0.2 * displacement_cost
        )

        return total_cost, True

    # ------------------------------------------------------------------
    # Step 2 – formation target optimisation (C2–C5)
    # ------------------------------------------------------------------
    def optimize(  # type: ignore[override]
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

        candidate_center = self._search_forward_centroid(current_center, formation_type, spacing)
        if candidate_center is None:
            self._reset_plan_state()
            raise ValueError("Unable to determine a forward centroid candidate")

        center = candidate_center
        self.previous_center = center
        self._record_centroid(current_center)

        # Only keep cost weights for C2–C5, matching the user specification.
        weights = weights or (0.0, 0.5, 0.5, 0.3, 1.0)
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

        plan_priority = self._compute_plan_priority(ego_index) if ego_index is not None else 0.0
        setattr(plan, "_source_index", ego_index)
        setattr(plan, "_priority", plan_priority)
        if ego_index is not None:
            self._plan_priorities[ego_index] = plan_priority

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

    def _evaluate_plan(  # type: ignore[override]
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
        alignment_angle = math.pi / 2.0
        has_alignment_direction = False
        goal_position = self._determine_navigation_goal()
        goal_distance = 0.0
        if goal_position is not None:
            goal_dx = goal_position[0] - center[0]
            goal_dy = goal_position[1] - center[1]
            goal_distance = math.hypot(goal_dx, goal_dy)
            if goal_distance > 1e-6:
                inv_distance = 1.0 / goal_distance
                goal_direction_x = goal_dx * inv_distance
                goal_direction_y = goal_dy * inv_distance
                dot_product = (
                    goal_direction_x * formation_axis_direction[0]
                    + goal_direction_y * formation_axis_direction[1]
                )
                clamped_dot_product = max(-1.0, min(1.0, dot_product))
                alignment_angle = math.acos(clamped_dot_product)
                has_alignment_direction = True
        if (goal_position is None or goal_distance <= 1e-6) and not has_alignment_direction:
            progress_state = self._centroid_progress_vector()
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
                    alignment_angle = math.acos(clamped_dot_product)
                    has_alignment_direction = True

        c5 = alignment_angle / math.pi

        cost_matrix = np.full((robot_count, robot_count), penalty, dtype=float)
        per_robot_costs: Dict[int, float] = {}

        clearance_tolerance = 1e-3

        for robot_index, robot in enumerate(simulator.robot_states):
            for target_index, target in enumerate(targets):
                clearance = simulator._robot_clearance(robot, target)
                if clearance <= clearance_tolerance:
                    continue

                if ego_index is None or robot_index == ego_index:
                    if simulator._robot_line_blocked(robot, target):
                        continue

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

                c2 = distance_robot_target
                c3 = distance_previous
                c4 = relative_angle / math.pi

                total_cost = (
                    weights[1] * c2
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

    # ------------------------------------------------------------------
    # Step 3 – sharing with priorities
    # ------------------------------------------------------------------
    def _aggregate_plan_candidates(  # type: ignore[override]
        self, plans: Iterable[FormationPlan]
    ) -> Dict[int, List[_PrioritisedRobotPlanCandidate]]:
        aggregated: Dict[int, List[_PrioritisedRobotPlanCandidate]] = {}
        robot_count = len(self.simulator.robot_states)

        for plan in plans:
            if not plan.targets or not plan.assignment:
                continue

            priority = float(getattr(plan, "_priority", 0.0))
            source_index = getattr(plan, "_source_index", None)

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
                candidate = _PrioritisedRobotPlanCandidate(
                    theta=float(plan.theta),
                    target_index=int(target_index),
                    cost=cost,
                    target=(float(target[0]), float(target[1])),
                    priority=priority,
                    source_index=source_index if isinstance(source_index, int) else None,
                )
                aggregated.setdefault(robot_index, []).append(candidate)

        return aggregated

    def _select_best_targets_for_robot(  # type: ignore[override]
        self, candidates: Dict[int, List[_PrioritisedRobotPlanCandidate]]
    ) -> Dict[int, Position]:
        simulator = self.simulator
        selected: Dict[int, Position] = {}
        self._ranked_targets = {}

        for robot_index, robot in enumerate(simulator.robot_states):
            feasible_candidates: List[_PrioritisedRobotPlanCandidate] = []
            for candidate in candidates.get(robot_index, []):
                if simulator._robot_line_blocked(robot, candidate.target):
                    continue

                clearance = simulator._robot_clearance(robot, candidate.target)
                if clearance <= 0.0:
                    continue

                feasible_candidates.append(candidate)

            feasible_candidates.sort(
                key=lambda item: (-item.priority, item.cost, item.theta)
            )

            ranked_targets: List[Tuple[Optional[Position], float]] = []
            for candidate in feasible_candidates[:2]:
                ranked_targets.append(((candidate.target[0], candidate.target[1]), candidate.priority))

            if not ranked_targets:
                self._ranked_targets[robot_index] = [(None, 0.0), (None, 0.0)]
                continue

            self._ranked_targets[robot_index] = ranked_targets
            selected_target = ranked_targets[0][0]
            if selected_target is not None:
                selected[robot_index] = (
                    float(selected_target[0]),
                    float(selected_target[1]),
                )

        return selected

    # ------------------------------------------------------------------
    # Priority helpers
    # ------------------------------------------------------------------
    def _compute_plan_priority(self, ego_index: int) -> float:
        if not (0 <= ego_index < len(self.simulator.robot_states)):
            return 0.0
        robot = self.simulator.robot_states[ego_index]
        obstacle_distance_m = robot.closest_obstacle[0]
        if not math.isfinite(obstacle_distance_m):
            return 0.0
        return 1.0 / max(obstacle_distance_m, 1e-3)

    @property
    def ranked_targets(self) -> Dict[int, List[Tuple[Optional[Position], float]]]:
        """Expose the per-robot ranked targets for debugging or UI overlays."""

        return {index: list(entries) for index, entries in self._ranked_targets.items()}
