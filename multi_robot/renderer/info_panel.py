"""Rendering utilities for the information panel."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import pygame

from .types import Color

if False:  # pragma: no cover - type checking helper
    from .simulator import MultiRobotSimulator


class InfoPanelRenderer:
    """Encapsulates the state and drawing logic for the side panel."""

    def __init__(self, simulator: "MultiRobotSimulator") -> None:
        self.simulator = simulator
        self.visible = True
        self.min_width = 320
        self.width = max(self.min_width, int(self.simulator.env.SCREEN_W * 0.4))
        self._last_width = self.width
        self._last_resize_target = self.width
        self.background = (245, 245, 245)
        self.text_color = (32, 32, 32)
        self.highlight_color = (220, 235, 250)
        self.divider_color = (200, 200, 200)
        self.padding = 16
        self.scroll = 0
        self.scroll_speed = 40
        self.content_height = self.simulator.env.SCREEN_H
        self.surface: Optional[pygame.Surface] = pygame.Surface(
            (self.width, self.simulator.env.SCREEN_H)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def toggle(self) -> None:
        self.set_visible(not self.visible)

    def set_visible(self, visible: bool) -> None:
        if self.visible == visible:
            return

        if visible:
            target_width = max(self.min_width, self._last_width)
            self.visible = True
            self.simulator.resize_display_surface(target_width)
        else:
            self._last_width = max(self.min_width, self.width)
            self.visible = False
            self.simulator.resize_display_surface(0)
            self.scroll = 0

    def draw(self) -> None:
        if not self.visible or self.width <= 0:
            self.simulator.present_info_panel_surface(
                None, 0, self.simulator.env.SCREEN_H
            )
            return

        while True:
            required_inner_width, required_height = self._render(resize_only=True)
            target_width = max(self.min_width, required_inner_width + 2 * self.padding)
            if target_width == self.width:
                break
            if target_width == self._last_resize_target:
                break
            self._last_resize_target = target_width
            self.simulator.resize_display_surface(target_width)

        panel_height = max(self.simulator.env.SCREEN_H, required_height + self.padding)
        if (
            self.surface is None
            or self.surface.get_width() != self.width
            or self.surface.get_height() != panel_height
        ):
            self.surface = pygame.Surface((self.width, panel_height))

        self.content_height = panel_height
        assert self.surface is not None
        self._render(resize_only=False, surface=self.surface)

        max_scroll = max(0, panel_height - self.simulator.env.SCREEN_H)
        self.scroll = max(0, min(self.scroll, max_scroll))

        self.simulator.present_info_panel_surface(
            self.surface, self.scroll, self.simulator.env.SCREEN_H
        )

    def scroll_panel(self, delta: float) -> None:
        max_scroll = max(0, self.content_height - self.simulator.env.SCREEN_H)
        self.scroll = max(0, min(self.scroll + int(delta), max_scroll))

    def update_width(self, new_width: int) -> None:
        self.width = new_width
        self._last_resize_target = new_width
        if self.width <= 0:
            self.surface = None
            return
        panel_height = max(self.simulator.env.SCREEN_H, self.content_height)
        self.surface = pygame.Surface((self.width, panel_height))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _render(
        self, *, resize_only: bool, surface: Optional[pygame.Surface] = None
    ) -> Tuple[int, int]:
        if not resize_only and surface is None:
            raise ValueError("surface must be provided when resize_only is False")

        if not resize_only:
            assert surface is not None
            surface.fill(self.background)
            pygame.draw.line(
                surface,
                self.divider_color,
                (0, 0),
                (0, surface.get_height()),
                width=2,
            )

        title_font = pygame.font.SysFont("arial", 20, bold=True)
        name_font = pygame.font.SysFont("arial", 18, bold=True)
        value_font = pygame.font.SysFont("consolas", 16)

        x_offset = self.padding
        y_offset = self.padding
        max_required_inner_width = 0

        def render_line(
            text: str,
            font: pygame.font.Font,
            color: Color,
            *,
            indent: int = 0,
            spacing: int = 4,
        ) -> None:
            nonlocal y_offset, max_required_inner_width
            if not text:
                return
            width, height = font.size(text)
            max_required_inner_width = max(max_required_inner_width, indent + width)
            if not resize_only and surface is not None:
                label = font.render(text, True, color)
                surface.blit(label, (x_offset + indent, y_offset))
            y_offset += height + spacing

        render_line("Simulation Info", title_font, self.text_color, spacing=10)

        selected_name = (
            self.simulator.robot_states[self.simulator.selected_robot_index].name
            if self.simulator.robot_states
            else "None"
        )
        general_lines = [
            f"FPS: {self.simulator.fps:5.1f}",
            f"Robots: {len(self.simulator.robot_states)}",
            f"Selected: {selected_name}",
        ]
        control_mode = "Policy" if self.simulator.policy_control_active else "Manual"
        if self.simulator.policy_controller is None and self.simulator.policy_model_path:
            control_mode += " (pending)"
        general_lines.append(f"Control: {control_mode}")

        formation = self.simulator.formation
        follow_state = "On" if formation.follow_formation_targets else "Off"
        vertical_state = "On" if formation.vertical_constraint_enabled else "Off"
        general_lines.append(f"Formation follow: {follow_state}")
        general_lines.append(f"Vertical constraint: {vertical_state}")

        if self.simulator.policy_controller is not None:
            model_status = (
                "Loaded" if self.simulator.policy_controller.is_available else "Not loaded"
            )
            general_lines.append(f"Policy model: {model_status}")
        for line in general_lines:
            render_line(line, value_font, self.text_color, spacing=4)

        controller = self.simulator.policy_controller
        if controller is not None and controller.last_error:
            render_line("Policy error:", value_font, self.text_color, spacing=2)
            wrap_width = max(10, self.width - 2 * self.padding - 10)
            for line in self._wrap_text_by_pixels(value_font, controller.last_error, wrap_width):
                render_line(line, value_font, self.text_color, indent=10, spacing=2)

        y_offset += 10

        render_line("Map", title_font, self.text_color, spacing=8)

        map_width_m = self.simulator.env.SCREEN_W / self.simulator.env.PIXELS_PER_METER
        map_height_m = self.simulator.env.SCREEN_H / self.simulator.env.PIXELS_PER_METER
        map_lines = [
            f"Type: {self.simulator.env.map_type}",
            f"Size: {map_width_m:.1f}m x {map_height_m:.1f}m",
        ]
        for line in map_lines:
            render_line(line, value_font, self.text_color, spacing=4)

        description = self.simulator.map_descriptions.get(self.simulator.env.map_type)
        if description:
            render_line("Description:", value_font, self.text_color, spacing=4)
            desc_indent = 10
            wrap_width = max(10, self.width - 2 * self.padding - desc_indent)
            for line in self._wrap_text_by_pixels(value_font, description, wrap_width):
                render_line(line, value_font, self.text_color, indent=desc_indent, spacing=2)

            y_offset += 6

        formation = self.simulator.formation
        if formation.plan_candidates:
            render_line("Formation Plans", title_font, self.text_color, spacing=8)

            for ego_index in sorted(formation.plan_candidates):
                plan = formation.plan_candidates[ego_index]
                if not plan.targets:
                    continue

                robot_name = (
                    self.simulator.robot_states[ego_index].name
                    if 0 <= ego_index < len(self.simulator.robot_states)
                    else f"Robot {ego_index}"
                )
                spacing_m = plan.spacing / self.simulator.env.PIXELS_PER_METER
                center_m = self.simulator.pixels_to_meters(*plan.center)
                heading_deg = math.degrees(plan.theta)
                cost_text = f"{plan.cost:.3f}" if plan.cost != float("inf") else "∞"

                highlight = plan is formation.current_plan
                indicator = "*" if highlight else "•"
                if highlight and 0 <= ego_index < len(self.simulator.robot_states):
                    robot_color = self.simulator.robot_states[ego_index].color
                    summary_color = tuple(
                        min(255, int(tc * 0.5 + rc * 0.5))
                        for tc, rc in zip(self.text_color, robot_color)
                    )
                else:
                    summary_color = self.text_color

                summary = (
                    f"{indicator} {robot_name}: {plan.formation_type.title()}"
                    f" θ={heading_deg:.1f}° cost={cost_text}"
                )
                render_line(summary, value_font, summary_color, spacing=2)

                if highlight:
                    detail_lines = [
                        f"Spacing: {spacing_m:.2f} m",
                        f"Center: ({center_m[0]:.2f}, {center_m[1]:.2f}) m",
                    ]
                    for line in detail_lines:
                        render_line(line, value_font, self.text_color, indent=10, spacing=2)

                    render_line(
                        "Assignments:",
                        value_font,
                        self.text_color,
                        indent=10,
                        spacing=2,
                    )
                    for robot_index, target_index in enumerate(plan.assignment):
                        robot_name = (
                            self.simulator.robot_states[robot_index].name
                            if 0 <= robot_index < len(self.simulator.robot_states)
                            else f"Robot {robot_index}"
                        )
                        render_line(
                            f"- {robot_name} → T{target_index}",
                            value_font,
                            self.text_color,
                            indent=20,
                            spacing=2,
                        )

                    y_offset += 8

        else:
            active_targets = [
                self.simulator.formation.get_navigation_target(
                    index, fallback=robot.target
                )
                for index, robot in enumerate(self.simulator.robot_states)
            ]
            if any(target is not None for target in active_targets):
                render_line("Target Assignments", title_font, self.text_color, spacing=8)

                for index, robot in enumerate(self.simulator.robot_states):
                    target = active_targets[index]
                    if target is None:
                        continue

                    target_m = self.simulator.pixels_to_meters(*target)
                    distance_pixels = math.hypot(
                        target[0] - robot.position[0],
                        target[1] - robot.position[1],
                    )
                    distance_m = distance_pixels / self.simulator.env.PIXELS_PER_METER
                    text = (
                        f"{robot.name}: ({target_m[0]:.2f}, {target_m[1]:.2f}) m"
                        f" / {distance_m:.2f} m away"
                    )

                    direct_target = robot.target
                    if direct_target is not None and (
                        abs(direct_target[0] - target[0]) > 1e-6
                        or abs(direct_target[1] - target[1]) > 1e-6
                    ):
                        direct_m = self.simulator.pixels_to_meters(*direct_target)
                        direct_distance = math.hypot(
                            direct_target[0] - robot.position[0],
                            direct_target[1] - robot.position[1],
                        )
                        direct_distance_m = (
                            direct_distance / self.simulator.env.PIXELS_PER_METER
                        )
                        text += (
                            f" | Direct: ({direct_m[0]:.2f}, {direct_m[1]:.2f}) m"
                            f" / {direct_distance_m:.2f} m"
                        )

                    render_line(text, value_font, self.text_color, indent=10, spacing=2)

            y_offset += 12

        render_line("Robots", title_font, self.text_color, spacing=8)

        if not self.simulator.robot_states:
            render_line("No robots", value_font, self.text_color, spacing=4)
            return max_required_inner_width, y_offset

        line_height = value_font.get_height() + 4

        for index, robot in enumerate(self.simulator.robot_states):
            robot_name_text = robot.name
            pos_m = self.simulator.pixels_to_meters(*robot.position)
            metrics: List[str] = [
                f"Pos: ({pos_m[0]:.2f}, {pos_m[1]:.2f}) m",
                f"Theta: {math.degrees(robot.theta):.1f}°",
                f"Speed: {robot.velocity:.2f} m/s",
            ]

            active_target = self.simulator.formation.get_navigation_target(
                index, fallback=robot.target
            )
            if active_target is None:
                metrics.append("Target: None")
            else:
                active_m = self.simulator.pixels_to_meters(*active_target)
                distance_active = math.hypot(
                    active_m[0] - pos_m[0], active_m[1] - pos_m[1]
                )
                metrics.append(f"Active target dist: {distance_active:.2f} m")

                direct_target = robot.target
                if direct_target is not None and (
                    abs(direct_target[0] - active_target[0]) > 1e-6
                    or abs(direct_target[1] - active_target[1]) > 1e-6
                ):
                    direct_m = self.simulator.pixels_to_meters(*direct_target)
                    distance_direct = math.hypot(
                        direct_m[0] - pos_m[0], direct_m[1] - pos_m[1]
                    )
                    metrics.append(f"Direct target dist: {distance_direct:.2f} m")

            if robot.closest_obstacle[0] != float("inf"):
                metrics.append(f"Closest obs: {robot.closest_obstacle[0]:.2f} m")
            else:
                metrics.append("Closest obs: --")

            metrics.append(f"LiDAR: {'On' if robot.lidar_enabled else 'Off'}")
            metrics.append(f"Path: {'On' if robot.path_enabled else 'Off'}")

            name_width, name_height = name_font.size(robot_name_text)
            block_height = name_height + 4 + len(metrics) * line_height + 12
            block_rect = pygame.Rect(
                8,
                y_offset - 6,
                self.width - 16,
                block_height,
            )

            text_indent = max(0, (block_rect.x + 10) - x_offset)
            max_required_inner_width = max(max_required_inner_width, text_indent + name_width)
            for metric in metrics:
                metric_width, _ = value_font.size(metric)
                max_required_inner_width = max(
                    max_required_inner_width, text_indent + metric_width
                )

            if not resize_only:
                background_color = (
                    self.highlight_color
                    if index == self.simulator.selected_robot_index
                    else self.background
                )
                pygame.draw.rect(
                    surface,
                    background_color,
                    block_rect,
                    border_radius=8,
                )
                pygame.draw.rect(
                    surface,
                    self.divider_color,
                    block_rect,
                    width=1,
                    border_radius=8,
                )

                text_x = block_rect.x + 10
                text_y = y_offset
                robot_name_surface = name_font.render(robot_name_text, True, self.text_color)
                surface.blit(robot_name_surface, (text_x, text_y))
                text_y += robot_name_surface.get_height() + 4

                for metric in metrics:
                    metric_label = value_font.render(metric, True, self.text_color)
                    surface.blit(metric_label, (text_x, text_y))
                    text_y += line_height

            y_offset = block_rect.bottom + 10

        return max_required_inner_width, y_offset

    def _wrap_text_by_pixels(
        self, font: pygame.font.Font, text: str, max_width: int
    ) -> List[str]:
        if max_width <= 0:
            return [text]

        words = text.split()
        if not words:
            return []

        lines: List[str] = []

        first_word = words[0]
        if font.size(first_word)[0] > max_width:
            segments = self._break_word_by_pixels(font, first_word, max_width)
            if segments:
                lines.extend(segments[:-1])
                current = segments[-1]
            else:
                current = first_word
        else:
            current = first_word

        for word in words[1:]:
            candidate = f"{current} {word}"
            if font.size(candidate)[0] <= max_width:
                current = candidate
            else:
                lines.append(current)
                if font.size(word)[0] <= max_width:
                    current = word
                else:
                    oversized = self._break_word_by_pixels(font, word, max_width)
                    if oversized:
                        lines.extend(oversized[:-1])
                        current = oversized[-1]
                    else:
                        current = word

        lines.append(current)
        return lines

    def _break_word_by_pixels(
        self, font: pygame.font.Font, word: str, max_width: int
    ) -> List[str]:
        if max_width <= 0:
            return [word]

        segments: List[str] = []
        current = ""

        for char in word:
            candidate = current + char
            if not current or font.size(candidate)[0] <= max_width:
                current = candidate
            else:
                segments.append(current)
                current = char

        if current:
            segments.append(current)

        return segments

