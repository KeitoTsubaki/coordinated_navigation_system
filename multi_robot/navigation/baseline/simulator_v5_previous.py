"""Pygame based renderer for a multi-robot simulator."""

from __future__ import annotations

import argparse
import copy
import math
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple
import yaml
from pathlib import Path
import datetime
import csv
import os

import numpy as np
import pygame
from scipy import ndimage
import imageio

try:  # pragma: no cover - optional dependency
    from pygame._sdl2.video import Renderer as SDLRenderer, Texture as SDLTexture, Window as SDLWindow
except ImportError:  # pragma: no cover - SDL2 extension may be unavailable during import time
    SDLRenderer = None  # type: ignore[assignment]
    SDLTexture = None  # type: ignore[assignment]
    SDLWindow = None  # type: ignore[assignment]

from ..env import LiDARSensor, SimCoreEnvironment, utils_module
# from .formation import FormationManager
# from .formation_previous_research import FormationManager
# from .formation_sidebyside_1 import FormationManager
from .formation_sidebyside_2 import FormationManager
# from .formation_sidebyside_3 import FormationManager
# from .formation_sidebyside_4 import FormationManager
# from .formation_sidebyside_5 import FormationManager

from ..info_panel import InfoPanelRenderer
# from .policy2 import PolicyController
# from .policy2 import PolicyController
from .policy2_newobs_previous import PolicyController
from ..types import Color, FormationPlan, Position, RobotState
from ..utils import brighten_color, color_from_palette, to_rgb, with_alpha
from ..vwastar_planner import VWAstarPlanner, VWAstarConfig
# from .vwastar_planner import VWAstarPlanner, VWAstarConfig, TeamNavigationController


class MultiRobotSimulator:
    """Utility that renders multiple differential drive robots using pygame."""

    def __init__(
        self,
        num_robots: Optional[int] = None,
        *,
        render_mode: str = "human",
        config_path: Optional[str] = None,
        path_history: int = 600,
        show_labels: bool = True,
        manual_control: bool = True,
        # policy_model_path: Optional[str] = "multi_robot/models/model_ep3700.zip",
        policy_model_path: Optional[str] = "multi_robot/models/model_ep16500.zip",
        # policy_model_path: Optional[str] = "multi_robot/models/model_ep20800.zip",
        show_info_panel: bool = True,
        formation_type: Optional[str] = None,
        formation_spacing_m: Optional[float] = None,
        image_map_path: Optional[str] = None,
        map_type: Optional[str] = None,
        show_trail: bool = True,
        trail_step_interval: int = 20, # steps
    ) -> None:
        if config_path is None:
            import importlib.resources
            config_path = importlib.resources.files("simulation_core") / "config" / "simulator_config.yaml"

        self._policy_controller: Optional[PolicyController] = (
            PolicyController(self, policy_model_path)
            if policy_model_path is not None
            else None
        )
        self.policy_control_active = False

        self.env = SimCoreEnvironment(render_mode=render_mode, config_path=config_path)

        

        self.show_lidar_visualization = False

        # 軌跡設定
        self.show_trail = show_trail
        self.trail_step_interval = trail_step_interval
        self.trail_width = 10

        self.filter_robot = True

        # --- 録画用変数 ---
        self.is_recording = False
        self.frames = []
        # ----------------------

        # 画像パスの解決ロジック
        self._image_map_path_override: Optional[str] = None
        if image_map_path is not None:
            override_path = Path(image_map_path)
            final_path = None
            if override_path.is_absolute():
                if override_path.exists():
                    final_path = override_path
            if final_path is None:
                cwd_based_path = (Path.cwd() / override_path).resolve()
                if cwd_based_path.exists():
                    final_path = cwd_based_path
            if final_path is None:
                config_based_path = (self.env._config_dir / override_path).resolve()
                if config_based_path.exists():
                    final_path = config_based_path

            if final_path is not None:
                print(f"\n[INFO] Image map successfully found at: {final_path}")
                self.env.image_map_config = dict(self.env.image_map_config)
                self.env.image_map_config["path"] = str(final_path)
                self.env.image_map_path = final_path
                self._image_map_path_override = str(final_path)
            else:
                print(f"\n[ERROR] Image map file NOT found: {image_map_path}")

        
        self.formation_centroid_sampling_interval = self.env.FORMATION_CENTROID_SAMPLING_INTERVAL
        self.formation_centroid_look_back_steps = self.env.FORMATION_LOOK_BACK_STEPS
        self.formation_centroid_angular_tolerance_deg = self.env.FORMATION_ANGULAR_TOLERANCE_DEG
        self.formation_centroid_angular_tolerance_rad = self.env.FORMATION_ANGULAR_TOLERANCE_RAD
        self.formation_centroid_min_displacement_m = self.env.FORMATION_MIN_DISPLACEMENT_M
        self.formation_centroid_min_displacement_pixels = self.env.FORMATION_MIN_DISPLACEMENT_PIXELS

        self.default_formation_robot_count = self.env.DEFAULT_FORMATION_ROBOT_COUNT
        self.default_formation_type = self.env.DEFAULT_FORMATION_TYPE
        self.default_formation_spacing = self.env.DEFAULT_FORMATION_SPACING_PIXELS
        self.default_formation_spacing_m = self.env.DEFAULT_FORMATION_SPACING_M
        
        if formation_type is not None:
            self.default_formation_type = formation_type
        if formation_spacing_m is not None:
            self.default_formation_spacing_m = formation_spacing_m
            self.default_formation_spacing = formation_spacing_m * self.env.PIXELS_PER_METER
        
        resolved_robot_count = num_robots if num_robots is not None else self.default_formation_robot_count
        self.path_history = max(0, path_history)
        self.show_labels = show_labels
        self.running = True
        self.manual_control = manual_control
        self._manual_control_enabled = manual_control
        self.selected_robot_index = 0
        self.manual_step_pixels = 5
        self.manual_rotation_step = 0.1
        self.show_debug_text = True
        self._clock = pygame.time.Clock()
        self._fps = 0.0

        # 色や定数の設定
        self.robot_radius_pixels = self.env.ROBOT_RADIUS_M * 0.8 * self.env.PIXELS_PER_METER
        self.robot_frame_radius = self.env.ROBOT_RADIUS_M * self.env.PIXELS_PER_METER
        self.obstacle_color_rgb = to_rgb(self.env.OBSTACLE_COLOR)
        self.lidar_line_color = to_rgb(self.env.LIDAR_LINE_COLOR)
        self.lidar_filtered_points_color: Color = (64, 224, 208)
        self._lidar_detection_surface = pygame.Surface((self.env.SCREEN_W, self.env.SCREEN_H), pygame.SRCALPHA)

        # === VWA* Planner（コストマップ計算用）の初期化 ===
        # FRT/FMR計測のために「広い領域」と「狭い領域」を判定するマップを作成する
        vw_config = VWAstarConfig()
        # 必要に応じてインフレート半径を調整してください（例: [Wide用, Narrow用]）
        m = 0.5 # safety margin
        formation_spacing_side = 0.5
        robot_radius_m = 0.5
        radius_wide = robot_radius_m * 2.0 + formation_spacing_side + m
        # 2. ロボット1台分 (Narrow): robot radius + safety margin
        radius_narrow = robot_radius_m + m
        vw_config.InflateRadiusListM = [radius_wide, radius_narrow] 
        self.vw_planner = VWAstarPlanner(vw_config, pixels_per_meter=self.env.PIXELS_PER_METER)
        # ========================================================
        
        self.metrics_enabled = True
        self.test_epsilons = [0.1, 0.2, 0.25, 0.3, 0.5, 1.0]
        self.metrics_data = {
            "total_steps": 0,
            "success": False,
            "saved": False,
            "wide_area_steps": 0,
            "maintained_steps": {eps: 0 for eps in self.test_epsilons},
            "frt_events": {eps: [] for eps in self.test_epsilons},
            "current_frt_start": {eps: None for eps in self.test_epsilons},
            "is_in_wide_area": False,
            "prev_centroid_m": None,
            "fps_sum": 0.0,
            "fps_count": 0
        }

        # =========================================================================
        # Config読み込みと VW-Astar パラメータ計算
        # =========================================================================
        previous_research_config_path = Path(__file__).resolve().parents[2] / "config" / "previous_research_params.yaml"
        previous_research_params = {
            "robot_radius_m": self.env.ROBOT_RADIUS_M, # デフォルト
            "start_pos_m": None, # デフォルトNone
            "goal_pos_m": None   # デフォルトNone
        }
        
        if previous_research_config_path.exists():
            try:
                with open(previous_research_config_path, "r", encoding="utf-8") as f:
                    loaded_params = yaml.safe_load(f)
                    if loaded_params:
                        previous_research_params.update(loaded_params)
                print(f"[INFO] Loaded planner params from {previous_research_config_path}")
            except Exception as e:
                print(f"[ERROR] Failed to load planner params: {e}")

        

        # === 可視化設定の適用 ===
        viz_params = previous_research_params.get("visualization", {})
        
        # デフォルト値は既存の変数の状態(CLI引数など)を維持しつつ、YAMLにあれば上書き
        if "show_velocity" in viz_params:
            self.show_velocity = viz_params["show_velocity"]
            
        if "show_lidar" in viz_params:
            self.show_lidar_visualization = viz_params["show_lidar"]

        if "show_trail" in viz_params:
            self.show_trail = viz_params["show_trail"]

        # 新規フラグ: VWA*パス表示用 (デフォルトTrue)
        self.show_global_path = viz_params.get("show_global_path", True)

        self.show_margin_map = viz_params.get("show_margin_map", True)
        self.show_robot_margin = viz_params.get("show_robot_margin", True)
        self.show_formation_margin = viz_params.get("show_formation_margin", True)

        # InfoPanelの制御用変数 (後でInfoPanel初期化後に適用するため変数に入れておく)
        self.show_info_panel = viz_params.get("show_info_panel", show_info_panel)

        self.show_debug_text = viz_params["show_debug_text"]

        self.recording_mode = viz_params.get("recording_mode", "full")
        print(f"[INFO] Recording mode: {self.recording_mode}")
        self.recording_fps = viz_params.get("recording_fps", 10)
        self.video_dir = "."
        # ==============================

        # CLI引数がある場合は、それを「側面間距離」として優先使用
        if formation_spacing_m is not None:
            if formation_spacing_m <= 0:
                raise ValueError("formation_spacing_m must be positive")
            previous_research_params["formation_spacing_m"] = formation_spacing_m

        # パラメータ取得
        self.robot_radius_m = float(previous_research_params["robot_radius_m"])
        self.env.ROBOT_BODY_RADIUS_M = self.robot_radius_m
        self.raw_start = previous_research_params.get("start_pos_m")
        self.raw_goal = previous_research_params.get("goal_pos_m")
        self.fixed_start_pos_m = (float(self.raw_start[0]), float(self.raw_start[1]))
        self.fixed_start_theta = float(self.raw_start[2])
        self.fixed_goal_pos_m = (float(self.raw_goal[0]), float(self.raw_goal[1]))
        self.fixed_goal_theta = float(self.raw_goal[2])

        # if s_side is not None:
        #     self.default_formation_spacing_m = s_side
        #     self.default_formation_spacing = s_side * self.env.PIXELS_PER_METER

        auto_start = previous_research_params.get("auto_start")
        if auto_start:
            self.start_policy_control()


        # 地図画像を保持する変数を初期化
        self.map_surface: Optional[pygame.Surface] = None 
        self._brake_font: Optional[pygame.font.Font] = None
        self._obstacle_mask: Optional[np.ndarray] = None
        self._obstacle_distance_map: Optional[np.ndarray] = None
        self.policy_model_path = policy_model_path
        self.formation_vertical_constraint_enabled = True

        self.robot_states: List[RobotState] = []
        for i in range(resolved_robot_count):
            name = f"robot_{i}"
            color = color_from_palette(i)
            robot = RobotState(
                name=name,
                position=(self.env.SCREEN_CENTER_X, self.env.SCREEN_CENTER_Y),
                theta=self.fixed_start_theta,
                color=color,
                path=deque(maxlen=self.path_history or None),
            )
            # 軌跡用の属性を初期化
            robot.trail_points = []
            robot.trail_step_counter = 0
            robot.trail_color = color
            robot.trail_width = self.trail_width
            
            self.robot_states.append(robot)
            self.robot_states[-1].sensor = self._create_sensor()

        self.formation = FormationManager(
            self,
            lookback=previous_research_params.get("centroid_look_back_steps")
        )

        self.formation.follow_formation_targets = True
        self.formation.weights = previous_research_params.get("optimizer_weights")
        self.omega_f = previous_research_params.get("omega_f")
        self.policy_controller.params["OMEGA_FIXED"] = self.omega_f
        

        self._last_display_size = None
        self._restart_snapshot = None

        # --- サイドバー設定 ---
        self.sidebar_width = 300
        self.sidebar_surface = pygame.Surface((self.sidebar_width, self.env.SCREEN_H))
        self.sidebar_font = pygame.font.SysFont("consolas", 16, bold=True)
        # -------------------

        # ディスプレイ設定
        if self.show_info_panel:
            total_width = self.env.SCREEN_W + self.sidebar_width
        else:
            total_width = self.env.SCREEN_W
        if self.env.render_mode == "human":
            pygame.display.set_mode((total_width, self.env.SCREEN_H))
            self.display_surface = pygame.display.get_surface()
            self._last_display_size = self.display_surface.get_size()
            self.env.screen = pygame.Surface((self.env.SCREEN_W, self.env.SCREEN_H))
            self.env.map_creator.map = self.env.screen
            self.env.clear_all_layers()
        else:
            self.display_surface = pygame.Surface((total_width, self.env.SCREEN_H))
            self._last_display_size = self.display_surface.get_size()
        self.simulation_surface = self.env.screen


        # マップ記述
        self.map_descriptions = {
            "trap_shaped": "Trap-shaped course...",
            "zigzag": "Zigzag corridors...",
            "random": "Randomly distributed obstacles...",
            "empty": "Open field...",
            "image": self._describe_image_map(),
        }

        # 指定されたマップタイプ、またはデフォルト設定で初期化（1回のみ実行）
        target_map_type = map_type if map_type is not None else self.env.map_type
        print(f"[DEBUG] Initializing map with type: {target_map_type}")
        self.set_map(target_map_type)

        


    # --- 録画制御メソッド ---
    def start_recording(self) -> None:
        self.is_recording = True
        self.frames = []
        print("[INFO] Video recording started.")

    def stop_recording(self, status) -> None:
        self.is_recording = False
        print(f"[INFO] Recording stopped. Saving {len(self.frames)} frames...")
        if not self.frames:
            return

        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.video_dir[-1] != "/":
                self.video_dir + "/"
            filename = f"{self.video_dir}/sim_{timestamp}_{status}.mp4"
            print(filename)
            # MP4保存 (fps=30)
            imageio.mimsave(filename, self.frames, fps=self.recording_fps)
            print(f"[INFO] Video saved to {filename}")
        except ImportError:
            print("[ERROR] 'imageio' not found. Please install it (pip install imageio[ffmpeg]) to save videos.")
        except Exception as e:
            print(f"[ERROR] Failed to save video: {e}")
        
        self.frames = [] # メモリ解放
    # ---------------------------
    
    def start_policy_control(self):
        controller = self._policy_controller
        if controller is None and self.policy_model_path:
            controller = PolicyController(self, self.policy_model_path)
            self._policy_controller = controller

        if controller is not None:
            previous_state = controller.active
            new_state = controller.toggle()
            if new_state != previous_state:
                self.policy_control_active = new_state
                if new_state:
                    self.manual_control = False
                    self.start_recording()

                    # --- 計測データのリセット ---
                    print("[INFO] Resetting metrics for new episode.")
                    self.metrics_data = {
                    "total_steps": 0,
                    "success": False,
                    "saved": False,
                    "wide_area_steps": 0,
                    "maintained_steps": {eps: 0 for eps in self.test_epsilons},
                    "frt_events": {eps: [] for eps in self.test_epsilons},
                    "current_frt_start": {eps: None for eps in self.test_epsilons},
                    "is_in_wide_area": False,
                    "prev_centroid_m": None,
                    "fps_sum": 0.0,
                    "fps_count": 0
                }
                    # --------------------------------
                else:
                    self.manual_control = self._manual_control_enabled
        else:
            print(
                "Policy control is unavailable. Call load_policy_model() to configure a model."
            )

    def _describe_image_map(self) -> str:
        """Return a human-friendly description for the image map configuration."""

        configured_path = self.env.image_map_config.get("path")
        if self.env.image_map_path is not None:
            return (
                "Obstacle layout loaded from configuration image file "
                f"{self.env.image_map_path}."
            )
        if configured_path:
            return (
                "Image-based map configured but the bitmap could not be resolved. "
                "Check the path in obstacles.image_map.path."
            )
        return (
            "Image-based map type. Configure obstacles.image_map.path in the "
            "YAML to enable loading an external bitmap."
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def fps(self) -> float:
        return self._fps

    @property
    def policy_controller(self) -> Optional[PolicyController]:
        return self._policy_controller

    # ------------------------------------------------------------------
    # Coordinate conversion helpers
    # ------------------------------------------------------------------
    def meters_to_pixels(self, x_m: float, y_m: float) -> Position:
        px = self.env.SCREEN_CENTER_X + x_m * self.env.PIXELS_PER_METER
        py = self.env.SCREEN_CENTER_Y - y_m * self.env.PIXELS_PER_METER
        return px, py

    def pixels_to_meters(self, px: float, py: float) -> Position:
        x_m = (px - self.env.SCREEN_CENTER_X) / self.env.PIXELS_PER_METER
        y_m = (self.env.SCREEN_CENTER_Y - py) / self.env.PIXELS_PER_METER
        return x_m, y_m

    # ------------------------------------------------------------------
    # Robot management API
    # ------------------------------------------------------------------
    def add_robot(
        self,
        name: str,
        *,
        color: Optional[Color] = None,
        position: Optional[Position] = None,
        theta: float = 0.0,
    ) -> int:
        color = color or color_from_palette(len(self.robot_states))
        position = position or (self.env.SCREEN_CENTER_X, self.env.SCREEN_CENTER_Y)
        robot = RobotState(
            name=name,
            color=color,
            position=position,
            theta=self.fixed_start_theta,
            path=deque(maxlen=self.path_history or None),
        )
        robot.trail_points = []
        robot.trail_step_counter = 0
        robot.trail_color = color
        robot.trail_width = self.trail_width
        robot.sensor = self._create_sensor()
        self.robot_states.append(robot)
        return len(self.robot_states) - 1

    def load_policy_model(self, model_path: str) -> None:
        self.policy_model_path = model_path
        self._policy_controller = PolicyController(self, model_path)
        self.policy_control_active = False
        self.manual_control = self._manual_control_enabled

    def set_robot_pose(
        self,
        robot_index: int,
        position: Position,
        theta: float,
        *,
        record_path: bool = True,
        in_meters: bool = False,
    ) -> None:
        robot = self.robot_states[robot_index]
        if in_meters:
            position = self.meters_to_pixels(*position)

        self._update_robot_motion(robot, (float(position[0]), float(position[1])))
        robot.theta = theta
        if record_path and self.path_history != 0:
            robot.path.append((robot.position, robot.theta))

    def set_robot_target(
        self,
        robot_index: int,
        target: Optional[Position],
        *,
        in_meters: bool = False,
        target_theta: Optional[float] = None,
    ) -> None:
        robot = self.robot_states[robot_index]
        if target is not None and in_meters:
            target = self.meters_to_pixels(*target)
        robot.target = target
        if target is None:
            robot.target_theta = None
        else:
            if target_theta is None:
                target_theta = self._infer_target_heading(robot.position, target)
            robot.target_theta = target_theta
        # FormationManagerにターゲットを通知して、リーダー/フォロワーの追従ロジックを有効化する
        self.formation.register_direct_target(robot_index, target, robot.target_theta)
        if target is None:
            self.formation.clear()

    def clear_paths(self) -> None:
        for robot in self.robot_states:
            robot.path.clear()
            if hasattr(robot, 'trail_points'):
                robot.trail_points.clear()
            if hasattr(robot, 'trail_step_counter'):
                robot.trail_step_counter = 0

    def _reinit_layers(self) -> None:
        """Force re-initialization of layers with transparency."""
        w, h = self.env.SCREEN_W, self.env.SCREEN_H
        
        # ロボット描画レイヤーを透明(SRCALPHA)で作り直す
        self.env.robot_and_lidar_layer = pygame.Surface((w, h), pygame.SRCALPHA)
        self.env.robot_and_lidar_layer.fill((0, 0, 0, 0))
        
        # 障害物レイヤーも同様に
        self.env.obstacle_layer = pygame.Surface((w, h), pygame.SRCALPHA)
        self.env.obstacle_layer.fill((0, 0, 0, 0))
        
        # 画面自体のSurfaceも確保（これは不透明でOK）
        if self.env.screen is None or self.env.screen.get_size() != (w, h):
            self.env.screen = pygame.Surface((w, h))
            
        print("[DEBUG] Layers re-initialized with SRCALPHA.")

    def set_map(self, map_type: str) -> None:
        self.map_descriptions["image"] = self._describe_image_map()
        if map_type == "image":
            configured_path = self.env.image_map_config.get("path")
            if not configured_path and not self._image_map_path_override:
                pass 

        # --- 画像パッチ（白黒画像から障害物を生成する処理） ---
        import types
        def patched_create_image_map(self_mc, image_path):
            if image_path is None: raise ValueError("image_path missing")
            if self_mc.obstacle_layer is None: raise ValueError("obstacle_layer missing")
            
            # レイヤーをクリア
            self_mc.obstacle_layer.fill((0, 0, 0, 0))
            
            # 画像を読み込み
            loaded_surface = pygame.image.load(str(image_path))
            scaled_surface = pygame.transform.smoothscale(
                loaded_surface, (self_mc.MAP_W, self_mc.MAP_H)
            )
            
            # 明るさによる壁判定（黒っぽい部分を壁とする）
            array3d = pygame.surfarray.array3d(scaled_surface)
            brightness = np.mean(array3d, axis=2)
            is_wall = brightness < 200 
            
            # 障害物レイヤーの作成
            result_surf = pygame.Surface((self_mc.MAP_W, self_mc.MAP_H), pygame.SRCALPHA)
            res_rgb = pygame.surfarray.pixels3d(result_surf)
            res_alpha = pygame.surfarray.pixels_alpha(result_surf)
            
            # 障害物の色を適用
            obs_color = pygame.Color(self_mc.OBSTACLE_COLOR)
            res_rgb[is_wall] = (obs_color.r, obs_color.g, obs_color.b)
            res_alpha[is_wall] = 255 # 壁は不透明
            res_alpha[~is_wall] = 0  # 床は透明
            
            del res_rgb, res_alpha
            
            self_mc.obstacle_layer.blit(result_surf, (0, 0))
            print(f"[DEBUG] Patched map creation applied.")

        # パッチを適用
        self.env.map_creator._create_image_map = types.MethodType(patched_create_image_map, self.env.map_creator)
        # -----------------------------------------------

        self._invalidate_obstacle_cache()
        self._capture_restart_snapshot(map_type)
        
        # 1. SimCore側のリセット
        # これにより self.env.obstacle_layer に「壁のみ（背景透明）」が描画される
        self.env.reset(map_type)

        # 2. 背景表示用の画像ロード（obstacle_layerには触らない！）
        target_image_path = self._image_map_path_override
        if not target_image_path and self.env.image_map_config.get("path"):
            target_image_path = str(self.env.image_map_config.get("path"))

        if map_type == "image" and target_image_path:
            try:
                print(f"[DEBUG] Loading visual background map from: {target_image_path}")
                image_surf = pygame.image.load(target_image_path)
                if image_surf.get_size() != (self.env.SCREEN_W, self.env.SCREEN_H):
                    image_surf = pygame.transform.scale(image_surf, (self.env.SCREEN_W, self.env.SCREEN_H))
                
                # 表示用なので不透明でOK
                self.map_surface = image_surf.convert()
            except Exception as e:
                print(f"[ERROR] Failed to load image map for display: {e}")
                self.map_surface = self.env.screen.copy()
        else:
            self.map_surface = self.env.screen.copy()

        # =========================================================================
        # マージン描画順序の制御
        # =========================================================================
        # 1. 現在の壁（障害物）をコピーして退避
        walls_surface = self.env.obstacle_layer.copy()
        
        # 2. マスク（マージン計算用）を取得
        mask = self._get_obstacle_mask()

        self.vw_planner.update_map(mask)
        
        # 3. 障害物レイヤーを一度クリア
        self.env.obstacle_layer.fill((0, 0, 0, 0))
        
        # 4. マージン（インフレート領域）を描画 [下層]
        # self.vw_planner.update_map(mask)
        # if self.show_margin_map:
        #     self.vw_planner.draw_inflation(self.env.obstacle_layer)
        
        # 5. 退避しておいた壁を上から重ねる [上層]
        self.env.obstacle_layer.blit(walls_surface, (0, 0))
        # =========================================================================

        self.clear_paths()
        print("[DEBUG] Recentering robots...")
        self._recenter_robots()
        print("[DEBUG] Robots recentered.")

    def _available_map_types(self) -> List[str]:
        map_types = ["trap_shaped", "zigzag", "random", "empty"]
        configured_path = self.env.image_map_config.get("path")
        if configured_path:
            map_types.append("image")
        return map_types

    def _capture_restart_snapshot(self, map_type: str) -> None:
        rng_state = copy.deepcopy(self.env.np_random.bit_generator.state)
        self._restart_snapshot = {
            "map_type": map_type,
            "rng_state": rng_state,
        }

    def _restart_current_map(self) -> bool:
        snapshot = self._restart_snapshot
        if snapshot is None:
            print("Restart unavailable: no map snapshot captured yet.")
            return False

        map_type = snapshot["map_type"]
        rng_state = snapshot["rng_state"]

        self.env.np_random.bit_generator.state = copy.deepcopy(rng_state)
        self._invalidate_obstacle_cache()
        self._capture_restart_snapshot(map_type)
        self.env.reset(map_type)
        self.clear_paths()
        self._recenter_robots()
        # self.formation.clear()
        return True
    
    # --- Sidebar Drawing ---
    def _draw_sidebar(self) -> None:
        self.sidebar_surface.fill((20, 20, 20)) # Dark gray background

        # lookahead = self.team_nav.lookahead_dist_m if hasattr(self.team_nav, "lookahead_dist_m") else "?"
        # spacing = self.team_nav.robot_spacing_m if hasattr(self.team_nav, "robot_spacing_m") else "?"
        # margins = self.vw_planner.config.InflateRadiusListM if hasattr(self, "vw_planner") else []
        # margin_val = margins[0] if margins else "?"
        
        # Policy Params (dummy for now)
        brake_dist = 2.0 
        brake_angle = 20 

        avg_fps = 0.0
        if self.metrics_data.get("fps_count", 0) > 0:
            avg_fps = self.metrics_data["fps_sum"] / self.metrics_data["fps_count"]

        texts = [
            f"=== Simulation Status ===",
            f"Sim Time: {pygame.time.get_ticks()/1000:.1f} s",
            f"Sim FPS:  {self.fps:.1f}",
            f"Avg FPS:  {avg_fps:.1f}",
            f"",
            f"=== Params ===",
            f"Start Pos: {self.raw_start}",
            f"Goal Pos:  {self.raw_goal}",
            f"Lookback Steps: {self.formation._centroid_look_back_steps} steps",
            f"Weight[0]:      {self.formation.weights[0]}",
            f"Weight[1]:      {self.formation.weights[1]}",
            f"Weight[2]:      {self.formation.weights[2]}",
            f"Weight[3]:      {self.formation.weights[3]}",
            f"Weight[4]:      {self.formation.weights[4]}",
            f"Omega_f:        {self.omega_f} rad/s",
            f"Brake Dist:     {brake_dist} m",
            f"Brake Angle:    {brake_angle} deg",
            f"",
            f"=== Recording ===",
            f"State:    {'[REC]' if self.is_recording else 'STOP'}",
            f"Mode:     {self.recording_mode}",
            f"FPS:      {self.recording_fps}",
            f"Frames:   {len(self.frames)}"
        ]

        padding_x = 10
        padding_y = 10
        line_height = 25

        for i, text in enumerate(texts):
            color = (200, 200, 200)
            if "[REC]" in text: color = (255, 50, 50)
            if "Mode:" in text: color = (100, 200, 255)
            
            surface = self.sidebar_font.render(text, True, color)
            self.sidebar_surface.blit(surface, (padding_x, padding_y + i * line_height))
        
        pygame.draw.line(self.sidebar_surface, (100, 100, 100), (0, 0), (0, self.env.SCREEN_H), 2)

    # ------------------------------------------------------------------
    # Rendering utilities
    # ------------------------------------------------------------------
    def draw_scene(self) -> pygame.Surface:
        self.simulation_surface = self.env.screen
        if self.env.render_mode == "human":
            self._clock.tick(60)
        else:
            self._clock.tick()
        self._fps = self._clock.get_fps()

        # === 平均FPS計測 (ポリシー実行中のみ) ===
        if self.policy_control_active and self.metrics_enabled:
            # 最初の数フレームは0になることがあるため除外してもよいが、単純な平均をとる
            self.metrics_data["fps_sum"] += self._fps
            self.metrics_data["fps_count"] += 1
        # ==========================================

        # if self.policy_control_active:
        #      self.team_nav.update()

        # ---------------------------------------------------------
        # 描画のベースを作成
        # 1. まず背景色で塗りつぶす（これで黒画面を防ぐ）
        # ---------------------------------------------------------
        self.env.screen.fill(self.env.BACKGROUND_COLOR)

        # 2. 地図画像を重ねる
        if self.map_surface is not None:
            self.env.screen.blit(self.map_surface, (0, 0))

        # 3. ロボットレイヤーをクリアして再描画
        self.env.clear_robot_and_lidar_layer()
        # 万が一 clear_robot_and_lidar_layer が不十分な場合の保険
        if hasattr(self.env, "robot_and_lidar_layer"):
             self.env.robot_and_lidar_layer.fill((0, 0, 0, 0))

        # --- 追加: チーム重心のマージンを描画 ---
        # self._draw_team_centroid_margin()
        # -------------------------------------

        for robot in self.robot_states:
            self._update_lidar(robot)
            if robot.path and robot.path_enabled:
                self._draw_path(robot)
            if robot.target is not None:
                self._draw_target(robot)

            # 軌跡の描画 (リスト内の全点を描画)
            if self.show_trail and hasattr(robot, 'trail_points'):
                for pt in robot.trail_points:
                    pygame.draw.circle(
                        self.env.robot_and_lidar_layer, 
                        robot.trail_color, 
                        (int(pt[0]), int(pt[1])), 
                        robot.trail_width # 点のサイズ
                    )
            self._draw_robot(robot)
            if self.show_labels and self.show_debug_text:
                self._draw_label(robot)

        # --- VWA*パスの描画制御 ---
        # if self.show_global_path:
        #     self.team_nav.draw(self.env.robot_and_lidar_layer)
        # elif self.team_nav.team_goal:
        #     pygame.draw.circle(self.env.robot_and_lidar_layer, (255, 0, 0), (int(self.team_nav.team_goal[0]), int(self.team_nav.team_goal[1])), 8)
        # -------------------------------

        self.formation.update_visualisation()
        self.formation.draw()
        self.formation.draw_parallel_overlay()

        self._draw_hud()

        # ロボットレイヤーを画面に合成
        self.env.screen.blit(self.env.robot_and_lidar_layer, (0, 0))

        self.display_surface.fill(self.env.BACKGROUND_COLOR)
        # Left: Simulation
        self.display_surface.blit(self.simulation_surface, (0, 0))
        # Right: Sidebar
        if self.show_info_panel:
            self._draw_sidebar()
            self.display_surface.blit(self.sidebar_surface, (self.env.SCREEN_W, 0))


        if self.env.render_mode == "human":
            pygame.display.flip()

        # --- 録画中のフレーム保存 ---
        if self.is_recording:
            # display_surface(表示されている画面そのまま)をキャプチャ
            # pygameは (width, height, 3) で返すので、(height, width, 3) に変換
            try:
                target_surface = self.display_surface
                if self.recording_mode == "simulation_only":
                    # シミュレーション部分のみのSurfaceを使用
                    target_surface = self.simulation_surface

                # array化して保存
                array = pygame.surfarray.array3d(target_surface)
                array = np.transpose(array, (1, 0, 2))
                self.frames.append(array)
            except Exception:
                pass # 失敗しても止めない
        # ---------------------------------

        return self.display_surface

    def draw_rgb_array(self):
        import numpy as np

        surface = self.draw_scene()
        array = pygame.surfarray.array3d(surface)
        return np.transpose(array, (1, 0, 2))

    # クラス内の適当な場所に追加してください（process_eventsの直前など）
    def _save_metrics(self) -> None:
        """実験結果（所要時間）をCSVに保存する"""
        if self.metrics_data["saved"]:
            return

        md = self.metrics_data
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = "experiment_results_previous.csv" # ファイル名を区別
        
        # ステップ数から時間を計算 (30FPS想定)
        duration_sec = md["total_steps"] * (1.0/10.0)

        wide_steps = md["wide_area_steps"]

        avg_fps = 0.0
        if md.get("fps_count", 0) > 0:
            avg_fps = md["fps_sum"] / md["fps_count"]

        header = ["Timestamp", "Result", "Duration_sec", "Avg_FPS", "Wide_Steps"]
        row = [timestamp, "Success" if md["success"] else "Failed", f"{duration_sec:.2f}", f"{avg_fps:.1f}", wide_steps]

        for eps in self.test_epsilons:
            # FMR
            maintained = md["maintained_steps"][eps]
            fmr = maintained / wide_steps if wide_steps > 0 else 0.0
            
            # FRT (平均)
            frt_list = md["frt_events"][eps]
            avg_frt = np.mean(frt_list) if frt_list else 0.0
            count_frt = len(frt_list)

            header.extend([f"FMR_e{eps}", f"FRT_e{eps}", f"Recov_Count_e{eps}"])
            row.extend([f"{fmr:.4f}", f"{avg_frt:.1f}", count_frt])
        
        try:
            file_exists = os.path.isfile(filename)
            with open(filename, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(header)
                
                writer.writerow(row)
            print(f"\n[INFO] Result saved to {filename} (Time: {duration_sec:.2f}s)")
            self.metrics_data["saved"] = True
            
        except Exception as e:
            print(f"[ERROR] Failed to save CSV: {e}")

    # === FMR / FRT 計測ロジック ===
    def _update_metrics(self) -> None:
        """重心位置と隊形誤差に基づき FMR, FRT を更新する"""
        if not self.policy_control_active or not self.metrics_enabled:
            return
        
        # ロボット位置 (メートル)
        positions_m = np.array([self.pixels_to_meters(*r.position) for r in self.robot_states])
        centroid_m = np.mean(positions_m, axis=0)
        
        # --- 1. 領域判定 (Wide or Narrow) ---
        # 重心位置のピクセル座標を取得
        cx_px, cy_px = self.meters_to_pixels(centroid_m[0], centroid_m[1])
        ix, iy = int(cx_px), int(cy_px)
        
        # 画面外チェック
        h, w = self.vw_planner.label_maps[0].shape
        if not (0 <= ix < w and 0 <= iy < h):
            is_wide = True # 画面外は一旦Wide扱い（あるいは計測除外）
        else:
            # label_maps[0] は「広いマージンでの障害物領域(True)」
            # つまり False ならフリー＝広い領域 (Wide Area)
            # True なら障害物または狭い領域 (Narrow Area)
            is_occupied_wide = self.vw_planner.label_maps[0][iy, ix]
            is_wide = not is_occupied_wide

        prev_is_wide = self.metrics_data["is_in_wide_area"]
        self.metrics_data["is_in_wide_area"] = is_wide

        # --- 2. 理想隊形との誤差計算 ---
        # 進行ベクトル計算
        if self.metrics_data["prev_centroid_m"] is not None:
            move_vec = centroid_m - self.metrics_data["prev_centroid_m"]
        else:
            move_vec = np.array([0.0, 0.0])
        self.metrics_data["prev_centroid_m"] = centroid_m

        dist = np.linalg.norm(move_vec)
        if dist > 1e-4:
            unit_v = move_vec / dist
        else:
            # 動いていない場合は、平均向きなどを使うか、計算スキップ
            # ここでは便宜上、ロボット0の向きを使うなどの代替案も考えられるが
            # 前回値を保持していないため、X軸正方向(仮)またはスキップとする
            unit_v = np.array([1.0, 0.0]) 

        # 理想位置：進行方向に垂直（右側）に展開
        # unit_v = (vx, vy) -> right_vec = (vy, -vx)  (数学座標系)
        # ※Pygame座標系(y下)でもベクトル演算としては同様の直交関係でOK
        right_vec = np.array([unit_v[1], -unit_v[0]])

        num = len(self.robot_states)
        spacing = self.default_formation_spacing_m
        offsets = (np.arange(num) - (num - 1) / 2) * spacing

        # 各ロボットを理想ラインに投影して、順序を合わせる（ソート）
        # これによりIDが入れ替わっても幾何学的な形状誤差を評価できる
        projections = [np.dot(pos - centroid_m, right_vec) for pos in positions_m]
        sorted_indices = np.argsort(projections)
        sorted_positions = positions_m[sorted_indices]

        errors = []
        for i in range(num):
            ideal_pos = centroid_m + offsets[i] * right_vec
            err = np.linalg.norm(sorted_positions[i] - ideal_pos)
            errors.append(err)
        
        max_error = max(errors) if errors else 0.0

        # --- 3. 指標の更新 (各εに対して) ---
        current_step = self.metrics_data["total_steps"]

        # FMR用の分母加算
        if is_wide:
            self.metrics_data["wide_area_steps"] += 1

        for eps in self.test_epsilons:
            is_maintained = (max_error <= eps)

            # --- FMR (Formation Maintenance Ratio) ---
            if is_wide and is_maintained:
                self.metrics_data["maintained_steps"][eps] += 1

            # --- FRT (Formation Recovery Time) ---
            # トリガー: Narrow -> Wide への切り替わり
            if (not prev_is_wide) and is_wide:
                # 計測開始（上書き）
                self.metrics_data["current_frt_start"][eps] = current_step
            
            # 計測終了判定
            start_step = self.metrics_data["current_frt_start"][eps]
            if start_step is not None:
                # 広い領域にいる状態で、誤差がε以下になったら終了
                if is_wide and is_maintained:
                    duration = current_step - start_step
                    self.metrics_data["frt_events"][eps].append(duration)
                    self.metrics_data["current_frt_start"][eps] = None # 計測完了
                elif not is_wide:
                    # もし回復せずに狭い領域に戻ってしまった場合は計測キャンセル
                    self.metrics_data["current_frt_start"][eps] = None
    # ============================================

    def process_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_c:
                    self._cycle_map()
                elif event.key == pygame.K_SPACE:
                    self.set_map(self.env.map_type)
                elif event.key == pygame.K_r:
                    if not self._restart_current_map():
                        print("Press SPACE to generate a map before restarting with R.")
                elif event.key == pygame.K_TAB:
                    self._select_next_robot()
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    index = event.key - pygame.K_1
                    if index < len(self.robot_states):
                        self.selected_robot_index = index
                elif event.key == pygame.K_0 and len(self.robot_states) >= 10:
                    self.selected_robot_index = 9
                elif event.key == pygame.K_f:
                    enabled = self.formation.toggle_follow_formation()
                    if enabled:
                        print("Formation following enabled: robots will pursue formation targets.")
                    else:
                        print("Formation following disabled: robots will pursue direct goals.")
                elif event.key == pygame.K_v:
                    enabled = self.formation.toggle_vertical_constraint()
                    state_text = "enabled" if enabled else "disabled"
                    action_text = "respect" if enabled else "ignore"
                    print(
                        "Vertical formation heading constraint"
                        f" {state_text}: formation search will {action_text} perpendicular targets."
                    )
                elif event.key == pygame.K_d:
                    self.show_debug_text = not self.show_debug_text
                elif event.key == pygame.K_l:
                    self.show_lidar_visualization = not self.show_lidar_visualization
                    # robot = self.robot_states[self.selected_robot_index]
                    # robot.lidar_enabled = not robot.lidar_enabled
                elif event.key == pygame.K_p:
                    robot = self.robot_states[self.selected_robot_index]
                    robot.path_enabled = not robot.path_enabled
                elif event.key == pygame.K_m:
                    self.start_policy_control()

        if self.manual_control and self.robot_states:
            self._operate_moving()

        if self._policy_controller is not None:
            was_active = self.policy_control_active # ステップ前の状態
            self._policy_controller.step()
            self.policy_control_active = self._policy_controller.active

            # --- ステップカウントとゴール判定 ---
            if self.policy_control_active:
                self.metrics_data["total_steps"] += 1

                self._update_metrics() # <--- 追加
                
                # ゴール判定 (重心がゴール付近に到達したか)
                if self.fixed_goal_pos_m:
                    positions_m = np.array([self.pixels_to_meters(*r.position) for r in self.robot_states])
                    centroid = np.mean(positions_m, axis=0)
                    goal_arr = np.array(self.fixed_goal_pos_m)
                    
                    dist = np.linalg.norm(centroid - goal_arr)
                    
                    # 1.0m以内ならゴールとみなす
                    if dist < 0.5:
                        print(f"[INFO] Goal Reached! Distance: {dist:.2f}m")
                        self.metrics_data["success"] = True
                        self._save_metrics()
                        
                        # 録画停止などの終了処理
                        if self.is_recording:
                            self.stop_recording("goal")
                        
                        # 制御を止める場合
                        self.policy_control_active = False
                        self._policy_controller.active = False
            # ---------------------------------------------

            # Active -> Inactive に切り替わった場合 (=ゴール到達完了)
            if was_active and not self.policy_control_active:
                if self.is_recording:
                    self.stop_recording("goal")
            # ------------------------------------

    def _draw_final_result(self) -> None:
        """終了時のスナップショット用に、LiDARを除外してきれいに再描画する"""
        
        # 1. 背景のリセット
        self.env.screen.fill(self.env.BACKGROUND_COLOR)
        if self.map_surface is not None:
            self.env.screen.blit(self.map_surface, (0, 0))
            
        # 2. 障害物レイヤーの描画 (マージン含む)
        self.env.screen.blit(self.env.obstacle_layer, (0, 0))
        
        # 3. ロボットレイヤーのクリア
        self.env.robot_and_lidar_layer.fill((0, 0, 0, 0))
        
        # 4. チームマージン
        # self._draw_team_centroid_margin()
        
        # 5. ロボット・パス・履歴の描画 (LiDAR更新は行わない)
        for robot in self.robot_states:
            if robot.path and robot.path_enabled: self._draw_path(robot)
            if robot.target is not None: self._draw_target(robot)
            self._draw_robot(robot) # ここでinterval_positionsも描画される
            if self.show_labels and self.show_debug_text: self._draw_label(robot)
            
        # 6. VWAstarパス
        # self.team_nav.draw(self.env.robot_and_lidar_layer)
        
        # 7. 合成
        self.env.screen.blit(self.env.robot_and_lidar_layer, (0, 0))
        
        # 8. 保存
        try:
            filename = "final_result.png"
            pygame.image.save(self.env.screen, filename)
            print(f"[INFO] Saved final simulation result to: {filename}")
        except Exception as e:
            print(f"[ERROR] Failed to save result image: {e}")

    def close(self) -> None:
        if self.env.screen is not None:
            self._draw_final_result()

        # 終了時にもし録画中なら保存する
        if self.is_recording:
            self.stop_recording("abort")

        pygame.quit()

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    # def _draw_team_centroid_margin(self) -> None:
    #     """チームの重心にフォーメーション用（広い）マージンを描画する"""
    #     if not self.robot_states or not hasattr(self, 'vw_planner') or not self.vw_planner:
    #         return

    #     # チームの重心（Centroid）を計算
    #     avg_x = sum(r.position[0] for r in self.robot_states) / len(self.robot_states)
    #     avg_y = sum(r.position[1] for r in self.robot_states) / len(self.robot_states)

    #     # マージン設定を取得
    #     margins = self.vw_planner.config.InflateRadiusListM
        
    #     # 1つ目のマージン（Index 0: 広い方）があれば描画
    #     if len(margins) > 0:
    #         radius_m = margins[0]
    #         radius_px = int(radius_m * self.env.PIXELS_PER_METER)
    #         color = (0, 255, 255) # シアン
    #         centor_color = (255, 165, 0)
            
    #         # 重心に円を描画
    #         if self.show_formation_margin:
    #             pygame.draw.circle(
    #                 self.env.robot_and_lidar_layer, 
    #                 color, 
    #                 (int(avg_x), int(avg_y)), 
    #                 radius_px, 
    #                 width=1
    #             )
    #         # 重心点も小さく表示（分かりやすさのため）
    #         pygame.draw.circle(
    #             self.env.robot_and_lidar_layer, 
    #             centor_color, 
    #             (int(avg_x), int(avg_y)), 
    #             3
    #         )

    def _draw_robot(self, robot: RobotState) -> None:
        pos = robot.position
        
        # # --- 修正: ロボット個別には「2つ目のマージン（狭い方）」だけを描画 ---
        # if hasattr(self, 'vw_planner') and self.vw_planner and self.show_robot_margin:
        #     margins = self.vw_planner.config.InflateRadiusListM
            
        #     # Index 1 (2番目のマージン) が存在すれば描画
        #     if len(margins) > 1:
        #         margin_m = margins[1]
        #         radius_px = int(margin_m * self.env.PIXELS_PER_METER)
        #         color = (255, 0, 255) # マゼンタ
                
        #         pygame.draw.circle(
        #             self.env.robot_and_lidar_layer, 
        #             color, 
        #             (int(pos[0]), int(pos[1])), 
        #             radius_px, 
        #             width=1
        #         )
        # # -------------------------------------------------------------

        # ロボット本体（物理サイズ）の描画
        body_radius_px = int(self.env.ROBOT_BODY_RADIUS_M * self.env.PIXELS_PER_METER)
        pygame.draw.circle(
            self.env.robot_and_lidar_layer,
            robot.color,
            (int(pos[0]), int(pos[1])),
            body_radius_px,
        )

        # 安全マージン（衝突判定サイズ）の描画
        # pygame.draw.circle(
        #     self.env.robot_and_lidar_layer,
        #     robot.color,
        #     (int(pos[0]), int(pos[1])),
        #     int(self.robot_frame_radius),
        #     width=2,
        # )

        # ヘディング（向き）の描画
        head = self._compute_orientation_endpoint(pos, robot.theta, self.robot_frame_radius)
        pygame.draw.line(
            self.env.robot_and_lidar_layer,
            (255, 255, 255),
            (int(pos[0]), int(pos[1])),
            (int(head[0]), int(head[1])),
            width=2,
        )

        # 選択中のロボット強調
        # if robot is self.robot_states[self.selected_robot_index] and not self.policy_control_active:
        #     pygame.draw.circle(
        #         self.env.robot_and_lidar_layer,
        #         (0, 255, 0),
        #         (int(pos[0]), int(pos[1])),
        #         int(self.robot_frame_radius + 6),
        #         width=2,
        #     )


        if self.show_velocity:
            self._draw_velocity_arrow(robot)
            self._draw_angular_velocity_arc(robot)
            self._draw_speed_bar(robot)
        if robot.braking:
            self._draw_brake_overlay(robot)

    def _draw_velocity_arrow(self, robot: RobotState) -> None:
        v_o = getattr(
            robot,
            "output_velocity",
            getattr(robot, "executable_velocity", 0.0),
        )
        if abs(v_o) < 1e-3:
            return

        pixels_per_meter = self.env.PIXELS_PER_METER
        length_pixels = v_o * pixels_per_meter
        if abs(length_pixels) < 1.0:
            return

        surface = self.env.robot_and_lidar_layer
        start_x, start_y = robot.position
        direction_x = -math.sin(robot.theta)
        direction_y = -math.cos(robot.theta)
        end_x = start_x + direction_x * length_pixels
        end_y = start_y + direction_y * length_pixels

        color = (255, 215, 0)
        width = 4
        start_pos = (int(round(start_x)), int(round(start_y)))
        end_pos = (int(round(end_x)), int(round(end_y)))
        pygame.draw.line(surface, color, start_pos, end_pos, width=width)

        angle = math.atan2(end_y - start_y, end_x - start_x)
        head_length = max(6.0, min(12.0, abs(length_pixels) * 0.25))
        head_angle = math.pi / 8

        left_x = end_x - head_length * math.cos(angle - head_angle)
        left_y = end_y - head_length * math.sin(angle - head_angle)
        right_x = end_x - head_length * math.cos(angle + head_angle)
        right_y = end_y - head_length * math.sin(angle + head_angle)

        pygame.draw.line(
            surface,
            color,
            end_pos,
            (int(round(left_x)), int(round(left_y))),
            width=width,
        )
        pygame.draw.line(
            surface,
            color,
            end_pos,
            (int(round(right_x)), int(round(right_y))),
            width=width,
        )

    def _draw_angular_velocity_arc(self, robot: RobotState) -> None:
        omega_o = getattr(
            robot,
            "output_angular_velocity",
            getattr(robot, "executable_angular_velocity", 0.0),
        )
        if abs(omega_o) < 1e-3:
            return

        arc_angle = min(abs(omega_o), math.tau)
        if arc_angle < 1e-3:
            return

        center_x, center_y = robot.position
        radius = self.robot_frame_radius + 20.0
        direction_x = -math.sin(robot.theta)
        direction_y = -math.cos(robot.theta)
        forward_angle = math.atan2(direction_y, direction_x)
        # Positive angular velocity should appear as a counter-clockwise rotation
        # around the robot. The screen space Y axis is inverted (down is
        # positive), so we flip the sign when mapping from the angular velocity
        # to the direction of the arc sweep.
        sign = -1.0 if omega_o >= 0.0 else 1.0
        segments = max(4, int(math.ceil(arc_angle / (math.pi / 24.0))))

        points: list[tuple[float, float]] = []
        for i in range(segments + 1):
            t = i / segments
            angle = forward_angle + sign * t * arc_angle
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            points.append((x, y))

        if len(points) < 2:
            return

        surface = self.env.robot_and_lidar_layer
        color = (100, 149, 237)
        width = 4
        point_list = [(int(round(x)), int(round(y))) for x, y in points]
        pygame.draw.lines(surface, color, False, point_list, width)

        end_x, end_y = points[-1]
        prev_x, prev_y = points[-2]
        angle = math.atan2(end_y - prev_y, end_x - prev_x)
        head_length = 10.0
        head_angle = math.pi / 8
        left_x = end_x - head_length * math.cos(angle - head_angle)
        left_y = end_y - head_length * math.sin(angle - head_angle)
        right_x = end_x - head_length * math.cos(angle + head_angle)
        right_y = end_y - head_length * math.sin(angle + head_angle)

        pygame.draw.line(
            surface,
            color,
            (int(round(end_x)), int(round(end_y))),
            (int(round(left_x)), int(round(left_y))),
            width=width,
        )
        pygame.draw.line(
            surface,
            color,
            (int(round(end_x)), int(round(end_y))),
            (int(round(right_x)), int(round(right_y))),
            width=width,
        )

    def _draw_speed_bar(self, robot: RobotState) -> None:
        v_max = getattr(self.env, "V_MAX", None)
        if v_max is None or v_max <= 0:
            return

        surface = self.env.robot_and_lidar_layer
        velocity = max(0.0, float(robot.velocity or 0.0))
        ratio = min(velocity / v_max, 1.0)

        bar_width = max(20, int(round(self.robot_frame_radius * 2)))
        bar_height = max(6, int(round(self.robot_radius_pixels * 0.4)))
        bar_offset = self.robot_frame_radius + bar_height + 6

        bar_left = robot.position[0] - bar_width / 2
        bar_top = robot.position[1] - bar_offset
        bar_rect = pygame.Rect(
            int(round(bar_left)),
            int(round(bar_top)),
            bar_width,
            bar_height,
        )

        pygame.draw.rect(surface, (60, 60, 60), bar_rect)

        fill_width = int(round((bar_rect.width - 2) * ratio))
        if fill_width > 0:
            fill_rect = pygame.Rect(
                bar_rect.left + 1,
                bar_rect.top + 1,
                fill_width,
                max(1, bar_rect.height - 2),
            )
            pygame.draw.rect(surface, (0, 200, 100), fill_rect)

        pygame.draw.rect(surface, (20, 20, 20), bar_rect, width=1)

        tick_color = (200, 200, 200)
        pygame.draw.line(
            surface,
            tick_color,
            (bar_rect.left, bar_rect.top - 1),
            (bar_rect.left, bar_rect.bottom + 1),
            width=1,
        )
        pygame.draw.line(
            surface,
            tick_color,
            (bar_rect.right - 1, bar_rect.top - 1),
            (bar_rect.right - 1, bar_rect.bottom + 1),
            width=1,
        )

    def _draw_brake_overlay(self, robot: RobotState) -> None:
        if self._brake_font is None:
            font_size = max(24, int(round(self.robot_radius_pixels*2)))
            self._brake_font = pygame.font.SysFont("arial", font_size, bold=True)

        font = self._brake_font
        text_surface = font.render("B", True, (255, 255, 255))
        shadow_surface = font.render("B", True, (0, 0, 0))
        center = (int(robot.position[0]), int(robot.position[1]))

        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            shadow_rect = shadow_surface.get_rect(
                center=(center[0] + dx, center[1] + dy)
            )
            self.env.robot_and_lidar_layer.blit(shadow_surface, shadow_rect)

        text_rect = text_surface.get_rect(center=center)
        self.env.robot_and_lidar_layer.blit(text_surface, text_rect)

    def _draw_target(self, robot: RobotState) -> None:
        assert robot.target is not None
        color = brighten_color(robot.color)
        pygame.draw.circle(
            self.env.robot_and_lidar_layer,
            color,
            (int(robot.target[0]), int(robot.target[1])),
            max(1, int(round(self.robot_radius_pixels))),
            width=2,
        )
        if robot.target_theta is not None:
            start_x, start_y = robot.target
            end_x, end_y = self._compute_orientation_endpoint(
                robot.target, robot.target_theta, self.robot_frame_radius
            )
            pygame.draw.line(
                self.env.robot_and_lidar_layer,
                color,
                (int(round(start_x)), int(round(start_y))),
                (int(round(end_x)), int(round(end_y))),
                width=2,
            )

    def _draw_path(self, robot: RobotState) -> None:
        points = list(robot.path)
        for (start_pos, _), (end_pos, _) in zip(points[:-1], points[1:]):
            pygame.draw.line(
                self.env.robot_and_lidar_layer,
                with_alpha(robot.color, 0.6),
                (int(start_pos[0]), int(start_pos[1])),
                (int(end_pos[0]), int(end_pos[1])),
                width=2,
            )

    def _draw_label(self, robot: RobotState) -> None:
        font = pygame.font.SysFont("arial", 16)
        lines = [robot.name]
        if robot.target is not None:
            tx, ty = robot.target
            distance = math.hypot(
                tx - robot.position[0], ty - robot.position[1]
            ) / self.env.PIXELS_PER_METER
            lines.append(f"target: {distance:.1f}m")
        if robot.closest_obstacle[0] != float("inf"):
            lines.append(f"obstacle: {robot.closest_obstacle[0]:.1f}m")
        for line_index, text in enumerate(lines):
            label = font.render(text, True, (0, 0, 0))
            text_position = (
                int(robot.position[0] - label.get_width() / 2),
                int(
                    robot.position[1]
                    - self.robot_frame_radius
                    - label.get_height() * (len(lines) - line_index)
                    - 4
                ),
            )
            self.env.robot_and_lidar_layer.blit(label, text_position)

    def _draw_hud(self) -> None:
        return



    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _update_robot_motion(self, robot: RobotState, new_position: Position) -> None:
        previous_position = robot.position
        dx = new_position[0] - previous_position[0]
        dy = new_position[1] - previous_position[1]
        distance_pixels = math.hypot(dx, dy)

        current_time = (
            pygame.time.get_ticks() / 1000.0 if pygame.get_init() else None
        )


        velocity = 0.0
        if (
            current_time is not None
            and robot.last_update_time is not None
            and distance_pixels > 0.0
        ):
            dt = current_time - robot.last_update_time
            if dt > 0:
                distance_m = distance_pixels / self.env.PIXELS_PER_METER
                velocity = distance_m / dt

        robot.velocity = velocity
        robot.last_update_time = current_time
        robot.position = (float(new_position[0]), float(new_position[1]))

        # ステップ数ベースの軌跡記録
        if self.show_trail:
            # 属性の初期化 (念のため)
            if not hasattr(robot, 'trail_step_counter'):
                robot.trail_step_counter = 0

            # 最初の1点を追加
            if not robot.trail_points:
                robot.trail_points.append(robot.position)
                robot.trail_step_counter = 0
            else:
                robot.trail_step_counter += 1
                if robot.trail_step_counter >= self.trail_step_interval:
                    robot.trail_points.append(robot.position)
                    robot.trail_step_counter = 0

    def _create_sensor(self) -> LiDARSensor:
        return LiDARSensor(
            Range=self.env.LIDAR_RANGE_M,
            uncertainty=(0, 0),
            pixels_per_meter=self.env.PIXELS_PER_METER,
            lidar_resolution=self.env.LIDAR_RESOLUTION,
            screen_w=self.env.SCREEN_W,
            screen_h=self.env.SCREEN_H,
        )

    def _recenter_robots(self) -> None:
        if not self.robot_states: return
        
        spacing = self.default_formation_spacing
        
        # --- 修正: 固定スタート位置がある場合の分岐 ---
        if self.fixed_start_pos_m:
            cx_m, cy_m = self.fixed_start_pos_m
            start_theta = self.fixed_start_theta
            cx_px, cy_px = self.meters_to_pixels(cx_m, cy_m)
            
            # 指定座標を中心に横並び配置
            num = len(self.robot_states)
            offsets = (np.arange(num) - (num - 1) / 2) * spacing
            
            # 回転計算
            # Pygame座標系: 0=Up(North), 正=CCW(Left)と仮定
            # Side-by-sideのラインは進行方向に対して垂直(右方向)
            # Heading vector: (-sin(th), -cos(th)) ※draw_robot参照
            # Right vector (Headingから-90度): (cos(th), -sin(th))
            
            ux = math.cos(start_theta)
            uy = -math.sin(start_theta) # Y軸反転のためマイナス
            
            robot_positions = []
            for o in offsets:
                # 重心 + (右方向ベクトル * オフセット)
                px = cx_px + o * ux
                py = cy_px + o * uy
                robot_positions.append((px, py))
            
            print(f"[DEBUG] Placing robots at fixed start: {self.fixed_start_pos_m}, theta: {start_theta:.2f}")
        else:
            # 従来通りランダムエリアから生成
            start_area, _ = self._get_spawn_areas()
            robot_positions = self.formation.generate_randomized_positions(start_area, len(self.robot_states), spacing)
        # ------------------------------------------


        for robot, position in zip(self.robot_states, robot_positions):
            robot.position = (float(position[0]), float(position[1]))
            robot.theta = self.fixed_start_theta
            robot.path.clear()
            if self.show_trail:
                robot.trail_points = [robot.position] # リセット
                robot.trail_step_counter = 0

            robot.sensor_data = []
            robot.closest_obstacle = (float("inf"), 0.0)
            robot.last_update_time = None
            robot.velocity = 0.0
            robot.braking = False
            
        # ゴール設定へ（引数はランダム用エリアだが、メソッド内で固定ゴール判定を行う）
        _, target_area = self._get_spawn_areas()
        self._assign_random_targets(target_area)

    def _get_spawn_areas(self) -> Tuple[Optional[Sequence[float]], Optional[Sequence[float]]]:
        screen_center_xy = (self.env.SCREEN_CENTER_X, self.env.SCREEN_CENTER_Y)
        screen_size_wh = (self.env.SCREEN_W, self.env.SCREEN_H)
        start_spawn_area, target_spawn_area = self.env.map_creator.initialize_map_parameters(
            self.env.map_type,
            screen_center_xy,
            screen_size_wh,
            self.env.ROBOT_RADIUS_PIXELS,
        )
        return start_spawn_area, target_spawn_area

    def _assign_random_targets(self, target_spawn_area: Optional[Sequence[float]]) -> None:
        if not self.robot_states: return
        spacing = self.default_formation_spacing
        
        # --- 修正: 固定ゴール位置がある場合の分岐 ---
        if self.fixed_goal_pos_m:
            cx_m, cy_m = self.fixed_goal_pos_m
            goal_theta = self.fixed_goal_theta
            cx_px, cy_px = self.meters_to_pixels(cx_m, cy_m)
            
            # 指定座標を中心に横並び配置（ターゲット）
            num = len(self.robot_states)
            offsets = (np.arange(num) - (num - 1) / 2) * spacing

            ux = math.cos(goal_theta)
            uy = -math.sin(goal_theta) # Y軸反転のためマイナス
            
            target_positions = []
            for o in offsets:
                # 重心 + (右方向ベクトル * オフセット)
                px = cx_px + o * ux
                py = cy_px + o * uy
                target_positions.append((px, py))

            # target_positions = [(cx_px + o, cy_px) for o in offsets]
            
            print(f"[DEBUG] Placing robots at fixed goal: {self.fixed_goal_pos_m}, theta: {goal_theta:.2f}")
        else:
            # 従来通りランダムエリアから生成
            target_positions = self.formation.generate_randomized_positions(target_spawn_area, len(self.robot_states), spacing)
        # ----------------------------------------
        
        if not target_positions: return

        for index, (robot, target) in enumerate(zip(self.robot_states, target_positions)):
            heading = self._infer_target_heading(robot.position, target)
            robot.target = target
            robot.target_theta = goal_theta
            self.formation.register_direct_target(index, target, heading)
            
        if target_positions:
            avg_x = sum(t[0] for t in target_positions)/len(target_positions)
            avg_y = sum(t[1] for t in target_positions)/len(target_positions)
            print(f"Auto-setting team goal: {avg_x:.1f}, {avg_y:.1f}")
            # self.team_nav.set_team_goal((avg_x, avg_y))

    def _invalidate_obstacle_cache(self) -> None:
        self._obstacle_mask = None
        self._obstacle_distance_map = None

    def _get_obstacle_mask(self) -> np.ndarray:
        if self._obstacle_mask is None:
            obstacle_surface = self.env.obstacle_layer
            obstacle_rgb = pygame.surfarray.array3d(obstacle_surface)
            color_array = np.array(self.obstacle_color_rgb, dtype=obstacle_rgb.dtype)
            mask = np.all(obstacle_rgb == color_array[None, None, :], axis=2)
            mask = mask.swapaxes(0, 1)
            mask[0, :] = True
            mask[-1, :] = True
            mask[:, 0] = True
            mask[:, -1] = True
            self._obstacle_mask = mask
        return self._obstacle_mask

    def _get_obstacle_distance_map(self) -> np.ndarray:
        if self._obstacle_distance_map is None:
            mask = self._get_obstacle_mask()
            free_space = ~mask
            distance_pixels = ndimage.distance_transform_edt(free_space)
            self._obstacle_distance_map = distance_pixels / self.env.PIXELS_PER_METER
        return self._obstacle_distance_map

    # ------------------------------------------------------------------
    # Perception driven helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_orientation_endpoint(
        origin: Position, theta: float, length: float
    ) -> Position:
        return (
            float(origin[0] - length * math.sin(theta)),
            float(origin[1] - length * math.cos(theta)),
        )

    def _infer_target_heading(self, start: Position, target: Position) -> float:
        dx = float(target[0]) - float(start[0])
        dy = float(target[1]) - float(start[1])
        norm = math.hypot(dx, dy)
        if norm <= 0.0:
            return 0.0

        vx = dx / norm
        vy = dy / norm
        heading = math.atan2(-vx, -vy)
        return self._normalize_angle(heading)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize *angle* to the range [-pi, pi]."""

        return (angle + math.pi) % (2 * math.pi) - math.pi

    def _robot_relative_angle(self, robot: RobotState, point: Position) -> float:
        """Return the LiDAR-relative angle from *robot* to *point*.

        The renderer uses a screen coordinate system where ``y`` increases
        downward.  This helper mirrors the transformations applied by the
        LiDAR sensor so that formation planning can align world-space
        queries with per-robot measurements.
        """

        dx = point[0] - robot.position[0]
        dy = point[1] - robot.position[1]
        if dx == 0.0 and dy == 0.0:
            return 0.0

        global_angle = math.atan2(-dx, -dy)
        relative_angle = robot.theta - math.pi / 2 - global_angle
        return self._normalize_angle(relative_angle)

    def _robot_query_distance(self, robot: RobotState, point: Position) -> float:
        """Return the Euclidean distance in meters from *robot* to *point*."""

        dx = point[0] - robot.position[0]
        dy = point[1] - robot.position[1]
        return math.hypot(dx, dy) / self.env.PIXELS_PER_METER

    def _robot_observation_samples(
        self, robot: RobotState
    ) -> List[Tuple[float, float]]:
        """Collect LiDAR measurements relevant for visibility estimation."""

        if not robot.sensor_data:
            return []

        return [
            (float(distance), float(angle))
            for distance, angle in robot.sensor_data
        ]

    def _robot_observed_distance(
        self, robot: RobotState, point: Position, *, default: float = 0.0
    ) -> float:
        """Return the distance (in meters) measured along the ray to *point*.

        The estimate is derived solely from the robot's own LiDAR samples,
        ensuring that formation planning only reasons about obstacles that the
        robot has actually perceived.  When no point cloud is available we
        fall back to *default*, which should represent the most conservative
        assumption (typically ``0``).
        """

        samples = self._robot_observation_samples(robot)
        if not samples:
            return default

        target_angle = self._robot_relative_angle(robot, point)
        best_distance: Optional[float] = None
        best_diff = math.inf
        for distance, sample_angle in samples:
            diff = abs(self._normalize_angle(target_angle - sample_angle))
            if diff < best_diff - 1e-9:
                best_diff = diff
                best_distance = distance
            elif abs(diff - best_diff) <= 1e-9 and best_distance is not None:
                best_distance = min(best_distance, distance)

        if best_distance is None:
            return default
        return float(best_distance)

    def _robot_clearance(self, robot: RobotState, point: Position) -> float:
        """Estimate free space beyond *point* using only *robot* observations."""

        query_distance = self._robot_query_distance(robot, point)
        observed_distance = self._robot_observed_distance(robot, point, default=0.0)
        if observed_distance <= query_distance:
            return 0.0
        return observed_distance - query_distance

    def _robot_line_blocked(self, robot: RobotState, point: Position) -> bool:
        """Return ``True`` if *robot* perceives an obstacle before *point*.

        Unknown space is treated as blocked so that optimisation remains
        conservative when LiDAR data is unavailable.
        """

        query_distance = self._robot_query_distance(robot, point)
        if query_distance <= 0.0:
            return False

        observed_distance = self._robot_observed_distance(robot, point, default=0.0)
        if observed_distance <= 0.0:
            return True

        tolerance = 1e-3
        return observed_distance + tolerance < query_distance

    # ------------------------------------------------------------------
    # Debug helpers that consult the ground-truth map.
    # ------------------------------------------------------------------

    def _is_point_obstacle_map(self, point: Position) -> bool:
        x = int(round(point[0]))
        y = int(round(point[1]))
        if not (0 <= x < self.env.SCREEN_W and 0 <= y < self.env.SCREEN_H):
            return True
        mask = self._get_obstacle_mask()
        return bool(mask[y, x])

    def _distance_to_nearest_obstacle_map(self, point: Position) -> float:
        x = int(round(point[0]))
        y = int(round(point[1]))
        if not (0 <= x < self.env.SCREEN_W and 0 <= y < self.env.SCREEN_H):
            return 0.0
        distance_map = self._get_obstacle_distance_map()
        return float(distance_map[y, x])

    def _is_line_blocked_map(self, start: Position, end: Position) -> bool:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.hypot(dx, dy)
        steps = max(int(distance), 1)
        for step in range(1, steps + 1):
            t = step / steps
            x = start[0] + dx * t
            y = start[1] + dy * t
            if self._is_point_obstacle_map((x, y)):
                return True
        return False

    def optimize_formation(
        self,
        formation_type: Optional[str] = None,
        *,
        weights: Optional[Sequence[float]] = None,
        ego_index: Optional[int] = None,
        spacing: Optional[float] = None,
        apply: bool = True,
    ) -> FormationPlan:
        return self.formation.optimize(
            formation_type,
            weights=weights,
            ego_index=ego_index,
            spacing=spacing,
            apply=apply,
        )

    def _cycle_map(self) -> None:
        map_types = self._available_map_types()
        if not map_types:
            return

        current_type = self.env.map_type
        if current_type not in map_types:
            next_index = 0
        else:
            next_index = (map_types.index(current_type) + 1) % len(map_types)

        next_map_type = map_types[next_index]
        if next_map_type != current_type:
            self.set_map(next_map_type)

    def _operate_moving(self) -> None:
        robot = self.robot_states[self.selected_robot_index]
        keys = pygame.key.get_pressed()
        moved = False
        translated = False

        robot.braking = False
        robot.executable_velocity = 0.0
        robot.executable_angular_velocity = 0.0
        robot.output_velocity = 0.0
        robot.output_angular_velocity = 0.0

        if keys[pygame.K_w]:
            move_success = self._attempt_move(robot, -self.manual_step_pixels)
            moved |= move_success
            translated |= move_success
        if keys[pygame.K_q]:
            robot.theta += self.manual_rotation_step
            moved = True
        if keys[pygame.K_e]:
            robot.theta -= self.manual_rotation_step
            moved = True

        if moved:
            robot.theta = (robot.theta + math.pi) % (2 * math.pi) - math.pi

        if moved and self.path_history != 0:
            robot.path.append((robot.position, robot.theta))
        if not translated:
            robot.velocity = 0.0

    def _attempt_move(self, robot: RobotState, step: float) -> bool:
        dx = step * math.sin(robot.theta)
        dy = step * math.cos(robot.theta)
        next_position = (robot.position[0] + dx, robot.position[1] + dy)

        if self._is_collision(next_position, ignore_robot=robot):
            return False

        self._update_robot_motion(robot, next_position)
        return True

    def _is_collision(self, position: Position, *, ignore_robot: Optional[RobotState] = None) -> bool:
        # 安全マージンではなく、物理サイズ(本体半径)で判定する
        body_radius_px = self.env.ROBOT_BODY_RADIUS_M * self.env.PIXELS_PER_METER

        # 1. 壁（障害物）との衝突判定
        for i in range(self.env.NUM_CHECK_POINTS):
            angle = 2 * math.pi * i / self.env.NUM_CHECK_POINTS
            # 修正: body_radius_px を使用
            check_x = int(position[0] + body_radius_px * math.cos(angle))
            check_y = int(position[1] + body_radius_px * math.sin(angle))
            
            if not (0 <= check_x < self.env.SCREEN_W and 0 <= check_y < self.env.SCREEN_H): return True
            
            pixel_color = self.env.obstacle_layer.get_at((check_x, check_y))
            if (pixel_color[0], pixel_color[1], pixel_color[2]) == self.obstacle_color_rgb: return True

        # 2. 他のロボットとの衝突判定
        for other in self.robot_states:
            if other is ignore_robot: continue
            distance = math.hypot(position[0] - other.position[0], position[1] - other.position[1])
            
            # 修正: 本体同士が接触する距離で判定
            min_clearance = body_radius_px * 2 
            if distance < min_clearance: return True
            
        return False


    def _update_lidar(self, robot: RobotState) -> None:
        if not robot.lidar_enabled or robot.sensor is None:
            robot.sensor_data = []
            robot.closest_obstacle = (float("inf"), 0.0)
            return

        robot.sensor.position = (
            int(robot.position[0]),
            int(robot.position[1]),
        )
        robot.sensor.theta = robot.theta
        
        self._lidar_detection_surface.fill((0, 0, 0, 0))
        self._lidar_detection_surface.blit(self.env.obstacle_layer, (0, 0))

        body_radius_px = int(self.env.ROBOT_BODY_RADIUS_M * self.env.PIXELS_PER_METER)

        for other in self.robot_states:
            if other is robot:
                continue

            pygame.draw.circle(
                self._lidar_detection_surface,
                self.obstacle_color_rgb,
                (int(other.position[0]), int(other.position[1])),
                body_radius_px,
            )

        raw_sensor_data = robot.sensor.sense_obstacles(
            self._lidar_detection_surface, self.env.OBSTACLE_COLOR
        )
        robot.sensor_data = list(raw_sensor_data) if raw_sensor_data else []
        robot.filtered_sensor_data = []
        if robot.sensor_data and self.filter_robot:
            self._filter_robot_reflections(robot)

        if robot.sensor_data:
            if self.show_lidar_visualization:
                filtered_counts: Dict[float, int] = {}
                if robot.filtered_sensor_data:
                    for _distance, relative_angle in robot.filtered_sensor_data:
                        filtered_counts[relative_angle] = (
                            filtered_counts.get(relative_angle, 0) + 1
                        )

                for distance, relative_angle in robot.sensor_data:
                    color = robot.color
                    if filtered_counts.get(relative_angle):
                        filtered_counts[relative_angle] -= 1
                        color = self.lidar_filtered_points_color

                    obstacle_angle = robot.theta - math.pi / 2 - relative_angle
                    point = utils_module.AD2pos(
                        distance,
                        obstacle_angle,
                        robot.position,
                        self.env.PIXELS_PER_METER,
                    )
                    pygame.draw.circle(
                        self.env.robot_and_lidar_layer,
                        color,
                        (int(point[0]), int(point[1])),
                        2,
                    )
            robot.closest_obstacle = robot.sensor.get_closest_obstacle_info(
                robot.sensor_data
            )
        else:
            robot.closest_obstacle = (float("inf"), 0.0)

    def _filter_robot_reflections(self, robot: RobotState) -> None:
        if robot.sensor is None:
            return

        robot_radius_m = self.env.ROBOT_RADIUS_M
        lidar_range = robot.sensor.LIDAR_RANGE_M
        pixels_per_meter = self.env.PIXELS_PER_METER
        other_robot_filters = []

        for other in self.robot_states:
            if other is robot:
                continue

            dx_pixels = other.position[0] - robot.position[0]
            dy_pixels = other.position[1] - robot.position[1]
            distance_pixels = math.hypot(dx_pixels, dy_pixels)
            if distance_pixels == 0:
                continue

            distance_m = distance_pixels / pixels_per_meter
            if distance_m > lidar_range + robot_radius_m:
                continue

            global_angle = math.atan2(-dx_pixels, -dy_pixels)
            if global_angle < 0:
                global_angle += 2 * math.pi

            if distance_m <= robot_radius_m:
                angle_half_width = math.pi
            else:
                ratio = min(1.0, robot_radius_m / distance_m)
                angle_half_width = math.asin(ratio)

            distance_min = max(0.0, distance_m - robot_radius_m)
            distance_max = distance_m + robot_radius_m
            other_robot_filters.append(
                (global_angle, angle_half_width, distance_min, distance_max)
            )

        if not other_robot_filters:
            return

        filtered_data: List[Tuple[float, float]] = []
        replaced_measurements: List[Tuple[float, float]] = []
        theta = robot.theta
        buffer = 1.1
        for distance, relative_angle in robot.sensor_data:
            global_angle = (theta - math.pi / 2 - relative_angle) % (2 * math.pi)
            replaced = False

            for angle_center, angle_half_width, distance_min, distance_max in other_robot_filters:
                angle_diff = abs(global_angle - angle_center)
                angle_diff = min(angle_diff, 2 * math.pi - angle_diff)

                if angle_diff <= angle_half_width * buffer and distance_min * (2 - buffer) <= distance <= distance_max * buffer:
                    replaced_measurements.append((distance, relative_angle))
                    filtered_data.append((robot.sensor.out_of_range_value, relative_angle))
                    replaced = True
                    break

            if not replaced:
                filtered_data.append((distance, relative_angle))

        robot.sensor_data = filtered_data
        robot.filtered_sensor_data = replaced_measurements

    def _select_next_robot(self) -> None:
        if not self.robot_states:
            return
        self.selected_robot_index = (self.selected_robot_index + 1) % len(
            self.robot_states
        )


def _run_event_loop(simulator: "MultiRobotSimulator", fps_limit: int) -> None:
    clock = pygame.time.Clock()
    print("[DEBUG] Entering main event loop.") # 追加: ループ入り口確認
    # frame_count = 0
    
    try:
        while simulator.running:
            # 1. イベント処理
            simulator.process_events()
            
            # 2. 描画処理
            simulator.draw_scene()
            
            # 3. 生存確認ログ (約1秒ごとに表示)
            # frame_count += 1
            # if frame_count % 30 == 0:
                # print(f"[DEBUG] Loop running... Frame {frame_count}")
            
            clock.tick(fps_limit)
            
    except Exception as e:
        # エラーで落ちた場合に詳細を表示する
        print(f"[ERROR] Exception occurred in event loop: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("[DEBUG] Closing simulator.")
        simulator.close()


def demo() -> None:
    simulator = MultiRobotSimulator(num_robots=2, render_mode="human", show_labels=False)
    simulator.set_robot_target(0, simulator.meters_to_pixels(1.0, 1.0))
    simulator.set_robot_target(1, simulator.meters_to_pixels(-1.0, -1.0))
    _run_event_loop(simulator, fps_limit=30)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Interactive multi-robot simulator with formation control."
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        help="Path to the simulator configuration YAML file.",
    )
    parser.add_argument(
        "--robots",
        type=int,
        dest="robots",
        help="Number of robots to spawn. Defaults to the configuration value.",
    )
    parser.add_argument(
        "--formation-type",
        dest="formation_type",
        help="Override the default formation label (e.g. line, triangle, square).",
    )
    parser.add_argument(
        "--formation-spacing",
        dest="formation_spacing",
        type=float,
        help="Override the default spacing between robots in metres.",
    )
    parser.add_argument(
        "--render-mode",
        dest="render_mode",
        choices=("human", "rgb_array"),
        default="human",
        help="Rendering mode used by pygame (default: human).",
    )
    parser.add_argument(
        "--map-type",
        dest="map_type",
        choices=("trap_shaped", "zigzag", "random", "empty", "image"),
        help="Start the simulator on the selected map type (default: configuration value).",
    )
    parser.add_argument(
        "--image-map-path",
        dest="image_map_path",
        help="Override the configured image path used for the image map type.",
    )
    parser.add_argument(
        "--fps",
        dest="fps_limit",
        type=int,
        default=30,
        help="Limit the draw loop to the specified frames per second (default: 30).",
    )
    parser.add_argument(
        "--trail-interval",
        dest="trail_step_interval",
        type=int,
        default=20,
        help="Trail dot interval in steps."
    )
    parser.add_argument(
        "--video-dir",
        dest="video_dir",
        default=".",
        help="Override the video dir"
    )
    args = parser.parse_args(argv)

    if args.robots is not None and args.robots <= 0:
        parser.error("--robots must be a positive integer")
    if args.formation_spacing is not None and args.formation_spacing <= 0:
        parser.error("--formation-spacing must be positive")
    if args.fps_limit <= 0:
        parser.error("--fps must be positive")

    simulator = MultiRobotSimulator(
        num_robots=args.robots,
        render_mode=args.render_mode,
        config_path=args.config_path,
        formation_type=args.formation_type,
        formation_spacing_m=args.formation_spacing,
        image_map_path=args.image_map_path,
        map_type=args.map_type,
        show_trail=True,
        trail_step_interval=args.trail_step_interval,
    )
    simulator.video_dir = args.video_dir
    # if args.map_type is not None:
    #     try:
    #         simulator.set_map(args.map_type)
    #     except (FileNotFoundError, ValueError) as exc:
    #         parser.error(str(exc))
    _run_event_loop(simulator, fps_limit=args.fps_limit)


if __name__ == "__main__":
    main()

