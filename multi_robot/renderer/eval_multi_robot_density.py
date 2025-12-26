"""
実験2: マルチロボット密度に対する頑健性の評価 (全シナリオ自動実行版)
- experiments/multi_robot_density/exp_YYYYMMDD_HHMMSS/videos/ に動画を保存
- experiments/multi_robot_density/exp_YYYYMMDD_HHMMSS/result_summary.csv に結果を保存

構成:
- 5つのシナリオを順次実行
- 各シナリオで 2台～10台 のロボットをテスト
- 各条件につき 10回 試行
"""
import math
import pygame
import pandas as pd
import numpy as np
import cv2
import os
import datetime
from tqdm import tqdm
import random

# simulator_v5.py と同じ階層に配置してください
from multi_robot.renderer.simulator_v5 import MultiRobotSimulator

# TeamNavigationControllerの代わりに何もしないダミークラス（フォーメーション制御無効化用）
class DummyTeamNav:
    def update(self):
        pass
    def draw(self, surface):
        pass
    def set_team_goal(self, goal):
        pass

class MultiRobotEvalSimulator(MultiRobotSimulator):
    def __init__(self, num_robots, **kwargs):
        # 指定された台数で初期化
        kwargs['num_robots'] = num_robots
        super().__init__(**kwargs)
        
        # 実験設定
        self.map_w_m = self.env.SCREEN_W / self.env.PIXELS_PER_METER
        self.map_h_m = self.env.SCREEN_H / self.env.PIXELS_PER_METER
        
        # 配置半径 (画面サイズに合わせて調整: 中心から少し離す)
        self.spawn_radius_m = min(self.map_w_m, self.map_h_m) * 0.35
        self.goal_tolerance_m = 0.5
        
        # グローバルプランニング(フォーメーション)無効化
        self.team_nav = DummyTeamNav()
        self.vw_planner = None 

        self.show_debug_text = False
        self.info_panel.visible = False
        self.show_lidar_visualization = False
        
        # 他のロボットをLiDARで認識させるための設定
        self.filter_robot = False 

        # --- 動画保存用の変数を初期化 ---
        self.video_writer = None
        self.is_recording = False

    # ----------------------------------------------------------------
    # Scenarios
    # ----------------------------------------------------------------

    def generate_circular_scenario(self):
        """ロボットを円状に配置し、対角をゴールにする"""
        self.env.obstacle_layer.fill((0, 0, 0, 0))
        self._invalidate_obstacle_cache()
        self.clear_paths()

        num = len(self.robot_states)
        angle_step = 2 * math.pi / num
        
        angle_offset = np.random.uniform(0, 2 * math.pi)

        for i in range(num):
            angle = i * angle_step + angle_offset
            
            # ズレの追加
            radius_jitter = random.uniform(-1.5, 1.5)
            effective_radius = self.spawn_radius_m + radius_jitter
            angle_jitter = random.uniform(-0.1, 0.1)
            effective_angle = angle + angle_jitter
            
            sx = effective_radius * math.cos(effective_angle)
            sy = effective_radius * math.sin(effective_angle)
            
            gx = -self.spawn_radius_m * math.cos(angle)
            gy = -self.spawn_radius_m * math.sin(angle)
            
            yaw = math.atan2(gy - sy, gx - sx) - math.pi / 2

            self.set_robot_pose(i, (sx, sy), yaw, in_meters=True)
            self.set_robot_target(i, (gx, gy), in_meters=True)
            
            self.robot_states[i].finished = False
            self.robot_states[i].collided = False

    def generate_random_scenario(self):
        """スタートもゴールもランダムな位置に配置する"""
        self.env.obstacle_layer.fill((0, 0, 0, 0))
        self._invalidate_obstacle_cache()
        self.clear_paths()

        num = len(self.robot_states)
        
        margin = 1.0 
        range_x = (self.map_w_m / 2.0) - margin
        range_y = (self.map_h_m / 2.0) - margin
        
        min_robot_separation = 1.5
        min_travel_dist = 6.0 # 少し緩和
        
        start_positions = []
        goal_positions = []
        
        for i in range(num):
            # 1. Start
            sx, sy = 0, 0
            for _ in range(100):
                sx = random.uniform(-range_x, range_x)
                sy = random.uniform(-range_y, range_y)
                conflict = False
                for (ex, ey) in start_positions:
                    if math.hypot(sx - ex, sy - ey) < min_robot_separation:
                        conflict = True; break
                if not conflict:
                    start_positions.append((sx, sy))
                    break
            
            # 2. Goal
            gx, gy = 0, 0
            for _ in range(100):
                gx = random.uniform(-range_x, range_x)
                gy = random.uniform(-range_y, range_y)
                
                if math.hypot(gx - sx, gy - sy) < min_travel_dist: continue

                conflict = False
                for (ogx, ogy) in goal_positions:
                    if math.hypot(gx - ogx, gy - ogy) < min_robot_separation:
                        conflict = True; break
                if not conflict:
                    goal_positions.append((gx, gy))
                    break
            
            yaw = math.atan2(gy - sy, gx - sx) - math.pi / 2
            self.set_robot_pose(i, (sx, sy), yaw, in_meters=True)
            self.set_robot_target(i, (gx, gy), in_meters=True)
            self.robot_states[i].finished = False
            self.robot_states[i].collided = False

    def generate_narrow_passage_scenario(self):
        """中央に狭い通路があるシナリオ"""
        self.env.obstacle_layer.fill((0, 0, 0, 0))
        self._invalidate_obstacle_cache()
        self.clear_paths()

        num = len(self.robot_states)
        
        wall_y_m = 0.0
        wall_thickness_m = 2.0
        gap_width_m = 2.5
        
        cx, cy = self.meters_to_pixels(0, wall_y_m)
        w_px = self.env.SCREEN_W
        h_px = int(wall_thickness_m * self.env.PIXELS_PER_METER)
        gap_px = int(gap_width_m * self.env.PIXELS_PER_METER)
        
        rect_left = pygame.Rect(0, cy - h_px//2, (w_px - gap_px)//2, h_px)
        pygame.draw.rect(self.env.obstacle_layer, self.obstacle_color_rgb, rect_left)
        
        left_start = (w_px + gap_px) // 2
        rect_right = pygame.Rect(left_start, cy - h_px//2, w_px - left_start + 100, h_px)
        pygame.draw.rect(self.env.obstacle_layer, self.obstacle_color_rgb, rect_right)
        
        margin_y = 2.0
        start_y = -(self.map_h_m / 2.0) + margin_y
        goal_y = (self.map_h_m / 2.0) - margin_y
        
        available_w = self.map_w_m * 0.8
        step_x = available_w / (num + 1)
        start_x_base = -available_w / 2.0
        
        for i in range(num):
            jitter = random.uniform(-0.2, 0.2)
            offset_x = (i + 1) * step_x + start_x_base + jitter
            
            sx = offset_x
            sy = start_y + random.uniform(-0.5, 0.5)
            
            gx = offset_x
            gy = goal_y
            
            yaw = math.atan2(gy - sy, gx - sx) - math.pi / 2
            
            self.set_robot_pose(i, (sx, sy), yaw, in_meters=True)
            self.set_robot_target(i, (gx, gy), in_meters=True)
            self.robot_states[i].finished = False
            self.robot_states[i].collided = False

    def generate_random_with_obstacles_scenario(self, num_obstacles=8):
        """障害物がある環境でスタート・ゴール完全ランダム配置"""
        self.env.obstacle_layer.fill((0, 0, 0, 0))
        self._invalidate_obstacle_cache()
        self.clear_paths()

        num_robots = len(self.robot_states)
        
        obstacle_list = []
        margin_m = 1.0
        range_x = (self.map_w_m / 2.0) - margin_m
        range_y = (self.map_h_m / 2.0) - margin_m
        
        min_obs_radius = 0.5
        max_obs_radius = 1.5
        
        for _ in range(num_obstacles):
            ox = random.uniform(-range_x, range_x)
            oy = random.uniform(-range_y, range_y)
            r_m = random.uniform(min_obs_radius, max_obs_radius)
            obstacle_list.append((ox, oy, r_m))
            
            px, py = self.meters_to_pixels(ox, oy)
            r_px = int(r_m * self.env.PIXELS_PER_METER)
            pygame.draw.circle(self.env.obstacle_layer, self.obstacle_color_rgb, (int(px), int(py)), r_px)
            
        self._invalidate_obstacle_cache()

        min_robot_separation = 2.0
        min_obs_dist = 1.0
        min_travel_dist = 6.0
        robot_radius = self.env.ROBOT_RADIUS_M

        start_positions = []
        goal_positions = []

        for i in range(num_robots):
            sx, sy = 0, 0
            for _ in range(200):
                sx = random.uniform(-range_x, range_x)
                sy = random.uniform(-range_y, range_y)
                valid = True
                for (ex, ey) in start_positions:
                    if math.hypot(sx - ex, sy - ey) < min_robot_separation:
                        valid = False; break
                if not valid: continue
                for (ox, oy, r_m) in obstacle_list:
                    if math.hypot(sx - ox, sy - oy) < (r_m + robot_radius + min_obs_dist):
                        valid = False; break
                if valid:
                    start_positions.append((sx, sy)); break
            
            gx, gy = 0, 0
            for _ in range(200):
                gx = random.uniform(-range_x, range_x)
                gy = random.uniform(-range_y, range_y)
                valid = True
                if math.hypot(gx - sx, gy - sy) < min_travel_dist: valid = False
                if not valid: continue
                for (ogx, ogy) in goal_positions:
                    if math.hypot(gx - ogx, gy - ogy) < min_robot_separation:
                        valid = False; break
                if not valid: continue
                for (ox, oy, r_m) in obstacle_list:
                    if math.hypot(gx - ox, gy - oy) < (r_m + robot_radius + min_obs_dist):
                        valid = False; break
                if valid:
                    goal_positions.append((gx, gy)); break

            yaw = math.atan2(gy - sy, gx - sx) - math.pi / 2
            self.set_robot_pose(i, (sx, sy), yaw, in_meters=True)
            self.set_robot_target(i, (gx, gy), in_meters=True)
            self.robot_states[i].finished = False
            self.robot_states[i].collided = False

    def generate_crossing_scenario(self):
        """上下対向移動シナリオ（整列あり・密度調整）"""
        self.env.obstacle_layer.fill((0, 0, 0, 0))
        self._invalidate_obstacle_cache()
        self.clear_paths()

        num = len(self.robot_states)
        
        margin_x = 1.0
        margin_y = 1.5
        max_physical_range_x = (self.map_w_m / 2.0) - margin_x
        
        width_per_robot = 1.5 
        calculated_range = (num * width_per_robot) / 2.0
        range_x = min(max_physical_range_x, max(1.5, calculated_range))
        
        top_y_base = (self.map_h_m / 3.0) - margin_y
        bottom_y_base = -(self.map_h_m / 3.0) + margin_y
        min_spacing = 1.2

        def generate_sorted_points(count, base_y, x_limit, spacing):
            points = []
            for _ in range(count):
                for _ in range(100):
                    rx = random.uniform(-x_limit, x_limit)
                    ry = base_y + random.uniform(-0.5, 0.5)
                    valid = True
                    for (ex, ey) in points:
                        if math.hypot(rx - ex, ry - ey) < spacing:
                            valid = False; break
                    if valid:
                        points.append((rx, ry)); break
            points.sort(key=lambda p: p[0])
            return points

        down_robots = [i for i in range(num) if i % 2 == 0]
        up_robots = [i for i in range(num) if i % 2 != 0]

        if down_robots:
            starts_down = generate_sorted_points(len(down_robots), top_y_base, range_x, min_spacing)
            goals_down = generate_sorted_points(len(down_robots), bottom_y_base, range_x, min_spacing)
            for idx, r_idx in enumerate(down_robots):
                s = starts_down[idx]
                g = goals_down[idx]
                yaw = math.atan2(g[1] - s[1], g[0] - s[0]) - math.pi / 2
                self.set_robot_pose(r_idx, s, yaw, in_meters=True)
                self.set_robot_target(r_idx, g, in_meters=True)

        if up_robots:
            starts_up = generate_sorted_points(len(up_robots), bottom_y_base, range_x, min_spacing)
            goals_up = generate_sorted_points(len(up_robots), top_y_base, range_x, min_spacing)
            for idx, r_idx in enumerate(up_robots):
                s = starts_up[idx]
                g = goals_up[idx]
                yaw = math.atan2(g[1] - s[1], g[0] - s[0]) - math.pi / 2
                self.set_robot_pose(r_idx, s, yaw, in_meters=True)
                self.set_robot_target(r_idx, g, in_meters=True)

        for i in range(num):
            self.robot_states[i].finished = False
            self.robot_states[i].collided = False

    def generate_line_to_line_scenario(self):
        """下部ラインから上部ラインへの移動シナリオ（予備）"""
        self.env.obstacle_layer.fill((0, 0, 0, 0))
        self._invalidate_obstacle_cache()
        self.clear_paths()
        num = len(self.robot_states)
        margin_x, margin_y = 1.0, 1.5
        range_x = (self.map_w_m / 2.0) - margin_x
        start_y = -(self.map_h_m / 2.0) + margin_y
        goal_y = (self.map_h_m / 2.0) - margin_y
        min_spacing = 1.2
        start_positions = []
        goal_positions = []
        for i in range(num):
            sx = 0
            for _ in range(100):
                sx = random.uniform(-range_x, range_x)
                sy = start_y + random.uniform(-0.5, 0.5)
                conflict = False
                for (ex, ey) in start_positions:
                    if math.hypot(sx - ex, sy - ey) < min_spacing:
                        conflict = True; break
                if not conflict: start_positions.append((sx, sy)); break
            gx = 0
            for _ in range(100):
                gx = random.uniform(-range_x, range_x)
                gy = goal_y + random.uniform(-0.5, 0.5)
                conflict = False
                for (ogx, ogy) in goal_positions:
                    if math.hypot(gx - ogx, gy - ogy) < min_spacing:
                        conflict = True; break
                if not conflict: goal_positions.append((gx, gy)); break
            yaw = math.atan2(goal_positions[-1][1] - start_positions[-1][1], goal_positions[-1][0] - start_positions[-1][0]) - math.pi / 2
            self.set_robot_pose(i, start_positions[-1], yaw, in_meters=True)
            self.set_robot_target(i, goal_positions[-1], in_meters=True)
            self.robot_states[i].finished = False
            self.robot_states[i].collided = False

    # ----------------------------------------------------------------
    # Simulation Logic
    # ----------------------------------------------------------------
    def step_simulation(self):
        if self.policy_controller:
            self.policy_controller._last_action_timestamp = None 
            self.policy_controller.step()
        self.process_events()
        self.draw_scene()

    def start_recording(self, filepath, fps=30):
        if self.video_writer is not None: self.stop_recording()
        width = self.env.SCREEN_W
        height = self.env.SCREEN_H
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
        self.is_recording = True

    def capture_frame(self):
        if not self.is_recording or self.video_writer is None: return
        surface = self.display_surface 
        frame_rgb = pygame.surfarray.array3d(surface)
        frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        self.video_writer.write(frame_bgr)

    def stop_recording(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self.is_recording = False

def run_experiment():
    # 1. フォルダ構成の作成
    now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = f"experiments/multi_robot_density/exp_{now_str}"
    video_dir = os.path.join(experiment_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)
    
    print(f"Experiment Directory: {experiment_dir}")
    print(f"Video Directory:      {video_dir}")

    # 実験パラメータ
    robot_counts = list(range(2, 11)) # 2台～10台
    robot_counts = [2, 4, 6, 8, 10]
    num_trials = 50                  # 各10試行
    max_steps = 600                  # タイムアウト
    
    # 実行するシナリオのリスト (表示名, メソッド名)
    scenarios = [
        ("circular", "generate_circular_scenario"),
        # ("random", "generate_random_scenario"),
        # ("narrow", "generate_narrow_passage_scenario"),
        # ("obstacles", "generate_random_with_obstacles_scenario"),
        ("crossing", "generate_crossing_scenario")
    ]
    
    results = []

    print("=== Experiment 2: Multi-Robot Density All Scenarios ===")

    # シナリオごとにループ
    for scenario_name, method_name in scenarios:
        print(f"\n--- Scenario: {scenario_name} ---")
        
        # ロボット台数ごとにループ
        for n_robots in robot_counts:
            print(f"Testing with {n_robots} robots...")
            
            # シミュレータの初期化 (台数変更のため再作成)
            sim = MultiRobotEvalSimulator(
                num_robots=n_robots,
                render_mode="human",
                show_info_panel=False,
                manual_control=False,
                policy_model_path="multi_robot/models/model_ep16500.zip",
                map_type="empty"
            )

            if sim.policy_controller:
                sim.policy_controller.active = True
                sim.policy_control_active = True
            else:
                print("Error: Policy model not loaded!")
                sim.close()
                return

            success_count = 0
            collision_count = 0
            timeout_count = 0
            total_steps_all = 0
            total_steps_success = 0

            # 試行ループ
            for i in tqdm(range(num_trials), desc=f"{scenario_name} | {n_robots} bots"):
                # シナリオ生成メソッドを呼び出し
                getattr(sim, method_name)()
                
                # 動画ファイル名 (シナリオ名を含める)
                video_filename = os.path.join(
                    video_dir, 
                    f"{scenario_name}_robots_{n_robots:02d}_trial_{i:03d}.mp4"
                )
                sim.start_recording(video_filename, fps=30)
                
                step = 0
                episode_done = False
                result_status = "timeout"

                while step < max_steps:
                    sim.step_simulation()
                    sim.capture_frame()

                    all_reached = True
                    any_collision = False

                    for robot in sim.robot_states:
                        if robot.finished: continue

                        # 衝突判定
                        is_collided = getattr(robot, 'is_collided', False)
                        if is_collided or sim._is_collision(robot.position, ignore_robot=robot):
                            robot.collided = True
                            any_collision = True
                            robot.finished = True 
                            break 
                        
                        # ゴール判定
                        if robot.target:
                            dist = math.hypot(robot.position[0]-robot.target[0], robot.position[1]-robot.target[1]) / sim.env.PIXELS_PER_METER
                            if dist < sim.goal_tolerance_m: robot.finished = True
                            else: all_reached = False
                        else: all_reached = False

                    if any_collision:
                        collision_count += 1
                        episode_done = True
                        result_status = "collision"
                    elif all_reached:
                        success_count += 1
                        episode_done = True
                        result_status = "success"
                    
                    if episode_done: break
                    step += 1
                
                sim.stop_recording()
                if not episode_done:
                    timeout_count += 1
                    result_status = "timeout"

                # ステップ数記録
                total_steps_all += step
                if result_status == "success":
                    total_steps_success += step
                
                # 結果に応じてファイル名リネーム
                final_filename = video_filename.replace(".mp4", f"_{result_status}.mp4")
                try:
                    if os.path.exists(video_filename): os.rename(video_filename, final_filename)
                except OSError: pass

            # 台数ごとの集計
            res = {
                "scenario": scenario_name,
                "num_robots": n_robots,
                "success_rate": success_count / num_trials,
                "collision_rate": collision_count / num_trials,
                "timeout_rate": timeout_count / num_trials,
                "avg_steps": total_steps_all / num_trials,  # 全試行平均
                "avg_success_steps": (total_steps_success / success_count) if success_count > 0 else None # 成功時平均
            }
            results.append(res)
            # print(f"Result: {res}")
            
            sim.close()

    # CSV保存
    csv_filename = os.path.join(experiment_dir, "result_summary.csv")
    df = pd.DataFrame(results)
    df.to_csv(csv_filename, index=False)
    
    print(f"\nAll Experiments Completed.")
    print(f"Summary CSV: {csv_filename}")
    print(f"Videos:      {video_dir}")

if __name__ == "__main__":
    run_experiment()