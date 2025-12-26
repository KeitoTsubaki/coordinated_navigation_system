"""
実験1: 障害物密度に対する頑健性の評価 (0~15個まで変動)
- experiments/local_planner_obstacle_density/exp_YYYYMMDD_HHMMSS/videos/ に動画を保存
- experiments/local_planner_obstacle_density/exp_YYYYMMDD_HHMMSS/result_summary.csv に結果を保存
"""
import math
import random
import pygame
import pandas as pd
import numpy as np
import cv2
import os
import datetime
from tqdm import tqdm

# simulator_v5.py と同じ階層、またはパスが通った場所に配置してください
from multi_robot.renderer.simulator_v5 import MultiRobotSimulator

# TeamNavigationControllerの代わりに何もしないダミークラス
class DummyTeamNav:
    def update(self):
        pass
    def draw(self, surface):
        pass
    def set_team_goal(self, goal):
        pass

class ObstacleEvalSimulator(MultiRobotSimulator):
    def __init__(self, **kwargs):
        # 強制的に1台設定
        kwargs['num_robots'] = 1
        super().__init__(**kwargs)
        
        # 実験設定
        self.map_w_m = self.env.SCREEN_W / self.env.PIXELS_PER_METER
        self.map_h_m = self.env.SCREEN_H / self.env.PIXELS_PER_METER
        
        self.obstacle_radius_m = 1.0
        self.robot_radius_m = self.env.ROBOT_RADIUS_M
        self.safe_margin_m = 1.0
        self.min_obstacle_spacing = (self.obstacle_radius_m * 2) + (self.robot_radius_m * 2) + 0.5
        # print(f"min obstacle spacing: {self.min_obstacle_spacing}")
        self.start_goal_min_dist = 15.0
        self.start_goal_safe_radius = 2.0
        
        # グローバルプランニング無効化
        self.team_nav = DummyTeamNav()
        self.vw_planner = None 

        self.show_debug_text = False
        self.info_panel.visible = False
        self.show_lidar_visualization = False

        # --- 動画保存用の変数を初期化 ---
        self.video_writer = None
        self.is_recording = False

    def generate_eval_map(self, num_obstacles: int):
        """評価用のマップとスタート・ゴールを生成する"""
        self.env.obstacle_layer.fill((0, 0, 0, 0))
        obstacle_positions = []

        range_x = (self.map_w_m / 2.0) - self.safe_margin_m
        range_y = (self.map_h_m / 2.0) - self.safe_margin_m

        # スタートとゴールの決定
        while True:
            sx = random.uniform(-range_x, range_x)
            sy = random.uniform(-range_y, range_y)
            gx = random.uniform(-range_x, range_x)
            gy = random.uniform(-range_y, range_y)
            
            dist = math.hypot(gx - sx, gy - sy)
            if dist >= self.start_goal_min_dist:
                diff_x = gx - sx
                diff_y = gy - sy
                yaw = math.atan2(diff_y, diff_x) - math.pi / 2

                self.start_pos_m = (sx, sy)
                self.start_yaw = yaw
                self.goal_pos_m = (gx, gy)
                break

        # 障害物の配置
        attempts = 0
        # while len(obstacle_positions) < num_obstacles and attempts < 100000:
        while len(obstacle_positions) < num_obstacles:
            if attempts > 100000:
                # 配置に失敗しすぎた場合は諦めてクリアして再試行（またはその数で妥協）
                # ここでは単純にリストを空にして最初からやり直すロジックになっています
                obstacle_positions = []
                attempts = 0 

            attempts += 1
            ox = random.uniform(-range_x + self.obstacle_radius_m, range_x - self.obstacle_radius_m)
            oy = random.uniform(-range_y + self.obstacle_radius_m, range_y - self.obstacle_radius_m)
            
            valid = True
            if math.hypot(ox - self.start_pos_m[0], oy - self.start_pos_m[1]) < self.start_goal_safe_radius:
                valid = False
            if valid and math.hypot(ox - self.goal_pos_m[0], oy - self.goal_pos_m[1]) < self.start_goal_safe_radius:
                valid = False
            
            if valid:
                for (ex, ey) in obstacle_positions:
                    if math.hypot(ox - ex, oy - ey) < self.min_obstacle_spacing:
                        valid = False
                        break
            
            if valid:
                obstacle_positions.append((ox, oy))
                px, py = self.meters_to_pixels(ox, oy)
                radius_px = int(self.obstacle_radius_m * self.env.PIXELS_PER_METER)
                pygame.draw.circle(
                    self.env.obstacle_layer,
                    self.obstacle_color_rgb,
                    (int(px), int(py)),
                    radius_px
                )
        
        self._invalidate_obstacle_cache()

        if len(self.robot_states) > 0:
            self.set_robot_pose(0, self.start_pos_m, self.start_yaw, in_meters=True)
            self.set_robot_target(0, self.goal_pos_m, in_meters=True)

    def step_simulation(self):
        if self.policy_controller:
            self.policy_controller._last_action_timestamp = None 
            self.policy_controller.step()
        
        self.process_events()
        self.draw_scene()

    # --- 動画保存用メソッド ---
    def start_recording(self, filepath, fps=30):
        if self.video_writer is not None:
            self.stop_recording()
            
        width = self.env.SCREEN_W
        height = self.env.SCREEN_H
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
        self.is_recording = True

    def capture_frame(self):
        if not self.is_recording or self.video_writer is None:
            return

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
    # -----------------------

def run_experiment():
    # 1. フォルダ構成の作成
    # experiments/exp_YYYYMMDD_HHMMSS/videos/ の形を作る
    now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = f"experiments/local_planner_obstacle_density/exp_{now_str}"
    video_dir = os.path.join(experiment_dir, "videos")
    
    # 再帰的にディレクトリを作成 (experiments/ -> exp_.../ -> videos/)
    os.makedirs(video_dir, exist_ok=True)
    
    print(f"Experiment Directory: {experiment_dir}")
    print(f"Video Directory:      {video_dir}")

    # ==========================================
    # 変更点: 0個から15個まで変動させる
    # ==========================================
    # obstacle_counts = list(range(0, 16)) # [0, 1, 2, ..., 15]
    obstacle_counts = [0, 5, 10, 15]
    num_trials = 100
    max_steps = 600
    
    results = []

    sim = ObstacleEvalSimulator(
        render_mode="human", 
        show_info_panel=False,
        manual_control=False,
        policy_model_path="multi_robot/models/model_ep16500.zip"
    )
    
    if sim.policy_controller:
        sim.policy_controller.active = True
        sim.policy_control_active = True
    else:
        print("Error: Policy model not loaded!")
        return

    print("=== Experiment 1: Obstacle Robustness ===")

    for obs_count in obstacle_counts:
        success_count = 0
        collision_count = 0
        timeout_count = 0
        total_steps_all = 0
        total_steps_success = 0
        
        print(f"Testing with {obs_count} obstacles...")
        
        for i in tqdm(range(num_trials)):
            sim.generate_eval_map(obs_count)
            sim.clear_paths()
            
            # 一時ファイル名 (videosフォルダ内に保存)
            temp_filename = os.path.join(video_dir, f"temp_obs_{obs_count:02d}_trial_{i:03d}.mp4")
            sim.start_recording(temp_filename, fps=30)
            
            step = 0
            done = False
            result_status = "timeout"

            while step < max_steps:
                sim.step_simulation()
                sim.capture_frame()

                robot = sim.robot_states[0]
                
                # ゴール判定
                goal_px = sim.meters_to_pixels(*sim.goal_pos_m)
                dist_to_goal = math.hypot(
                    robot.position[0] - goal_px[0],
                    robot.position[1] - goal_px[1]
                ) / sim.env.PIXELS_PER_METER
                
                if dist_to_goal < 0.5:
                    success_count += 1
                    done = True
                    result_status = "success"
                    break
                
                # 衝突判定
                is_collided = getattr(robot, 'is_collided', False)
                if is_collided or sim._is_collision(robot.position, ignore_robot=robot):
                    collision_count += 1
                    done = True
                    result_status = "collision"
                    break
                
                step += 1
            
            sim.stop_recording()

            if not done:
                timeout_count += 1
                result_status = "timeout"

            # ステップ数記録
            total_steps_all += step
            if result_status == "success":
                total_steps_success += step

            # ファイル名のリネーム (結果をファイル名に含める)
            final_filename = os.path.join(
                video_dir, 
                f"obs_{obs_count:02d}_trial_{i:03d}_{result_status}.mp4"
            )
            try:
                if os.path.exists(temp_filename):
                    os.rename(temp_filename, final_filename)
            except OSError as e:
                print(f"[Warning] Failed to rename video file: {e}")
            
            print(f"steps: {step}")

        
        res = {
            "num_obstacles": obs_count,
            "success_rate": success_count / num_trials,
            "collision_rate": collision_count / num_trials,
            "timeout_rate": timeout_count / num_trials,
            "avg_steps": total_steps_all / num_trials,  # 全試行平均
            "avg_success_steps": (total_steps_success / success_count) if success_count > 0 else None # 成功時平均
        }
        results.append(res)
        print(f"Result: {res}")

    sim.close()
    
    # 2. CSVを実験フォルダ直下 (videosの親フォルダ) に保存
    csv_filename = os.path.join(experiment_dir, "result_summary.csv")
    df = pd.DataFrame(results)
    df.to_csv(csv_filename, index=False)
    
    print(f"Experiment completed.")
    print(f"Summary CSV: {csv_filename}")
    print(f"Videos:      {video_dir}")

if __name__ == "__main__":
    run_experiment()