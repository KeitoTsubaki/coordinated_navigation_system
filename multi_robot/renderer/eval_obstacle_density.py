"""
実験1: 障害物密度に対する頑健性の評価 (修正版)
- チーム制御なし
- グローバルプランニングなし
- ロボット単体 (1台)
"""
import math
import random
import pygame
import pandas as pd
from tqdm import tqdm

# simulator_v5.py と同じ階層、またはパスが通った場所に配置してください
from multi_robot.renderer.simulator_v5 import MultiRobotSimulator

# TeamNavigationControllerの代わりに何もしないダミークラスを定義
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
        # 画面サイズからマップ範囲を動的に取得
        self.map_w_m = self.env.SCREEN_W / self.env.PIXELS_PER_METER
        self.map_h_m = self.env.SCREEN_H / self.env.PIXELS_PER_METER
        
        self.obstacle_radius_m = 0.3
        self.robot_radius_m = self.env.ROBOT_RADIUS_M
        self.safe_margin_m = 1.0
        # 障害物配置の制約
        self.min_obstacle_spacing = (self.obstacle_radius_m * 2) + (self.robot_radius_m * 2) + 1.0
        self.start_goal_min_dist = 10.0
        self.start_goal_safe_radius = 2.0
        
        # 【重要】グローバルプランニング(TeamNavigationController)を無効化
        # simulator_v5が作成したteam_navをダミーに差し替える
        self.team_nav = DummyTeamNav()
        self.vw_planner = None # プランナーも削除

        self.show_debug_text = False
        self.info_panel.visible = False
        self.show_lidar = False

    def generate_eval_map(self, num_obstacles: int):
        """評価用のマップとスタート・ゴールを生成する"""
        
        # 1. 障害物レイヤーのクリア
        self.env.obstacle_layer.fill((0, 0, 0, 0))
        obstacle_positions = []

        # 壁の内側に安全マージンを取った範囲で生成
        range_x = (self.map_w_m / 2.0) - self.safe_margin_m
        range_y = (self.map_h_m / 2.0) - self.safe_margin_m

        # 2. スタートとゴールの決定
        while True:
            sx = random.uniform(-range_x, range_x)
            sy = random.uniform(-range_y, range_y)
            gx = random.uniform(-range_x, range_x)
            gy = random.uniform(-range_y, range_y)
            
            dist = math.hypot(gx - sx, gy - sy)
            if dist >= self.start_goal_min_dist:
                self.start_pos_m = (sx, sy)
                self.goal_pos_m = (gx, gy)
                break

        # 3. 障害物の配置
        attempts = 0
        while len(obstacle_positions) < num_obstacles and attempts < 10000:
            attempts += 1
            ox = random.uniform(-range_x + self.obstacle_radius_m, range_x - self.obstacle_radius_m)
            oy = random.uniform(-range_y + self.obstacle_radius_m, range_y - self.obstacle_radius_m)
            
            # 制約チェック
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
                # 描画
                px, py = self.meters_to_pixels(ox, oy)
                radius_px = int(self.obstacle_radius_m * self.env.PIXELS_PER_METER)
                pygame.draw.circle(
                    self.env.obstacle_layer,
                    self.obstacle_color_rgb,
                    (int(px), int(py)),
                    radius_px
                )
        
        self._invalidate_obstacle_cache()

        # 4. ロボットの配置 (1台のみ)
        if len(self.robot_states) > 0:
            self.set_robot_pose(0, self.start_pos_m, 0.0, in_meters=True)
            # グローバルプランナーを使わず、直接ゴールをターゲットに設定
            self.set_robot_target(0, self.goal_pos_m, in_meters=True)

    def step_simulation(self):
        # 【重要】高速実行のためにタイムスタンプ制限を解除
        # これにより、実時間に関係なく毎回ステップが実行されます
        if self.policy_controller:
            self.policy_controller._last_action_timestamp = None 
            self.policy_controller.step()
        
        if self.env.render_mode == "human":
            self.process_events()
            self.draw_scene()
        else:
            if self.env.screen: self.env.screen.fill(self.env.BACKGROUND_COLOR)
            for robot in self.robot_states:
                self._update_lidar(robot)

def run_experiment():
    # パラメータ設定
    obstacle_counts = [0, 5, 10, 15, 20, 25, 30]
    num_trials = 50
    max_steps = 1000
    
    results = []

    # GUIで確認したい場合は render_mode="human" に設定
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

    print("=== Experiment 1: Obstacle Robustness (Single Robot, No Global Plan) ===")

    for obs_count in obstacle_counts:
        success_count = 0
        collision_count = 0
        timeout_count = 0
        
        print(f"Testing with {obs_count} obstacles...")
        
        for _ in tqdm(range(num_trials)):
            sim.generate_eval_map(obs_count)
            sim.clear_paths()
            
            step = 0
            done = False
            
            while step < max_steps:
                sim.step_simulation()
                robot = sim.robot_states[0]
                robot.lidar_enabled = sim.show_lidar
                
                # ゴール到達判定
                goal_px = sim.meters_to_pixels(*sim.goal_pos_m)
                dist_to_goal = math.hypot(
                    robot.position[0] - goal_px[0],
                    robot.position[1] - goal_px[1]
                ) / sim.env.PIXELS_PER_METER
                
                if dist_to_goal < 0.5:
                    success_count += 1
                    done = True
                    break
                
                # 【修正】衝突判定
                # PolicyControllerがセットした is_collided フラグ、または現在位置での物理的な接触を確認
                is_collided = getattr(robot, 'is_collided', False)
                if is_collided or sim._is_collision(robot.position, ignore_robot=robot):
                    collision_count += 1
                    done = True
                    break
                
                step += 1
            
            if not done:
                timeout_count += 1
        
        res = {
            "num_obstacles": obs_count,
            "success_rate": success_count / num_trials,
            "collision_rate": collision_count / num_trials,
            "timeout_rate": timeout_count / num_trials
        }
        results.append(res)
        print(f"Result: {res}")

    sim.close()
    
    df = pd.DataFrame(results)
    df.to_csv("exp1_obstacle_robustness.csv", index=False)
    print("Saved to exp1_obstacle_robustness.csv")

if __name__ == "__main__":
    run_experiment()