import numpy as np
import pygame
import time
import matplotlib.pyplot as plt
from pathlib import Path
from multi_robot.navigation.simulator_v5_multi import MultiRobotSimulator

MAPS_DIR = Path(__file__).resolve().parents[1] / "maps"

# ==========================================
# 実験設定
# ==========================================
CONFIG = {
    "num_robots": 2,                # ロボット台数
    "num_trials": 1,                # 試行回数を1回に固定
    "max_steps": 2000,              # 1エピソードの最大ステップ数
    "map_path": str(MAPS_DIR / "straight.jpg"),  # 画像マップのパス (multi_robot/maps/)
    "formation_spacing": 0.5,       # ロボット間隔 (m)
    "render": True,                 # 描画ON
    "fps_limit": 60,                # 1回だけなので見やすい速度(60fps)に設定
    
    # === スタート・ゴール位置の設定 (メートル単位) ===
    # Start: (x, y, theta_rad) 
    # ※thetaはラジアン (0: 右, pi/2: 下, pi: 左, -pi/2: 上 ※pygame座標系依存)
    "start_pos": (-7.5, 0.0, -1.57),   
    
    # Goal: (x, y)
    "goal_pos": (8.0, 0.0),       

    # === 比較検証用設定 ===
    # 1回の試行に対して「もしこのepsilonだったらFMRはどうだったか」を計算します
    "test_epsilons": [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
    "base_epsilon": 0.1,
}

class ExperimentData:
    def __init__(self):
        self.trials = [] 

    def add_trial(self, duration, error_series, success):
        self.trials.append({
            "duration": duration,
            "error_series": np.array(error_series),
            "success": success
        })

    def analyze_and_plot(self):
        """1回の試行データに対して、複数のepsilonで感度分析を行う"""
        print("\n" + "="*60)
        print(" SINGLE TRIAL SENSITIVITY ANALYSIS")
        print("="*60)
        print(f"{'Epsilon [m]':<12} | {'FMR':<10} | {'Total FRT [s]':<12} | {'Recovery Counts'}")
        print("-" * 60)

        epsilons = CONFIG["test_epsilons"]
        fmr_values = []
        frt_values = [] # ここでは「1回の試行での合計復帰時間」などを出すのが適切か、あるいは平均か

        dt = 1/30.0 # 1ステップあたりの想定時間 (sec)

        # 今回は1試行しかないので、self.trials[0] を直接解析
        if not self.trials:
            print("No data available.")
            return

        trial_data = self.trials[0]
        errors = trial_data["error_series"]
        success = trial_data["success"]

        if not success:
            print("[WARNING] The trial failed (Collision or Timeout). Data may be incomplete.")

        for eps in epsilons:
            # --- FMR (維持率) 計算 ---
            valid_mask = ~np.isnan(errors)
            valid_errors = errors[valid_mask]
            
            if len(valid_errors) > 0:
                maintain_count = np.sum(valid_errors < eps)
                fmr = maintain_count / len(valid_errors)
            else:
                fmr = 0.0
            
            # --- FRT (復帰時間) 計算 ---
            is_broken = False
            break_start_idx = 0
            frt_events = []
            
            for i, err in enumerate(errors):
                if np.isnan(err): continue
                
                if not is_broken and err > eps:
                    is_broken = True
                    break_start_idx = i
                elif is_broken and err <= eps:
                    is_broken = False
                    duration = (i - break_start_idx) * dt
                    frt_events.append(duration)

            avg_frt = np.mean(frt_events) if frt_events else 0.0
            
            fmr_values.append(fmr)
            frt_values.append(avg_frt)

            print(f"{eps:<12.2f} | {fmr:<10.3f} | {avg_frt:<12.3f} | {len(frt_events)}")

        self._plot_graphs(epsilons, fmr_values, frt_values)

        # === CSV保存機能 ===
        import csv
        csv_filename = "experiment_result.csv"
        with open(csv_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epsilon", "FMR", "Avg_FRT", "Recovery_Counts"])
            for eps, fmr, frt in zip(epsilons, fmr_values, frt_values):
                writer.writerow([eps, fmr, frt, "N/A"]) # countはループ内で計算していたためここでは簡易的に省略、必要ならリストに保存して出力
        
        print(f"[INFO] Data saved as '{csv_filename}'")

    def _plot_graphs(self, epsilons, fmr, frt):
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:blue'
        ax1.set_xlabel('Error Tolerance Epsilon (m)')
        ax1.set_ylabel('Formation Maintenance Ratio (FMR)', color=color)
        ax1.plot(epsilons, fmr, marker='o', color=color, label="FMR")
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0, 1.1)

        ax2 = ax1.twinx() 
        color = 'tab:orange'
        ax2.set_ylabel('Avg Formation Recovery Time (s)', color=color)
        ax2.plot(epsilons, frt, marker='x', linestyle='--', color=color, label="Avg FRT")
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('Performance Sensitivity (Single Trial)')
        fig.tight_layout()
        plt.grid(True)
        
        filename = "single_run_analysis.png"
        plt.savefig(filename)
        print(f"\n[INFO] Graph saved as '{filename}'")

def run_experiment():
    sim = MultiRobotSimulator(
        num_robots=CONFIG["num_robots"],
        render_mode="human" if CONFIG["render"] else "rgb_array",
        map_type="image",
        image_map_path=CONFIG["map_path"],
        formation_spacing_m=CONFIG["formation_spacing"],
        show_labels=False,
        manual_control=False,
        show_trail=False,
        # trail_step_interval=10
    )
    
    # === 固定座標を設定 ===
    # 内部の _recenter_robots, _assign_random_targets でこの値が使われます
    start_config = CONFIG["start_pos"]
    sim.fixed_start_pos_m = (start_config[0], start_config[1]) # (x, y)のみ
    sim.fixed_start_theta = start_config[2]                    # thetaのみ
    sim.fixed_goal_pos_m = CONFIG["goal_pos"]

    data_store = ExperimentData()
    clock = pygame.time.Clock()

    # 1回のみ実行
    print(f"--- Single Trial Execution ---")
    print(f"Start: {CONFIG['start_pos']}, Goal: {CONFIG['goal_pos']}")
    
    sim.set_map("image")
    # if not sim.policy_control_active:
    #     print("[INFO] Enabling Policy Control...")
    #     sim.start_policy_control()
    # else:
    #     print("[INFO] Policy Control is already active.")
    
    step = 0
    prev_centroid_m = None
    trial_errors_series = []
    
    done = False
    success = False
    
    while not done and step < CONFIG["max_steps"]:
        sim.process_events()
        if not sim.running: return

        sim.draw_scene()
        
        # --- 計測ロジック ---
        positions_m = np.array([sim.pixels_to_meters(*r.position) for r in sim.robot_states])
        curr_centroid_m = np.mean(positions_m, axis=0)
        
        max_error_in_step = np.nan 

        if prev_centroid_m is not None:
            move_vec = curr_centroid_m - prev_centroid_m
            dist_moved = np.linalg.norm(move_vec)
            
            if dist_moved > 0.001:
                unit_v = move_vec / dist_moved
                # 進行方向に対して垂直(右)ベクトル
                perp_v = np.array([unit_v[1], -unit_v[0]])
                
                offsets = (np.arange(CONFIG["num_robots"]) - (CONFIG["num_robots"] - 1) / 2) * CONFIG["formation_spacing"]
                
                step_errors = []
                for i, r_pos in enumerate(positions_m):
                    ideal_pos = curr_centroid_m + offsets[i] * perp_v
                    err = np.linalg.norm(r_pos - ideal_pos)
                    step_errors.append(err)
                
                max_error_in_step = max(step_errors)
        
        trial_errors_series.append(max_error_in_step)
        prev_centroid_m = curr_centroid_m
        step += 1
        
        # --- ゴール判定 ---
        if sim.team_nav.team_goal:
            goal_m = sim.team_nav.team_goal
            # 全員がゴール半径0.5m以内に入ったら成功
            dists = [np.linalg.norm(p - np.array(goal_m)) for p in positions_m]
            if all(d < 0.7 for d in dists):
                success = True
                done = True
        
        if CONFIG["render"]:
            clock.tick(CONFIG["fps_limit"])

    duration = step * (1/30.0)
    if success:
        print(f"  -> Success! ({duration:.2f}s)")
    else:
        print("  -> Failed / Timeout.")

    data_store.add_trial(duration, trial_errors_series, success)
    sim.close()
    
    # 結果表示
    data_store.analyze_and_plot()

if __name__ == "__main__":
    try:
        run_experiment()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()