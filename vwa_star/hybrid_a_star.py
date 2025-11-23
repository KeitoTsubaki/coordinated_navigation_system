import yaml
from pathlib import Path
from PIL import Image
from scipy.ndimage import distance_transform_edt
import math
import heapq
import numpy as np
import matplotlib.pyplot as plt
import copy
import time

# ==========================================
# 1. Hybrid A* Planner (前進のみ設定)
# ==========================================
class DiffDriveHybridAStarPlanner:
    def __init__(
        self,
        occ_grid: np.ndarray,
        resolution: float,
        origin_x: float,
        origin_y: float,
        robot_radius: float = 0.4,
        linear_speed: float = 0.6,          # [m/s]
        max_steer: float = 0.8,             # [rad/s] 最大角速度
        dt: float = 0.2,                    # [s]
        num_heading_bins: int = 72,
    ):
        self.grid = occ_grid
        self.h, self.w = occ_grid.shape
        self.res = resolution
        self.ox = origin_x
        self.oy = origin_y
        self.robot_radius = robot_radius
        self.linear_speed = linear_speed
        self.dt = dt
        self.num_heading_bins = num_heading_bins

        # 前進のみ (Forward Only)
        moves = []
        steers = [-max_steer, -max_steer*0.5, 0, max_steer*0.5, max_steer]
        for s in steers:
            moves.append((linear_speed, s)) # 速度は常にプラス
        self.motion_primitives = moves

        # 衝突判定用：半径[セル]
        self.collision_radius_cells = max(1, int(math.ceil(robot_radius / resolution)))

    def world_to_grid(self, x, y):
        gx = int(np.round((x - self.ox) / self.res))
        gy = int(np.round((y - self.oy) / self.res))
        return gx, gy

    def grid_to_world(self, gx, gy):
        x = self.ox + gx * self.res
        y = self.oy + gy * self.res
        return x, y

    def heading_to_bin(self, theta):
        theta = (theta + 2 * math.pi) % (2 * math.pi)
        bin_size = 2 * math.pi / self.num_heading_bins
        return int(theta / bin_size)

    def is_in_bounds(self, gx, gy):
        return 0 <= gx < self.w and 0 <= gy < self.h

    def is_collision(self, x, y):
        gx, gy = self.world_to_grid(x, y)
        if not self.is_in_bounds(gx, gy):
            return True
        if self.grid[gy, gx] > 50:
            return True
        return False

    def calc_holonomic_heuristic(self, goal_gx, goal_gy):
        h_map = np.full((self.h, self.w), np.inf)
        h_map[goal_gy, goal_gx] = 0.0
        queue = [(0.0, goal_gx, goal_gy)]
        motions = [
            (1, 0, 1.0), (0, 1, 1.0), (-1, 0, 1.0), (0, -1, 1.0),
            (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)
        ]
        while queue:
            cost, cx, cy = heapq.heappop(queue)
            if cost > h_map[cy, cx]:
                continue
            for dx, dy, w in motions:
                nx, ny = cx + dx, cy + dy
                if not self.is_in_bounds(nx, ny): continue
                if self.grid[ny, nx] > 50: continue
                ncost = cost + w * self.res
                if ncost < h_map[ny, nx]:
                    h_map[ny, nx] = ncost
                    heapq.heappush(queue, (ncost, nx, ny))
        
        return h_map

    def plan(self, start_world, goal_world, max_iter=100000):
        sx, sy, stheta = start_world
        gx_w, gy_w = goal_world
        
        sgx, sgy = self.world_to_grid(sx, sy)
        goal_gx, goal_gy = self.world_to_grid(gx_w, gy_w)

        if not self.is_in_bounds(sgx, sgy) or not self.is_in_bounds(goal_gx, goal_gy):
            return None
        if self.grid[goal_gy, goal_gx] > 50:
             return None

        print("Calculating Heuristic Map...")
        self.h_map = self.calc_holonomic_heuristic(goal_gx, goal_gy)
        if self.h_map[sgy, sgx] == np.inf:
            print("Error: Goal unreachable.")
            return None

        print("Starting Hybrid A*...")
        open_heap = []
        start_state = (sx, sy, stheta)
        start_id = 0
        heapq.heappush(open_heap, (self.h_map[sgy, sgx], 0.0, start_id, start_state))

        came_from = {start_id: None}
        state_data = {start_id: start_state}
        visited = np.full((self.h, self.w, self.num_heading_bins), np.inf)
        visited[sgy, sgx, self.heading_to_bin(stheta)] = 0.0

        node_id_counter = 1
        goal_found_id = None
        dist_tolerance = 0.01

        for _ in range(max_iter):
            if not open_heap: break
            f, g, cid, (cx, cy, ctheta) = heapq.heappop(open_heap)

            if math.hypot(cx - gx_w, cy - gy_w) < dist_tolerance:
                goal_found_id = cid
                break

            for v, omega in self.motion_primitives:
                # Diff Drive Kinematics
                nx = cx + v * math.cos(ctheta) * self.dt
                ny = cy + v * math.sin(ctheta) * self.dt
                ntheta = (ctheta + omega * self.dt + math.pi) % (2 * math.pi) - math.pi

                if self.is_collision(nx, ny): continue
                n_gx, n_gy = self.world_to_grid(nx, ny)
                if not self.is_in_bounds(n_gx, n_gy): continue

                step_cost = v * self.dt * (1.1 if abs(omega) > 0.1 else 1.0)
                ng = g + step_cost
                hbin = self.heading_to_bin(ntheta)

                if ng < visited[n_gy, n_gx, hbin]:
                    visited[n_gy, n_gx, hbin] = ng
                    nid = node_id_counter
                    node_id_counter += 1
                    state_data[nid] = (nx, ny, ntheta)
                    came_from[nid] = cid
                    h_val = max(self.h_map[n_gy, n_gx], math.hypot(nx - gx_w, ny - gy_w))
                    heapq.heappush(open_heap, (ng + 1.0 * h_val, ng, nid, (nx, ny, ntheta)))

        if goal_found_id is None:
            print("Path not found.")
            return None
        
        path = []
        curr = goal_found_id
        while curr is not None:
            path.append(state_data[curr])
            curr = came_from[curr]
        path.reverse()
        return path

# ==========================================
# 2. Path Smoother
# ==========================================
class PathSmoother:
    def __init__(self, max_iter=100, alpha=0.1, beta=0.5, obs_weight=0.2, obs_dist_thresh=1.0):
        """
        alpha: 平滑化の強さ (大きくすると直線的になる)
        beta:  元のパスへの追従力 (大きくすると元のパスに戻ろうとする)
        obs_weight: 障害物からの反発力
        obs_dist_thresh: これ以上障害物に近いと反発を受ける距離 [m]
        """
        self.max_iter = max_iter
        self.alpha = alpha 
        self.beta = beta
        self.obs_weight = obs_weight
        self.obs_dist_thresh = obs_dist_thresh

    def smooth(self, path, dist_map, resolution, origin_x, origin_y):
        """
        path: [(x,y,theta), ...]
        dist_map: 障害物からの距離マップ [m] (scipy.distance_transform_edt の出力)
        """
        if not path or len(path) < 3:
            return path

        # (x, y) だけ取り出して最適化する
        path_xy = np.array([[p[0], p[1]] for p in path], dtype=float)
        orig_xy = path_xy.copy()
        n_points = len(path_xy)
        
        h, w = dist_map.shape

        for _ in range(self.max_iter):
            # 両端(Start, Goal)は固定
            for i in range(1, n_points - 1):
                curr = path_xy[i]
                prev = path_xy[i-1]
                next_p = path_xy[i+1]
                orig = orig_xy[i]

                # 1. Smoothness (Neighbors)
                gradient_smooth = (prev + next_p - 2 * curr)
                
                # 2. Data Fidelity (Original Path)
                gradient_data = (orig - curr)
                
                # 3. Obstacle Repulsion
                gradient_obs = np.zeros(2)
                # 現在位置のグリッド座標
                gx = int(np.round((curr[0] - origin_x) / resolution))
                gy = int(np.round((curr[1] - origin_y) / resolution))
                
                if 0 <= gx < w and 0 <= gy < h:
                    dist = dist_map[gy, gx]
                    if dist < self.obs_dist_thresh:
                        # 障害物に近い場合、勾配方向に押し戻す
                        # 簡易的に周囲の勾配を見る (Sobel的な処理の簡易版)
                        # ここでは「最も距離が大きい方向」へ動かしたい
                        # 周囲の距離値を取得して数値微分
                        d_x = 0.0
                        d_y = 0.0
                        if gx+1 < w and gx-1 >= 0:
                            d_x = dist_map[gy, gx+1] - dist_map[gy, gx-1]
                        if gy+1 < h and gy-1 >= 0:
                            d_y = dist_map[gy+1, gx] - dist_map[gy-1, gx]
                        
                        # 距離マップの勾配方向 = 障害物から離れる方向
                        norm = math.hypot(d_x, d_y)
                        if norm > 0.01:
                            gradient_obs = np.array([d_x, d_y]) / norm * (self.obs_dist_thresh - dist)

                # Update
                path_xy[i] += self.alpha * gradient_smooth + \
                              self.beta * gradient_data + \
                              self.obs_weight * gradient_obs

        # theta の再計算 (移動方向に向ける)
        new_path = []
        # Start
        new_path.append((path_xy[0][0], path_xy[0][1], path[0][2]))
        
        for i in range(1, n_points):
            px, py = path_xy[i]
            prev_x, prev_y = path_xy[i-1]
            yaw = math.atan2(py - prev_y, px - prev_x)
            new_path.append((px, py, yaw))
            
        return new_path


# ==========================================
# Main Execution
# ==========================================

yaml_path = "grid_maps/room1_map/room1_map.yaml" 
yaml_path = "grid_maps/room2_map/room2_map.yaml" 
inflation_radius = 1

# マップ読み込み
try:
    with open(yaml_path, "r") as f:
        info = yaml.safe_load(f)
except FileNotFoundError:
    info = {"image": "dummy", "resolution": 0.05, "origin": [-10.0, -10.0, 0.0]}

resolution = info["resolution"]
origin = info["origin"]
image_file = info.get("image", "")
image_path = Path(yaml_path).parent / image_file

if image_path.exists():
    img = Image.open(image_path).convert("L")
    img_arr = np.array(img, dtype=np.uint8)
    negate = info.get("negate", 0)
    occ_thresh = info.get("occupied_thresh", 0.65)
    if negate == 0:
        occ_prob = (255 - img_arr) / 255.0
    else:
        occ_prob = img_arr / 255.0
    occ_grid = np.zeros(img_arr.shape, dtype=np.int16)
    occ_grid[occ_prob >= occ_thresh] = 100 
else:
    print("Warning: Image not found, using dummy map.")
    occ_grid = np.zeros((400, 400), dtype=np.int16)
    occ_grid[150:250, 200:210] = 100 
    occ_grid[150:160, 150:200] = 100 
    occ_grid[240:250, 150:200] = 100 

# インフレーション & 距離マップ計算
# dist_map_raw: 障害物までの実距離[m]
dist_cells = distance_transform_edt(occ_grid != 100)
dist_map_raw = dist_cells * resolution

inflated_grid = np.zeros_like(occ_grid, dtype=np.int16)
inflated_grid[dist_map_raw <= inflation_radius] = 100 

# Planner (前進のみ)
planner = DiffDriveHybridAStarPlanner(
    inflated_grid,
    resolution=resolution,
    origin_x=origin[0],
    origin_y=origin[1],
    robot_radius=0.4,
    linear_speed=0.6,
    dt=0.2,
    num_heading_bins=72
)

start_world = (5.0, 5.0, 3.14) # 左向き
goal_world  = (3.0, -2.0)

# 1. Plan

start_time = time.time()

raw_path = planner.plan(start_world, goal_world)

plan_end_time = time.time()
print(f"Planning done in {plan_end_time - start_time:.2f} seconds.")


if raw_path is not None:
    print("Path found! Smoothing...")
    # 2. Smooth
    smoother = PathSmoother(
        max_iter=500,       # 反復回数
        alpha=0.5,          # 平滑化ゲイン (高いとより滑らか)
        beta=0.01,           # 原形維持ゲイン (低いと大きく変形できる)
        obs_weight=0.2,     # 障害物反発ゲイン
        obs_dist_thresh=0.8 # 障害物から 0.8m 以内なら反発
    )
    
    # Smootherには生の距離マップ(dist_map_raw)を渡して、インフレーション領域内でも最適化できるようにする
    # ただしplannerで使ったinflated_gridではなく、壁までの実距離を使うのがポイント
    smoothed_path = smoother.smooth(
        raw_path, 
        dist_map_raw, 
        resolution, 
        origin[0], 
        origin[1]
    )
    
    end_time = time.time()
    print(f"Smoothing done in {end_time - start_time:.2f} seconds.")

    # ... (前略: end_timeの計測までのコードはそのまま) ...
    
    # === 可視化設定 ===
    # 1行2列のレイアウトを作成 (figsizeを横長に変更)
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    
    # 共通の設定値
    h, w = inflated_grid.shape
    extent = [origin[0], origin[0] + w * resolution, origin[1], origin[1] + h * resolution]

    # === 1枚目: 左側 (Raw Path & Heuristic Map) ===
    ax0 = axes[0]
    ax0.set_title("Step 1: Hybrid A* Raw Path & Heuristic")
    
    # 背景: ヒューリスティックマップを表示して探索の様子を確認
    ax0.imshow(inflated_grid + occ_grid, origin="lower", extent=extent, cmap="gray_r", vmin=0, vmax=150)
    ax0.imshow(planner.h_map, origin="lower", extent=extent, cmap="jet", alpha=0.5) # ヒートマップ
    
    # Raw Path (Red)
    rx = [p[0] for p in raw_path]
    ry = [p[1] for p in raw_path]
    ax0.plot(rx, ry, "--r", linewidth=2, label="Raw Path")
    
    # Start/Goal
    ax0.scatter([start_world[0]], [start_world[1]], c="b", s=50, label="Start")
    ax0.scatter([goal_world[0]], [goal_world[1]], c="g", s=50, label="Goal")
    ax0.legend()

    # === 2枚目: 右側 (Smoothed Path & Result) ===
    ax1 = axes[1]
    ax1.set_title("Step 2: Smoothed Path (Final Result)")

    # 背景: 障害物マップのみ（クリアに見せるため）
    ax1.imshow(inflated_grid + occ_grid, origin="lower", extent=extent, cmap="gray_r", vmin=0, vmax=150)
    
    # Raw Path (薄く表示して比較用)
    ax1.plot(rx, ry, "--r", linewidth=1, alpha=0.3, label="Raw Path (Ref)")

    # Smoothed Path (Blue)
    sx = [p[0] for p in smoothed_path]
    sy = [p[1] for p in smoothed_path]
    ax1.plot(sx, sy, "-b", linewidth=3, label="Smoothed")
    
    # Start/Goal
    ax1.scatter([start_world[0]], [start_world[1]], c="b", s=50)
    ax1.scatter([goal_world[0]], [goal_world[1]], c="g", s=50)

    # 矢印 (Smoothed Path上の向き)
    step = max(1, len(smoothed_path)//15)
    step = 10
    for i in range(0, len(smoothed_path), step):
        px, py, pt = smoothed_path[i]
        # ax.arrowはplt.arrowと少し挙動が違うため注意
        ax1.arrow(px, py, 0.4*math.cos(pt), 0.4*math.sin(pt), 
                  head_width=0.3, color='green', alpha=0.8)

    ax1.legend()

    # レイアウト調整と表示
    plt.tight_layout()
    plt.show()

else:
    print("Path planning failed, skipping smoothing.")