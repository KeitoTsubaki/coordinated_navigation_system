import yaml
import math
from pathlib import Path

from PIL import Image
from scipy.ndimage import binary_dilation, distance_transform_edt
import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# --------------------------------------------------
# 1. A* 本体
# --------------------------------------------------
class VWAStarPlanner:
    def __init__(self):
        map_yaml_path = "grid_maps/room1_map/room1_map.yaml"
        map_yaml_path = "grid_maps/room2_map/room2_map.yaml"
    
        vwa_star_config_path = "vwa_star_config.yaml"

        self.start_pos = None
        self.goal_pos = None
        # self.expansion_radius = None
        self.expansion_radius_dict = {}
        self.costmap = None
        self.disp_map = None
        # self.expanded_costmap = None
        self.formation_priorities = []
        self.expanded_costmap_dict = {}
        self.start_rc = None
        self.goal_rc = None
        self.formation_cost = {}
        self.formation_colors = []
        self.path = None

        self.load_config(vwa_star_config_path)
        self.load_map(map_yaml_path)
        

    def load_config(self, config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.start_pos = tuple(config["start_pos"])  # (x, y) [m]
        self.goal_pos  = tuple(config["goal_pos"])  # (x, y) [m]
        self.expansion_radius_dict = config["expansion_radius_dict"]  # m
        self.formation_priorities = config["formation_priorities"]
        self.formation_cost = config["formation_cost"]
        self.formation_colors = config["formation_colors"]

    def load_map(self, map_yaml_path):
        with open(map_yaml_path, "r") as f:
            map_yaml = yaml.safe_load(f)

        image_file = map_yaml["image"]
        self.res = map_yaml["resolution"]      # 0.05
        origin = map_yaml["origin"]              # [-15.0, -15.0, 0.0]
        negate = map_yaml.get("negate", 0)
        occ_thresh = map_yaml.get("occupied_thresh", 0.65)
        free_thresh = map_yaml.get("free_thresh", 0.196)

        print("image:", image_file)
        print("self.res:", self.res)
        print("origin:", origin)

        image_path = Path(map_yaml_path).parent / image_file
        img = Image.open(image_path).convert("L")  # 'L' = 8bit grayscale
        img_arr = np.array(img, dtype=np.uint8)

        # 占有確率を計算
        if negate == 0:
            occ_prob = (255 - img_arr) / 255.0
        else:
            occ_prob = img_arr / 255.0

        # OccupancyGrid 風の値に変換
        # -1: unknown, 0: free, 100: occupied
        self.costmap = np.full(img_arr.shape, -1, dtype=np.int16)
        self.costmap[occ_prob >= occ_thresh] = 100
        self.costmap[occ_prob <= free_thresh] = 0

        print("grid shape:", self.costmap.shape)

        # マップのサイズ（メートル）を計算
        self.height, self.width = self.costmap.shape
        self.res = self.res
        self.origin_x, self.origin_y, self.origin_yaw = origin

        # Inflation
        for formation, expansion_radius in self.expansion_radius_dict.items():
            inflate_cells = int(expansion_radius / self.res)
            print(f"inflate cells: {inflate_cells} for formation: {formation}")

            obstacle = (self.costmap == 100)
            dist_cells = distance_transform_edt(~obstacle)

            # メートルに変換
            self.dist_map = dist_cells * self.res

            # self.expansion_radius 以内を全部障害物にする
            inflated_obstacle = self.dist_map <= expansion_radius

            expanded_costmap = np.zeros_like(self.costmap, dtype=np.int16)
            expanded_costmap[inflated_obstacle] = 100 / (len(self.formation_priorities) + 1)  # 軽いコストに設定
            if self.disp_map is None:
                self.disp_map = expanded_costmap
            else:
                self.disp_map += expanded_costmap
            self.expanded_costmap_dict[formation] = expanded_costmap
        self.disp_map += self.costmap

        self.start_rc = self.world_to_grid(self.start_pos[0], self.start_pos[1]) + (0,)
        self.goal_rc  = self.world_to_grid(self.goal_pos[0],  self.goal_pos[1])
        print("start grid:", self.start_rc, "goal grid:", self.goal_rc)

    # A* アルゴリズム本体
    def vwastar(self, allow_diagonal=True):
        """
        grid: 2D np.ndarray
            0 = free, それ以外 = obstacle として扱う
        start: (sy, sx)  グリッドの行・列
        goal:  (gy, gx)
        return: path のリスト [(y0,x0), (y1,x1), ...] or None
        """

        def in_bounds(y, x, costmap):
            h, w = costmap.shape
            return 0 <= y < h and 0 <= x < w

        def is_free(y, x, costmap):
            return costmap[y, x] == 0

        # 近傍（4 or 8 方向）
        # neighbors: [dy, dx, transition, cost]
        if allow_diagonal:
            neighbors = [
                (-1,  0, 1.0),
                ( 1,  0, 1.0),
                ( 0, -1, 1.0),
                ( 0,  1, 1.0),
                (-1, -1, np.sqrt(2)),
                (-1,  1, np.sqrt(2)),
                ( 1, -1, np.sqrt(2)),
                ( 1,  1, np.sqrt(2)),
            ]
        else:
            neighbors = [
                (-1,  0, 1.0),
                ( 1,  0, 1.0),
                ( 0, -1, 1.0),
                ( 0,  1, 1.0),
            ]

        # ヒューリスティック（ユークリッド距離）
        def heuristic(a, b):
            (y1, x1, formation_idx), (y2, x2) = a, b
            return np.hypot(y2 - y1, x2 - x1)

        # 優先度付きキュー (f, g, (y,x))
        open_heap = []
        heapq.heappush(open_heap, (0.0, 0.0, self.start_rc))

        came_from = {}  # (y,x) -> (py,px)
        g_score = {self.start_rc: 0.0}

        closed = set()

        while open_heap:
            f, g, current = heapq.heappop(open_heap)    # pick up lowest f score, which is a heapq characteristic
            cy, cx, cformation_idx = current

            if current in closed:
                continue
            closed.add(current)

            # ゴールに着いたら復元
            if current[:-1] == self.goal_rc:
                path = [current]
                while path[-1] in came_from:
                    path.append(came_from[path[-1]])
                path.reverse()
                self.path = path
                return path

            for formation_idx, formation in enumerate(self.formation_priorities):
                for dy, dx, cost in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if not in_bounds(ny, nx, self.expanded_costmap_dict[formation]):
                        continue
                    if not is_free(ny, nx, self.expanded_costmap_dict[formation]):
                        continue

                    tentative_g = g + cost + self.formation_cost[formation]
                    neighbor = (ny, nx, formation_idx)

                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        g_score[neighbor] = tentative_g
                        came_from[neighbor] = current
                        f_score = tentative_g + heuristic(neighbor, self.goal_rc)
                        heapq.heappush(open_heap, (f_score, tentative_g, neighbor))

        # 経路なし
        self.path = None
    
    def show(self, path=None):
        if path is not None:
            print("path length (cells):", len(path))

            # 経路をワールド座標配列に変換（プロット用）
            xs = []
            ys = []
            segment_colors = []
            for r, c, formation_idx in path:
                xs.append(self.grid_to_world(r, c)[0])
                ys.append(self.grid_to_world(r, c)[1])
                segment_colors.append(self.formation_colors[formation_idx])
            # path_xy = [self.grid_to_world(r, c) for (r, c, formation_idx) in path]
            # xs = [p[0] for p in path_xy]
            # ys = [p[1] for p in path_xy]

            
            points = np.array([xs, ys]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # 座標軸を [m] に変換
            extent = [
                self.origin_x,               # x_min
                self.origin_x + self.width * self.res, # x_max
                self.origin_y,               # y_min
                self.origin_y + self.height * self.res # y_max
            ]

            # 可視化: self.expanded_costmap の上にパスを重ねる
            plt.figure(figsize=(6, 6))
            plt.imshow(
                self.disp_map,
                origin="lower",
                extent=extent,
                cmap="gray_r",
                vmin=0,
                vmax=100,
            )
            # plt.plot(xs, ys, "-r", linewidth=2, label="VWA* path")
            lc = LineCollection(segments, colors=segment_colors[:-1], linewidths=2)
            plt.gca().add_collection(lc)
            plt.scatter([self.start_pos[0]], [self.start_pos[1]], c="b", s=40, label="start")
            plt.scatter([self.goal_pos[0]],  [self.goal_pos[1]],  c="g", s=40, label="goal")
            plt.xlabel("x [m]")
            plt.ylabel("y [m]")
            plt.title("VWA* path")
            plt.legend()
            plt.tight_layout()
            plt.show()



    # --------------------------------------------------
    # 2. ワールド座標 <-> グリッド座標 変換
    # --------------------------------------------------
    def world_to_grid(self, x, y):
        """
        x, y: [m] （地図座標系）
        self.origin_x, self.origin_y: YAML の origin (左下) [m]
        self.res: [m/cell]
        return: (row, col) = (y_index, x_index)
        """
        col = int((x - self.origin_x) / self.res)
        row = int((y - self.origin_y) / self.res)
        return row, col


    def grid_to_world(self, row, col):
        """
        グリッドのセル中心座標を [m] に変換
        """
        x = self.origin_x + (col + 0.5) * self.res
        y = self.origin_y + (row + 0.5) * self.res
        return x, y


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

class DiffDriveHybridAStarPlanner:
    def __init__(
        self,
        dist_map: np.ndarray,
        resolution: float,
        origin_x: float,
        origin_y: float,
        linear_speed: float = 0.6,
        max_steer: float = 0.8,
        dt: float = 0.2,
        num_heading_bins: int = 72,
    ):
        self.h_grid, self.w_grid = dist_map.shape
        self.res = resolution
        self.ox = origin_x
        self.oy = origin_y
        self.linear_speed = linear_speed
        self.dt = dt
        self.num_heading_bins = num_heading_bins

        # 生の距離マップを保持 (これを使って動的に半径チェックを行う)
        self.dist_map_raw = dist_map

        # Motion Primitives (Forward Only)
        self.motion_primitives = []
        steers = [-max_steer, -max_steer*0.5, 0, max_steer*0.5, max_steer]
        for s in steers:
            self.motion_primitives.append((linear_speed, s))

    def world_to_grid(self, x, y):
        gx = int(np.round((x - self.ox) / self.res))
        gy = int(np.round((y - self.oy) / self.res))
        return gx, gy

    def heading_to_bin(self, theta):
        theta = (theta + 2 * math.pi) % (2 * math.pi)
        bin_size = 2 * math.pi / self.num_heading_bins
        return int(theta / bin_size)

    def is_in_bounds(self, gx, gy):
        return 0 <= gx < self.w_grid and 0 <= gy < self.h_grid

    # --- 衝突判定: 指定された半径 (formation_radius) を使用 ---
    def is_collision(self, x, y, formation_radius):
        gx, gy = self.world_to_grid(x, y)
        if not self.is_in_bounds(gx, gy):
            return True
        
        # 距離マップの値が、現在のフォーメーションの半径より小さければ衝突
        if self.dist_map_raw[gy, gx] <= formation_radius:
            return True
        return False

    # --- ヒューリスティック: 指定された半径を考慮して障害物を避ける ---
    def calc_holonomic_heuristic(self, goal_gx, goal_gy, formation_radius):
        h_map = np.full((self.h_grid, self.w_grid), np.inf)
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
                
                # 壁判定: 指定半径以下の場所は通れないものとしてコスト無限大
                dist = self.dist_map_raw[ny, nx]
                # if dist <= formation_radius: 
                #     continue
                
                # 壁に近い場合のペナルティ (インフレーション効果)
                penalty = 0.0
                margin = 0.0 # マージン[m]
                if dist < formation_radius + margin:
                    penalty = 1.0 * (formation_radius + margin - dist)
                
                ncost = cost + w * self.res + penalty
                if ncost < h_map[ny, nx]:
                    h_map[ny, nx] = ncost
                    heapq.heappush(queue, (ncost, nx, ny))
        plt.figure(figsize=(6,6))
        plt.imshow(h_map, origin="lower", cmap="jet", alpha=0.5)
        plt.show()
        return h_map

    def normalize_angle(self, angle):
        while angle > math.pi: angle -= 2.0 * math.pi
        while angle < -math.pi: angle += 2.0 * math.pi
        return angle

    # --- Plan: formation_radius を受け取り、コストマップを切り替えたのと同等の処理を行う ---
    def plan(self, start_pose, goal_pose, formation_radius, max_iter=30000):
        sx, sy, stheta = start_pose
        gx, gy, gtheta = goal_pose # GoalにThetaを含める
        
        sgx, sgy = self.world_to_grid(sx, sy)
        ggx, ggy = self.world_to_grid(gx, gy)

        if not self.is_in_bounds(sgx, sgy) or not self.is_in_bounds(ggx, ggy):
            print("Start or Goal out of bounds")
            return None
        
        # ゴール地点が現在のフォーメーションで安全かチェック
        # if self.dist_map_raw[ggy, ggx] <= formation_radius:
        #      print(f"Goal is in collision for radius {formation_radius:.2f}")
        #      return None

        # フォーメーションに応じたヒューリスティックマップを計算
        # print(f"  Calclating Heuristic for radius {formation_radius}...")
        self.h_map = self.calc_holonomic_heuristic(ggx, ggy, formation_radius)
        
        if self.h_map[sgy, sgx] == np.inf:
            print("Error: Goal unreachable in Heuristic.")
            return None

        open_heap = []
        # f = g + h
        h_start = self.h_map[sgy, sgx]
        heapq.heappush(open_heap, (h_start, 0.0, 0, (sx, sy, stheta)))

        came_from = {0: None}
        state_data = {0: (sx, sy, stheta)}
        visited = np.full((self.h_grid, self.w_grid, self.num_heading_bins), np.inf)
        visited[sgy, sgx, self.heading_to_bin(stheta)] = 0.0

        node_id_counter = 1
        goal_found_id = None
        
        # 許容誤差
        dist_tolerance = 0.3      # [m]
        angle_tolerance = 0.3  # [rad] 約17度

        for _ in range(max_iter):
            if not open_heap: break
            f, g, cid, (cx, cy, ctheta) = heapq.heappop(open_heap)

            # === Goal Check with Theta ===
            dist_err = math.hypot(cx - gx, cy - gy)
            angle_err = abs(self.normalize_angle(ctheta - gtheta))
            
            if dist_err < dist_tolerance and angle_err < angle_tolerance:
                goal_found_id = cid
                break

            for v, omega in self.motion_primitives:
                nx = cx + v * math.cos(ctheta) * self.dt
                ny = cy + v * math.sin(ctheta) * self.dt
                ntheta = self.normalize_angle(ctheta + omega * self.dt)

                # 衝突判定 (Formation Radiusを使用)
                if self.is_collision(nx, ny, formation_radius): continue
                
                n_gx, n_gy = self.world_to_grid(nx, ny)
                if not self.is_in_bounds(n_gx, n_gy): continue

                # step_cost = v * self.dt * (1.0 + 0.5 * abs(omega)) 
                step_cost = v * self.dt * (1.1 if abs(omega) > 0.1 else 1.0)
                ng = g + step_cost
                hbin = self.heading_to_bin(ntheta)

                if ng < visited[n_gy, n_gx, hbin]:
                    visited[n_gy, n_gx, hbin] = ng
                    nid = node_id_counter
                    node_id_counter += 1
                    state_data[nid] = (nx, ny, ntheta)
                    came_from[nid] = cid
                    
                    h_val = self.h_map[n_gy, n_gx]
                    # f = g + h
                    heapq.heappush(open_heap, (ng + 1.2 * h_val, ng, nid, (nx, ny, ntheta)))

        if goal_found_id is None:
            return None
        
        path = []
        curr = goal_found_id
        while curr is not None:
            path.append(state_data[curr])
            curr = came_from[curr]
        path.reverse()
        return path


# ==========================================
# 3. Connection Logic (Formation Aware)
# ==========================================

def extract_waypoints_with_yaw(vwa_planner, grid_path, delta=3):
    """ 
    VWA*のgrid_pathからWaypointを抽出し、
    グリッドパスの局所的な形状に基づいた正確な接線(Yaw)を計算する。
    
    delta: 接線計算のために前後何マス見るか（ノイズ除去のため2~5推奨）
    """
    if not grid_path: return []
    
    waypoints = []
    path_len = len(grid_path)

    # インナー関数: インデックスiにおけるワールド座標を取得
    def get_world_pos(index):
        # indexが範囲外ならクリップする
        safe_idx = max(0, min(index, path_len - 1))
        r, c, _ = grid_path[safe_idx]
        return vwa_planner.grid_to_world(r, c)

    # 分岐点（Start, Change, Goal）のインデックスをリストアップ
    key_indices = [0] # Start
    for i in range(1, path_len - 1):
        # Formation IDが変わる地点
        if grid_path[i][2] != grid_path[i-1][2]:
            key_indices.append(i)
    key_indices.append(path_len - 1) # Goal

    # 各キーポイントでの処理
    for idx in key_indices:
        # 1. 現在位置の座標とフォーメーションID
        curr_r, curr_c, fid = grid_path[idx]
        wx, wy = vwa_planner.grid_to_world(curr_r, curr_c)
        
        # 2. 接線(Yaw)の計算：中央差分 (Central Difference)
        # grid_path上で、現在地よりdelta個前と、delta個先の点を見る
        prev_x, prev_y = get_world_pos(idx - delta)
        next_x, next_y = get_world_pos(idx + delta)
        
        # Start地点の特例: 前がないので、現在地と次の点を見る (前方差分)
        if idx < delta:
            prev_x, prev_y = wx, wy
            next_x, next_y = get_world_pos(idx + delta)
            
        # Goal地点の特例: 次がないので、前の点と現在地を見る (後方差分)
        elif idx >= path_len - delta:
            prev_x, prev_y = get_world_pos(idx - delta)
            next_x, next_y = wx, wy

        # 角度計算
        yaw = math.atan2(next_y - prev_y, next_x - prev_x)
        
        waypoints.append((wx, wy, yaw, fid))
        
    return waypoints

def connect_waypoints_formation_aware(vwa_planner, waypoints):
    """
    Waypoints間を接続する。
    セグメントごとに、そのFormationに対応するコストマップ（半径）を使用する。
    """
    
    # Hybrid A* 初期化 (マップ情報は共通)
    hybrid_planner = DiffDriveHybridAStarPlanner(
        dist_map=vwa_planner.dist_map,
        resolution=vwa_planner.res,
        origin_x=vwa_planner.origin_x,
        origin_y=vwa_planner.origin_y,
        linear_speed=0.6,
        dt=0.2
    )
    
    full_path = []
    
    print("\n=== Starting Formation-Aware Hybrid A* ===")
    
    for i in range(len(waypoints) - 1):
        wp_start = waypoints[i]
        wp_end   = waypoints[i+1]
        
        # セグメントのフォーメーションID (Start時点のものを使用)
        fid = wp_start[3]
        formation_name = vwa_planner.formation_priorities[fid]
        
        # そのフォーメーションの占有半径を取得
        # これにより実質的にコストマップを切り替える
        radius = vwa_planner.expansion_radius_dict[formation_name]
        
        start_pose = (wp_start[0], wp_start[1], wp_start[2])
        goal_pose  = (wp_end[0],   wp_end[1],   wp_end[2]) # Thetaを含む
        
        print(f"Segment {i+1} [{formation_name}, r={radius}m]:")
        print(f"  Start: ({start_pose[0]:.1f}, {start_pose[1]:.1f}, {math.degrees(start_pose[2]):.0f}°)")
        print(f"  Goal : ({goal_pose[0]:.1f}, {goal_pose[1]:.1f}, {math.degrees(goal_pose[2]):.0f}°)")
        
        # Plan実行 (radiusを渡す)
        segment_path = hybrid_planner.plan(start_pose, goal_pose, formation_radius=radius)
        
        if segment_path is None:
            print("  -> Plan Failed. Interpolating.")
            # 失敗時の補間処理 (省略)
        else:
            print(f"  -> Success: {len(segment_path)} steps.")
            for p in segment_path:
                full_path.append((p[0], p[1], p[2], fid))
                
    return full_path

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    # 1. Global Plan (VWA*)
    vwa = VWAStarPlanner()
    grid_path = vwa.vwastar() # 簡易デモ用の固定パス
    
    # 2. Extract Waypoints
    waypoints = extract_waypoints_with_yaw(vwa, grid_path)
    
    # 3. Local Plan (Hybrid A* with Formation Constraints)
    final_path = connect_waypoints_formation_aware(vwa, waypoints)
    
    # 4. Visualization
    plt.figure(figsize=(10, 10))
    
    # 背景: Raw Dist Map (障害物との距離)
    plt.imshow(vwa.dist_map, origin="lower", 
               extent=[vwa.origin_x, vwa.origin_x + vwa.width*vwa.res, 
                       vwa.origin_y, vwa.origin_y + vwa.height*vwa.res],
               cmap="bone_r")
    
    # Waypoints
    wx = [w[0] for w in waypoints]
    wy = [w[1] for w in waypoints]
    plt.scatter(wx, wy, c='red', s=80, marker='o', label="Waypoints", zorder=10)
    
    # Orientation Arrows at Waypoints
    for w in waypoints:
        plt.arrow(w[0], w[1], 0.5*math.cos(w[2]), 0.5*math.sin(w[2]), 
                  head_width=0.2, color='red', zorder=10)

    # Generated Path
    if final_path:
        px = [p[0] for p in final_path]
        py = [p[1] for p in final_path]
        
        # フォーメーションごとに色を変えてプロット
        # formation_colors = vwa.config["formation_colors"]
        # for i in range(len(final_path)-1):
        #     fid = final_path[i][3]
        #     color = formation_colors[fid] if fid < len(formation_colors) else "black"
        #     plt.plot([px[i], px[i+1]], [py[i], py[i+1]], color=color, linewidth=2)
        
        plt.plot(px, py, "b-", linewidth=2, label="Hybrid A* Path")

    plt.title("Hybrid A* Connection with Formation-Dependent Radius & Goal Theta")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.legend()
    plt.grid(True)
    plt.show()