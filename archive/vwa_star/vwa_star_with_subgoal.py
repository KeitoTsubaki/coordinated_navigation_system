import yaml
import math
import time
from pathlib import Path

from PIL import Image
from scipy.ndimage import binary_dilation, distance_transform_edt
import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

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
        # img_arr = np.flipud(img_arr)

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
                
                self.segment_colors = []
                for r, c, formation_idx in path:
                    self.segment_colors.append(self.formation_colors[formation_idx])
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
            
            path_xy = [self.grid_to_world(r, c) for (r, c, formation_idx) in path]
            xs = [p[0] for p in path_xy]
            ys = [p[1] for p in path_xy]

            
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
            lc = LineCollection(segments, colors=self.segment_colors[:-1], linewidths=2)
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



if __name__ == "__main__":
    vwa_star_planner = VWAStarPlanner()
    # 1. Get the raw path in Grid Coordinates (row, col, formation_index)
    smoother = PathSmoother(
        max_iter=500,       
        alpha=0.5,          
        beta=0.01,           
        obs_weight=1,     
        obs_dist_thresh=1.0 
    )

    start_time = time.time()
    grid_path = vwa_star_planner.vwastar(allow_diagonal=True)
    end_time = time.time()
    print(f"Path planning took {end_time - start_time:.2f} seconds.")
    # vwa_star_planner.show(grid_path)
    if grid_path is None:
        print("No path found!")
    else:
        # vwa_star_planner.show(grid_path)
        
        # --- CRITICAL FIX 1: Convert Grid Path to World Coordinates ---
        # The smoother expects [x, y, theta] in meters, not grid indices.
        world_path = []
        for r, c, formation_idx in grid_path:
            wx, wy = vwa_star_planner.grid_to_world(r, c)
            # Passing 0.0 as dummy theta since the smoother calculates yaw later
            world_path.append((wx, wy, 0.0))

        
        # 2. Smooth the World Coordinate Path
        smoothed_path = smoother.smooth(
            world_path, 
            vwa_star_planner.dist_map, 
            vwa_star_planner.res, 
            vwa_star_planner.origin_x, 
            vwa_star_planner.origin_y
        )
        smooth_time = time.time()
        print(f"Path smoothing took {smooth_time - end_time:.2f} seconds.")
        

        # Extract X and Y for plotting
        sx = [p[0] for p in smoothed_path]
        sy = [p[1] for p in smoothed_path]
        
        plt.figure(figsize=(6,6))
        plt.imshow(
            vwa_star_planner.disp_map,
            origin="lower",
            extent=[
                vwa_star_planner.origin_x,
                vwa_star_planner.origin_x + vwa_star_planner.width * vwa_star_planner.res,
                vwa_star_planner.origin_y,
                vwa_star_planner.origin_y + vwa_star_planner.height * vwa_star_planner.res
            ],
            cmap="gray_r",
            vmin=0,   # Ensure scale matches logic (0=free, 100=occupied)
            vmax=100
        )
        plt.rcParams["font.family"] = "DejaVu Serif"   # 使用するフォント
        plt.rcParams["font.size"] = 15 
        points = np.array([sx, sy]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, colors=vwa_star_planner.segment_colors[:-1], linewidths=2)
        plt.gca().add_collection(lc)
        
        legend_elements = [
            Line2D([0], [0], color='red', lw=2, label='Side-by-side formation'),   # 色とラベルを指定
            Line2D([0], [0], color='blue', lw=2, label='One robot'),
            mpatches.Patch(color='0.66', label='margin for side-by-side'),
            mpatches.Patch(color='0.33', label='margin for one robot')
        ]
        
        # --- CRITICAL FIX 2: Use plt.plot instead of plt.imshow ---
        # plt.plot(sx, sy, "-b", linewidth=2, label="Smoothed path")
        plt.text(vwa_star_planner.start_pos[0], vwa_star_planner.start_pos[1]+0.5, "Start", color="black", ha="center")
        plt.text(vwa_star_planner.goal_pos[0],  vwa_star_planner.goal_pos[1]+0.5,  "Goal",  color="black", ha="center")
        plt.scatter([vwa_star_planner.start_pos[0]], [vwa_star_planner.start_pos[1]], c="b", s=60, zorder=5)
        plt.scatter([vwa_star_planner.goal_pos[0]],  [vwa_star_planner.goal_pos[1]],  c="g", s=60, zorder=5)

        # step = int(1 / vwa_star_planner.res)    # 1m ごとに点を打つ
        # steped_x = sx[::step]
        # steped_y = sy[::step]
        # # スライス機能 [::step] を使う
        # plt.scatter(steped_x, steped_y, color='skyblue', s=20, zorder=5)
        
        # 1. 計算用にNumpy配列化
        path_x = np.array(sx)
        path_y = np.array(sy)

        # 2. 全点の勾配（接線ベクトル）を計算
        # np.gradient を使うと、始点・終点・中間点をよしなに処理して微分値を算出してくれます
        dx_dt = np.gradient(path_x)
        dy_dt = np.gradient(path_y)

        # 3. 表示する点の間隔（インデックス）
        # 例: 1.0m 間隔 (解像度が0.05mなら20点飛ばし)
        step = int(1.0 / vwa_star_planner.res)+ 1
        indices = range(step, len(path_x), step)

        # 4. 垂直線（ヒゲ）のセグメントを作成
        perp_lines = []
        tips_x = []  # 両端の円のためのX座標リスト
        tips_y = []  # 両端の円のためのY座標リスト
        line_length = 1  # 線の長さ [m]
        
        use_formation_in_narrow_areas = True

        for i in indices:
            
            if not use_formation_in_narrow_areas and vwa_star_planner.segment_colors[i] == 'blue':
                continue
            
            # 接線ベクトル
            tx, ty = dx_dt[i], dy_dt[i]
            
            # 正規化（長さ1にする）
            norm = np.hypot(tx, ty)
            if norm < 1e-6: continue # 動きがない点はスキップ
            tx /= norm
            ty /= norm
            
            # 法線ベクトル（接線を90度回転: x -> -y, y -> x）
            if vwa_star_planner.segment_colors[i] == 'red':
                nx, ny = ty, -tx
            else:
                nx, ny = tx, ty
            # nx, ny = -ty, tx
            
            # 中心座標
            cx, cy = path_x[i], path_y[i]
            
            # 線の始点(p1) と 終点(p2) を計算
            # 中心から法線方向に +/- 半分の長さだけ伸ばす
            p1 = (cx - nx * line_length / 2, cy - ny * line_length / 2)
            p2 = (cx + nx * line_length / 2, cy + ny * line_length / 2)
            
            
            perp_lines.append([p1, p2])
            
            tips_x.extend([p1[0], p2[0]])
            tips_y.extend([p1[1], p2[1]])

        # 5. LineCollection で一括描画
        # colors='skyblue' など好きな色に設定
        lc_perp = LineCollection(perp_lines, colors='skyblue', linewidths=2, zorder=10)
        plt.gca().add_collection(lc_perp)
        plt.scatter(tips_x, tips_y, color='skyblue', s=30, zorder=11, edgecolors='steelblue', linewidths=0.5)

        # (必要なら) 中心に点も打つ
        # plt.scatter(path_x[::step], path_y[::step], color='skyblue', s=10, zorder=11)
        
        
        plt.legend(handles=legend_elements, loc='best', ncol=2)
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title("VWA* path")
        plt.show()