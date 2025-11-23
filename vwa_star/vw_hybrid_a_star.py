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
import yaml
from pathlib import Path

from PIL import Image
from scipy.ndimage import binary_dilation, distance_transform_edt
import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

class VWHybridAStarPlanner:
    def __init__(self):
        map_yaml_path = "grid_maps/room1_map/room1_map.yaml"
        map_yaml_path = "grid_maps/room2_map/room2_map.yaml"
    
        vw_hybrid_a_star_config_path = "vw_hybrid_a_star_config.yaml"

        self.start_pos = None
        self.goal_pos = None
        self.expansion_radius_dict = {}
        self.costmap = None
        self.disp_map = None
        self.formation_priorities = []
        self.expanded_costmap_dict = {}
        self.start_rc = None
        self.goal_rc = None
        self.formation_cost = {}
        self.formation_colors = []
        # self.path = []

        self.origin_x = None
        self.origin_y = None
        self.origin_yaw = None
        self.height = None
        self.width = None
        self.res = None

        self.robot_radius = None
        self.linear_speed = None
        self.max_steer = None
        self.dt = None
        self.num_heading_bins = None
        

        self.load_config(vw_hybrid_a_star_config_path)
        self.load_map(map_yaml_path)
        
        # (Forward Only)
        moves = []
        steers = [-self.max_steer, -self.max_steer*0.5, 0, self.max_steer*0.5, self.max_steer]
        for s in steers:
            moves.append((self.linear_speed, s)) 
        self.motion_primitives = moves

        # 衝突判定用：半径[セル]
        self.collision_radius_cells = max(1, int(math.ceil(self.robot_radius / self.res)))

    def load_config(self, config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.start_pos = tuple(config["start_pos"])  # (x, y, theta) [m]
        self.goal_pos  = tuple(config["goal_pos"])  # (x, y) [m]
        self.expansion_radius_dict = config["expansion_radius_dict"]  # m
        self.formation_priorities = config["formation_priorities"]
        self.formation_cost = config["formation_cost"]
        self.formation_colors = config["formation_colors"]
        self.robot_radius = config.get("robot_radius", 0.4)      # [m]  ロボットの半径
        self.linear_speed = config.get("linear_speed", 0.6)          # [m/s]
        self.max_steer = config.get("max_steer", 0.8)             # [rad/s] 最大角速度
        self.dt = config.get("dt", 0.2)                    # [s]
        self.num_heading_bins = config.get("num_heading_bins", 72)      # 離散化する方位の数

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
        i = 1
        for formation, expansion_radius in self.expansion_radius_dict.items():
            inflate_cells = int(expansion_radius / self.res)
            print(f"inflate cells: {inflate_cells} for formation: {formation}")

            obstacle = (self.costmap == 100)
            dist_cells = distance_transform_edt(~obstacle)

            # メートルに変換
            dist_m = dist_cells * self.res

            # self.expansion_radius 以内を全部障害物にする
            inflated_obstacle = dist_m <= expansion_radius

            expanded_costmap = np.zeros_like(self.costmap, dtype=np.int16)
            print(100 - 100 * i / (len(self.formation_priorities) + 1))
            expanded_costmap[inflated_obstacle] = 100 - 100 * i / (len(self.formation_priorities) + 1)
            if self.disp_map is None:
                self.disp_map = expanded_costmap.copy()
            else:
                self.disp_map += expanded_costmap
            self.expanded_costmap_dict[formation] = expanded_costmap
            i += 1
            plt.figure(figsize=(6,6))
            extent = [
                self.origin_x,               # x_min
                self.origin_x + self.width * self.res, # x_max
                self.origin_y,               # y_min
                self.origin_y + self.height * self.res # y_max
            ]
            plt.imshow(expanded_costmap, origin="lower", extent=extent, cmap="jet", alpha=0.5)
        plt.show()
        self.disp_map += self.costmap

        # self.start_rc = self.world_to_grid(self.start_pos[0], self.start_pos[1]) + (0,)
        # self.goal_rc  = self.world_to_grid(self.goal_pos[0],  self.goal_pos[1])
        # print("start grid:", self.start_rc, "goal grid:", self.goal_rc)

    
    def world_to_grid(self, x, y):
        """
        x, y: [m] （地図座標系）
        self.origin_x, self.origin_y: YAML の origin (左下) [m]
        self.res: [m/cell]
        return: (row, col) = (y_index, x_index)
        """
        # because x and y are continuous values
        gx = int(np.round((x - self.origin_x) / self.res))
        gy = int(np.round((y - self.origin_y) / self.res))
        return gx, gy


    def grid_to_world(self, gx, gy):
        """
        グリッドのセル中心座標を [m] に変換
        """
        x = self.origin_x + gx * self.res
        y = self.origin_y + gy * self.res
        return x, y
    
    def heading_to_bin(self, theta):
        theta = (theta + 2 * math.pi) % (2 * math.pi)
        bin_size = 2 * math.pi / self.num_heading_bins
        return int(theta / bin_size)

    def is_in_bounds(self, gx, gy):
        # h, w = costmap.shape
        return 0 <= gx < self.width and 0 <= gy < self.height
    
    def is_collision(self, x, y, costmap):
        gx, gy = self.world_to_grid(x, y)
        if not self.is_in_bounds(gx, gy):
            return True
        if costmap[gy, gx] > 0:
            return True
        return False

    def calc_holonomic_heuristic(self, goal_gx, goal_gy, formations=None):
        # Heuristic Map from all pixels to the goal within no obstacle map
        if formations is None:
            print("Formation is None")
            return None
        h_map_dict = {}
        motions = [
            (1, 0, 1.0), (0, 1, 1.0), (-1, 0, 1.0), (0, -1, 1.0),
            (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)
        ]
        for id, formation in enumerate(formations):
            if formation not in self.expanded_costmap_dict:
                print(f"Formation {formation} not in expanded_costmap_dict")
                return None
            
            h_map = np.full((self.height, self.width), np.inf)
            h_map[goal_gy, goal_gx] = 0.0
            queue = [(0.0, goal_gx, goal_gy)]
            while queue:
                cost, cx, cy = heapq.heappop(queue)
                if cost > h_map[cy, cx]:
                    continue
                for dx, dy, w in motions:
                    nx, ny = cx + dx, cy + dy
                    if not self.is_in_bounds(nx, ny): continue

                    extra_cost = 0.0
                    if self.expanded_costmap_dict[formation][ny, nx] > 0:
                        if self.expanded_costmap_dict[formations[(id+1) % len(formations)]][ny, nx] > 0:
                            continue
                        # else: extra_cost = 1.0
                    # if self.is_collision(nx, ny, self.expanded_costmap_dict[formation]): continue
                    ncost = cost + w * self.res + extra_cost
                    if ncost < h_map[ny, nx]:
                        h_map[ny, nx] = ncost
                        heapq.heappush(queue, (ncost, nx, ny))
            
            h_map[self.expanded_costmap_dict[formation] > 0] = np.inf
            
            h_map_dict[formation] = h_map
            plt.figure(figsize=(6,6))
            extent = [
                self.origin_x,               # x_min
                self.origin_x + self.width * self.res, # x_max
                self.origin_y,               # y_min
                self.origin_y + self.height * self.res # y_max
            ]
            plt.imshow(h_map, origin="lower", extent=extent, cmap="jet", alpha=0.5)
        plt.show()
        
        return h_map_dict

    # A* アルゴリズム本体
    def vw_hybrid_a_star(self, start_world, goal_world, max_iter=100000):
        sx, sy, stheta = start_world
        goal_x, goal_y = goal_world

        start_gx, start_gy = self.world_to_grid(sx, sy)
        goal_gx, goal_gy = self.world_to_grid(goal_x, goal_y)

        print("Start grid:", (start_gx, start_gy))
        print("Goal grid:", (goal_gx, goal_gy))

        if not self.is_in_bounds(start_gx, start_gy) or \
            not self.is_in_bounds(goal_gx, goal_gy):
            print("Start or goal is out of bounds.")
            return None
        if self.is_collision(sx, sy, self.costmap) or \
            self.is_collision(goal_x, goal_y, self.costmap):
            print("Start or goal is in collision.")
            return None

        print("Calculating Heuristic Map...")
        self.h_map_dict = self.calc_holonomic_heuristic(goal_gx, goal_gy, self.formation_priorities)
        if self.h_map_dict[self.formation_priorities[0]][start_gy, start_gx] == np.inf:
            print("Error: Goal unreachable.")
            return None

        print("Starting VW-Hybrid-A*...")
        # 優先度付きキュー (f, g, id, (x, y, theta, formation_idx))
        open_heap = []
        start_formation_idx = 0
        start_state = (sx, sy, stheta, start_formation_idx)  # formation_idx = 0
        start_id = 0
        # print(f"first h value: {self.h_map_dict[self.formation_priorities[start_formation_idx]][start_gy, start_gx]}")
        initial_f = self.h_map_dict[self.formation_priorities[start_formation_idx]][start_gy, start_gx]
        heapq.heappush(open_heap, (initial_f, 0.0, start_id, start_state))

        came_from = {start_id: None}
        state_data = {start_id: start_state}
        visited = np.full((self.height, self.width, self.num_heading_bins, len(self.formation_priorities)), np.inf)
        visited[start_gy, start_gx, self.heading_to_bin(stheta), 0] = 0.0

        node_id_counter = 1
        goal_found_id = None
        dist_tolerance = 1.0  # [m]
        
        for _ in range(max_iter):
            if not open_heap: break

            f, g, cid, (cx, cy, ctheta, _formation_idx) = heapq.heappop(open_heap)    # pick up lowest f score, which is a heapq characteristic

            # print(f"dist to goal: {math.hypot(cx - goal_x, cy - goal_y):.2f} m, g: {g:.2f}, f: {f:.2f}")
            # Goal check
            if math.hypot(cx - goal_x, cy - goal_y) <= dist_tolerance:
                goal_found_id = cid
                print("Goal reached!")
                break
                
            # print(g)


            # for formation_idx, formation in enumerate(self.formation_priorities):
            formation_idx = 0
            formation = self.formation_priorities[formation_idx]
            for v, omega in self.motion_primitives:
                # Diff Drive Kinematics
                nx = cx + v * math.cos(ctheta) * self.dt
                ny = cy + v * math.sin(ctheta) * self.dt
                ntheta = (ctheta + omega * self.dt + math.pi) % (2 * math.pi) - math.pi

                
                if self.is_collision(nx, ny, self.expanded_costmap_dict[formation]): continue
                n_gx, n_gy = self.world_to_grid(nx, ny)
                if not self.is_in_bounds(n_gx, n_gy): continue

                # When the robot turns, the cost slightly increases
                step_cost = v * self.dt * (1.1 if abs(omega) > 0.1 else 1.0)
                ng = g + step_cost + self.formation_cost[formation]
                # print(f"ng: {ng:.2f}")
                # print(self.formation_cost[formation])
                hbin = self.heading_to_bin(ntheta)


                if ng < visited[n_gy, n_gx, hbin, formation_idx]:
                    visited[n_gy, n_gx, hbin, formation_idx] = ng
                    nid = node_id_counter
                    node_id_counter += 1
                    state_data[nid] = (nx, ny, ntheta, formation_idx)
                    came_from[nid] = cid
                    h_val = max(self.h_map_dict[formation][n_gy, n_gx], math.hypot(nx - goal_x, ny - goal_y)) # heuristic
                    # print(f"h_val: {h_val:.2f}")
                    nf = ng + 1.0 * h_val
                    heapq.heappush(open_heap, (nf, ng, nid, (nx, ny, ntheta, formation_idx)))

        if goal_found_id is None:
            print("Failed to find a path.")
            return None

        path = []
        curr = goal_found_id
        while curr is not None:
            path.append(state_data[curr])
            curr = came_from[curr]
        path.reverse()
        return path
    
    def show(self, path=None):
        if path is not None:
            print("path length (cells):", len(path))
            # 経路をワールド座標配列に変換（プロット用）
            xs = []
            ys = []
            segment_colors = []
            for x, y, theta, formation_idx in path:
                xs.append(x)
                ys.append(y)
                segment_colors.append(self.formation_colors[formation_idx])
            # path_xy = [self.grid_to_world(r, c) for (r, c, formation_idx) in self.path]
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
    


if __name__ == "__main__":
    vw_hybrid_a_star_planner = VWHybridAStarPlanner()
    start = vw_hybrid_a_star_planner.start_pos
    goal = vw_hybrid_a_star_planner.goal_pos
    print("Start:", start)
    print("Goal:", goal)
    path = vw_hybrid_a_star_planner.vw_hybrid_a_star(start, goal)
    vw_hybrid_a_star_planner.show(path)