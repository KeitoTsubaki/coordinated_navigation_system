import yaml
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
            dist_m = dist_cells * self.res

            # self.expansion_radius 以内を全部障害物にする
            inflated_obstacle = dist_m <= expansion_radius

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
                return

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
    
    def show(self):
        if self.path is None:
            print("経路が見つかりませんでした")
        else:
            print("path length (cells):", len(self.path))

            # 経路をワールド座標配列に変換（プロット用）
            xs = []
            ys = []
            segment_colors = []
            for r, c, formation_idx in self.path:
                xs.append(self.grid_to_world(r, c)[0])
                ys.append(self.grid_to_world(r, c)[1])
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




if __name__ == "__main__":
    vwa_star_planner = VWAStarPlanner()
    vwa_star_planner.vwastar(allow_diagonal=True)
    vwa_star_planner.show()