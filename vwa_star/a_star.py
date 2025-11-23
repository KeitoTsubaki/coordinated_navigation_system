import yaml
from pathlib import Path

from PIL import Image
from scipy.ndimage import binary_dilation, distance_transform_edt
import heapq
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. A* 本体
# --------------------------------------------------
def astar(grid, start, goal, allow_diagonal=True):
    """
    grid: 2D np.ndarray
          0 = free, それ以外 = obstacle として扱う
    start: (sy, sx)  グリッドの行・列
    goal:  (gy, gx)
    return: path のリスト [(y0,x0), (y1,x1), ...] or None
    """
    h, w = grid.shape

    def in_bounds(y, x):
        return 0 <= y < h and 0 <= x < w

    def is_free(y, x):
        return grid[y, x] == 0

    # 近傍（4 or 8 方向）
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
        (y1, x1), (y2, x2) = a, b
        return np.hypot(y2 - y1, x2 - x1)

    # 優先度付きキュー (f, g, (y,x))
    open_heap = []
    heapq.heappush(open_heap, (0.0, 0.0, start))

    came_from = {}  # (y,x) -> (py,px)
    g_score = {start: 0.0}

    closed = set()

    while open_heap:
        f, g, current = heapq.heappop(open_heap)
        cy, cx = current

        if current in closed:
            continue
        closed.add(current)

        # ゴールに着いたら復元
        if current == goal:
            path = [current]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
            path.reverse()
            return path

        for dy, dx, cost in neighbors:
            ny, nx = cy + dy, cx + dx
            if not in_bounds(ny, nx):
                continue
            if not is_free(ny, nx):
                continue

            tentative_g = g + cost
            neighbor = (ny, nx)

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_heap, (f_score, tentative_g, neighbor))

    # 経路なし
    return None


# --------------------------------------------------
# 2. ワールド座標 <-> グリッド座標 変換
# --------------------------------------------------
def world_to_grid(x, y, origin_x, origin_y, resolution):
    """
    x, y: [m] （地図座標系）
    origin_x, origin_y: YAML の origin (左下) [m]
    resolution: [m/cell]
    return: (row, col) = (y_index, x_index)
    """
    col = int((x - origin_x) / resolution)
    row = int((y - origin_y) / resolution)
    return row, col


def grid_to_world(row, col, origin_x, origin_y, resolution):
    """
    グリッドのセル中心座標を [m] に変換
    """
    x = origin_x + (col + 0.5) * resolution
    y = origin_y + (row + 0.5) * resolution
    return x, y


# --------------------------------------------------
# 3. 既に作ってある inflated_grid を使って経路計算 & 表示
# --------------------------------------------------
# 前のコードで作ったものを想定:
# - inflated_grid: 0 = free, 1 = obstacle
# - origin_x, origin_y, res = resolution
# - extent = [origin_x, origin_x + width*res, origin_y, origin_y + height*res]

# 例: スタートとゴールをワールド座標で指定
start_world = (5.0, 5.0)  # (x, y) [m]
goal_world  = ( 7.0,  -2.0)  # (x, y) [m]



# ==== ここを書き換える ====
yaml_path = "grid_maps/room1_map/room1_map.yaml"
yaml_path = "grid_maps/room2_map/room2_map.yaml"

inflation_radius = 1  # m
# =======================

# YAML を読む
with open(yaml_path, "r") as f:
    info = yaml.safe_load(f)

image_file = info["image"]
resolution = info["resolution"]      # 0.05
origin = info["origin"]              # [-15.0, -15.0, 0.0]
negate = info.get("negate", 0)
occ_thresh = info.get("occupied_thresh", 0.65)
free_thresh = info.get("free_thresh", 0.196)

print("image:", image_file)
print("resolution:", resolution)
print("origin:", origin)

# 画像を読む（グレースケール）
image_path = Path(yaml_path).parent / image_file
img = Image.open(image_path).convert("L")  # 'L' = 8bit grayscale
img_arr = np.array(img, dtype=np.uint8)

# ROS map_server と同じルールで占有確率を計算
# negate=0:  p = (255 - gray) / 255
# negate=1:  p = gray / 255
if negate == 0:
    occ_prob = (255 - img_arr) / 255.0
else:
    occ_prob = img_arr / 255.0

# OccupancyGrid 風の値に変換
# -1: unknown, 0: free, 100: occupied
occ_grid = np.full(img_arr.shape, -1, dtype=np.int16)
occ_grid[occ_prob >= occ_thresh] = 100
occ_grid[occ_prob <= free_thresh] = 0

print("grid shape:", occ_grid.shape)

# マップのサイズ（メートル）を計算
height, width = occ_grid.shape
res = resolution
origin_x, origin_y, origin_yaw = origin

inflate_cells = int(inflation_radius / resolution)
print("inflate cells:", inflate_cells)

obstacle = (occ_grid == 100)

# # 正方形カーネルで膨張（inflate_cells セル分）
# kernel_size = 2 * inflate_cells + 1
# structure = np.ones((kernel_size, kernel_size), dtype=bool)

# inflated_obstacle = binary_dilation(obstacle, structure=structure)

# # 新しいコストマップを作成（ここでは 0 or 100 のシンプル版）
# inflated_grid = np.zeros_like(occ_grid, dtype=np.int16)
# inflated_grid[inflated_obstacle] = 50

dist_cells = distance_transform_edt(~obstacle)

# メートルに変換
dist_m = dist_cells * resolution

# inflation_radius 以内を全部障害物にする
inflated_obstacle = dist_m <= inflation_radius

inflated_grid = np.zeros_like(occ_grid, dtype=np.int16)
inflated_grid[inflated_obstacle] = 50

start_rc = world_to_grid(start_world[0], start_world[1], origin_x, origin_y, res)
goal_rc  = world_to_grid(goal_world[0],  goal_world[1],  origin_x, origin_y, res)
print("start grid:", start_rc, "goal grid:", goal_rc)

path_rc = astar(inflated_grid, start_rc, goal_rc, allow_diagonal=True)

display_infl = inflated_grid + occ_grid

# extent で座標軸を [m] にできる
extent = [
    origin_x,               # x_min
    origin_x + width * res, # x_max
    origin_y,               # y_min
    origin_y + height * res # y_max
]

if path_rc is None:
    print("経路が見つかりませんでした")
else:
    print("path length (cells):", len(path_rc))

    # 経路をワールド座標配列に変換（プロット用）
    path_xy = [grid_to_world(r, c, origin_x, origin_y, res) for (r, c) in path_rc]
    xs = [p[0] for p in path_xy]
    ys = [p[1] for p in path_xy]

    # 可視化: inflated_grid の上にパスを重ねる
    plt.figure(figsize=(6, 6))
    plt.imshow(
        display_infl,
        origin="lower",
        extent=extent,
        cmap="gray_r",
        vmin=0,
        vmax=100,
    )
    plt.plot(xs, ys, "-r", linewidth=2, label="A* path")
    plt.scatter([start_world[0]], [start_world[1]], c="b", s=40, label="start")
    plt.scatter([goal_world[0]],  [goal_world[1]],  c="g", s=40, label="goal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("A* path on inflated grid")
    plt.legend()
    plt.tight_layout()
    plt.show()
