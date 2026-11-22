import yaml
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, distance_transform_edt

# ==== ここを書き換える ====
yaml_path = "grid_maps/room1_map/room1_map.yaml"   # あなたの yaml ファイル名

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

# extent で座標軸を [m] にできる
extent = [
    origin_x,               # x_min
    origin_x + width * res, # x_max
    origin_y,               # y_min
    origin_y + height * res # y_max
]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# 1枚目: 元の障害物
display_orig = occ_grid.copy()
display_orig[display_orig < 0] = 0  # unknown を free として表示

im0 = axes[0].imshow(
    display_orig,
    origin="lower",
    extent=extent,
    cmap="gray_r",  # 0=白, 1=黒
    vmin=0,
    vmax=100,
)
axes[0].set_title("Original occupancy")
axes[0].set_xlabel("x [m]")
axes[0].set_ylabel("y [m]")

# 2枚目: 膨張後
display_infl = inflated_grid + occ_grid  # 0/1 だけ
im1 = axes[1].imshow(
    display_infl,
    origin="lower",
    extent=extent,
    cmap="gray_r",
    vmin=0,
    vmax=100,
)
axes[1].set_title(f"Inflated (radius={inflation_radius} m)")
axes[1].set_xlabel("x [m]")
axes[1].set_ylabel("y [m]")

# fig.colorbar(im1, ax=axes.ravel().tolist(), label="Occupancy (0 free, 1 obstacle)")
plt.tight_layout()
plt.show()