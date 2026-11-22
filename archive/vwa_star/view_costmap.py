import yaml
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ==== ここを書き換える ====
yaml_path = "grid_maps/room1_map/room1_map.yaml"   # あなたの yaml ファイル名
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
occ_grid[occ_prob >= occ_thresh] = 1
occ_grid[occ_prob <= free_thresh] = 0

print("grid shape:", occ_grid.shape)

# マップのサイズ（メートル）を計算
height, width = occ_grid.shape
res = resolution
origin_x, origin_y, origin_yaw = origin

# extent で座標軸を [m] にできる
extent = [
    origin_x,               # x_min
    origin_x + width * res, # x_max
    origin_y,               # y_min
    origin_y + height * res # y_max
]

plt.figure(figsize=(6, 6))

# 表示用に unknown=-1 を 0 に潰して表示
display_map = occ_grid.copy()
display_map[display_map < 0] = 0

im = plt.imshow(
    display_map,
    origin="lower",     # map の y+ を上に
    extent=extent,      # メートル座標
    cmap="gray_r",
    vmin=0,
    vmax=1
)

plt.colorbar(im, label="Occupancy (0 free, 100 occupied)")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Room1 Map (from YAML)")
plt.tight_layout()
plt.show()
