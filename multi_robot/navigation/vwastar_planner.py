import numpy as np
import heapq
import math
import pygame
from scipy import ndimage

# =============================================================================
# Helper Function: Bresenham's Line Algorithm
# =============================================================================
def bresenham_points(start, end):
    x1, y1 = int(start[0]), int(start[1])
    x2, y2 = int(end[0]), int(end[1])
    
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    return points

class VWAstarConfig:
    def __init__(self):
        self.Euclidean = True
        self.OccupyThresh = 128
        # インフレート半径（メートル） [広い形態, 狭い形態]
        self.InflateRadiusListM = [0.8, 0.4] 
        self.FormationCosts = [0.0, 10.0]

class Node:
    def __init__(self, point, formation_idx, g=0.0, h=0.0, parent=None):
        self.point = point
        self.formation_idx = formation_idx
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f

class VWAstarPlanner:
    def __init__(self, config=None, pixels_per_meter=10.0):
        if config is None:
            config = VWAstarConfig()
        self.config = config
        self.pixels_per_meter = pixels_per_meter
        self.map_width = 0
        self.map_height = 0
        self.label_maps = []

    def update_map(self, obstacle_mask):
        """
        距離変換を用いて障害物を円形に膨張させる
        """
        self.map_height, self.map_width = obstacle_mask.shape
        self.label_maps = []
        
        # 障害物が全くない場合は処理をスキップして無限遠とする
        if np.all(~obstacle_mask): # 全てFree(False)の場合
             dist_to_obstacle = np.full(obstacle_mask.shape, np.inf)
        else:
             # distance_transform_edt は「0(False)の画素までの距離」を計算する
             # obstacle_mask は True=障害物 なので、反転(~)させて「障害物(False)までの距離」を計算する
             dist_to_obstacle = ndimage.distance_transform_edt(~obstacle_mask)
        
        for r_m in self.config.InflateRadiusListM:
            r_px = r_m * self.pixels_per_meter
            if r_px > 0:
                # 障害物までの距離が半径以下の場所を「障害物」とする（円形膨張）
                dilated = dist_to_obstacle <= r_px
                self.label_maps.append(dilated)
            else:
                self.label_maps.append(obstacle_mask.copy())

    def draw_inflation(self, surface):
        """
        Calculated label_maps (inflated obstacles) are drawn onto the given surface.
        """
        if not self.label_maps:
            return

        # 色定義: (R, G, B, Alpha) - 半透明
        colors = [
            (0, 255, 255, 30),  # 0: Wide (Cyan)
            (255, 0, 255, 30),  # 1: Narrow (Magenta)
        ]
        
        w, h = surface.get_size()
        inflation_surf = pygame.Surface((w, h), pygame.SRCALPHA)
        
        for i, binary_map in enumerate(self.label_maps):
            if i >= len(colors): break
            
            c = colors[i]
            # binary_map は (H, W) なので転置して (W, H)
            mask = binary_map.T > 0
            
            if mask.shape != (w, h): continue

            surf_rgb = pygame.surfarray.pixels3d(inflation_surf)
            surf_alpha = pygame.surfarray.pixels_alpha(inflation_surf)
            
            surf_rgb[mask] = (c[0], c[1], c[2])
            surf_alpha[mask] = c[3]
            
            del surf_rgb
            del surf_alpha
            
        surface.blit(inflation_surf, (0, 0))

    def check_line_collision(self, p1, p2, formation_idx):
        points = bresenham_points(p1, p2)
        target_map = self.label_maps[formation_idx]
        for x, y in points:
            if not (0 <= x < self.map_width and 0 <= y < self.map_height):
                return True
            if target_map[y, x]: 
                return True
        return False

    def smooth_path(self, nodes):
        """
        経路の不要なカーブを除去し、一定間隔で再サンプリングする。
        """
        if not nodes:
            return []

        # 1. ウェイポイント（直線の継ぎ目）の抽出
        waypoints = [nodes[0]]
        current_idx = 0
        total_nodes = len(nodes)

        while current_idx < total_nodes - 1:
            valid_idx = current_idx + 1
            current_node = nodes[current_idx]
            current_fmt = current_node.formation_idx
            current_pt = current_node.point

            for i in range(current_idx + 1, total_nodes):
                target_node = nodes[i]
                
                # フォーメーション変化時はセグメントを切る
                if target_node.formation_idx != current_fmt:
                    valid_idx = max(i - 1, current_idx + 1)
                    break

                # 衝突判定
                if self.check_line_collision(current_pt, target_node.point, current_fmt):
                    valid_idx = max(i - 1, current_idx + 1)
                    break
                else:
                    valid_idx = i
            
            waypoints.append(nodes[valid_idx])
            current_idx = valid_idx

        # 2. 密なパスの生成（全ピクセルを列挙）
        dense_path = []
        for i in range(len(waypoints) - 1):
            start_node = waypoints[i]
            end_node = waypoints[i+1]
            
            segment_points = bresenham_points(start_node.point, end_node.point)
            
            # セグメントの属性決定（始点と終点のうち、より狭い方を採用）
            # これにより狭い場所への進入路は「狭い」扱いになる
            fmt_idx = max(start_node.formation_idx, end_node.formation_idx)
            
            if i > 0:
                segment_points.pop(0) # 重複削除
            
            for pt in segment_points:
                dense_path.append(Node(pt, fmt_idx))

        # 3. 一定間隔での再サンプリング (Resampling)
        if not dense_path:
            return []

        resampled_nodes = []
        resampled_nodes.append(dense_path[0])
        
        spacing_px = 5.0 # 再サンプリング間隔（ピクセル）。小さくすると滑らかになる。
        accum_dist = 0.0
        prev_pt = np.array(dense_path[0].point)
        
        for i in range(1, len(dense_path)):
            curr_pt = np.array(dense_path[i].point)
            dist = np.linalg.norm(curr_pt - prev_pt)
            accum_dist += dist
            prev_pt = curr_pt
            
            # 指定距離進むごとにノードを採用
            if accum_dist >= spacing_px:
                resampled_nodes.append(dense_path[i])
                accum_dist = 0.0
        
        # 終点を確実に追加
        if resampled_nodes[-1].point != dense_path[-1].point:
            resampled_nodes.append(dense_path[-1])

        return resampled_nodes

    def plan(self, start_point, target_point):
        if not (0 <= start_point[0] < self.map_width and 0 <= start_point[1] < self.map_height):
            return []
        
        start_node = Node(start_point, 0, g=0.0, h=0.0)
        open_list = []
        heapq.heappush(open_list, start_node)
        
        closed_set = {}
        
        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),           (0, 1),
                     (1, -1),  (1, 0),  (1, 1)]

        final_node = None
        W, H = self.map_width, self.map_height

        while open_list:
            current_node = heapq.heappop(open_list)
            cx, cy = current_node.point
            c_fmt = current_node.formation_idx
            
            if (cx, cy, c_fmt) in closed_set and closed_set[(cx, cy, c_fmt)] <= current_node.g:
                continue
            closed_set[(cx, cy, c_fmt)] = current_node.g

            if math.hypot(cx - target_point[0], cy - target_point[1]) < 10.0:
                final_node = current_node
                break

            for dx, dy in neighbors:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < W and 0 <= ny < H):
                    continue

                move_cost = math.hypot(dx, dy) * 10.0
                
                num_formations = len(self.config.InflateRadiusListM)
                for next_fmt in range(num_formations):
                    if self.label_maps[next_fmt][ny, nx]: 
                        continue
                    
                    new_g = current_node.g + move_cost + self.config.FormationCosts[next_fmt]
                    
                    if self.config.Euclidean:
                        h = math.hypot(nx - target_point[0], ny - target_point[1]) * 10.0
                    else:
                        h = (abs(nx - target_point[0]) + abs(ny - target_point[1])) * 10.0
                    
                    if (nx, ny, next_fmt) in closed_set and closed_set[(nx, ny, next_fmt)] <= new_g:
                        continue
                    
                    new_node = Node((nx, ny), next_fmt, g=new_g, h=h, parent=current_node)
                    heapq.heappush(open_list, new_node)

        raw_path = self._reconstruct_path(final_node)
        smoothed_path = self.smooth_path(raw_path)
        return raw_path, smoothed_path

    def _reconstruct_path(self, node):
        path = []
        curr = node
        while curr:
            path.append(curr)
            curr = curr.parent
        return path[::-1]

class TeamNavigationController:
    def __init__(self, simulator, planner):
        self.sim = simulator
        self.planner = planner
        self.current_path_nodes = [] 
        self.current_path_points = []
        self.team_goal = None
        self.lookahead_dist_m = 1.0 
        self.robot_spacing_m = 1.0 
        self.narrow_area_mode = "line" 

    def set_team_goal(self, goal_pos_pixels):
        self.team_goal = goal_pos_pixels
        self.replan()
    
    def _save_raw_path_image(self, raw_nodes):
        if not self.sim.map_surface or not raw_nodes:
            return
        
        # 描画用のSurfaceを作成（地図のコピー）
        debug_surf = self.sim.map_surface.copy()
        
        # Raw Path を赤線で描画
        if len(raw_nodes) > 1:
            points = [(int(n.point[0]), int(n.point[1])) for n in raw_nodes]
            pygame.draw.lines(debug_surf, (255, 0, 0), False, points, 2)
            
            # 始点（緑）と終点（青）
            start = points[0]
            end = points[-1]
            pygame.draw.circle(debug_surf, (0, 255, 0), start, 5)
            pygame.draw.circle(debug_surf, (0, 0, 255), end, 5)

        # ファイル保存
        try:
            filename = "vwastar_raw_path.png"
            pygame.image.save(debug_surf, filename)
            print(f"[INFO] Saved VWAstar raw path image to: {filename}")
        except Exception as e:
            print(f"[ERROR] Failed to save raw path image: {e}")

    def replan(self):
        if not self.team_goal:
            return
        
        robots = self.sim.robot_states
        if not robots:
            return
        
        cx = sum(r.position[0] for r in robots) / len(robots)
        cy = sum(r.position[1] for r in robots) / len(robots)
        start_pos = (int(cx), int(cy))
        target_pos = (int(self.team_goal[0]), int(self.team_goal[1]))

        mask = self.sim._get_obstacle_mask()
        self.planner.update_map(mask)
        
        print(f"Planning from {start_pos} to {target_pos}...")
        raw_nodes, path_nodes = self.planner.plan(start_pos, target_pos)
        
        if raw_nodes:
            self._save_raw_path_image(raw_nodes)
            print(f"Path found! Length: {len(path_nodes)}")
            self.current_path_nodes = path_nodes 
            self.current_path_points = [n.point for n in self.current_path_nodes]
        else:
            self.current_path_nodes = []
            self.current_path_points = []
            print("Path planning failed or empty.")

    def _project_point_on_segment(self, p, a, b):
        ap = p - a
        ab = b - a
        norm_sq = np.dot(ab, ab)
        if norm_sq == 0:
            return a, 0.0, np.linalg.norm(ap)
        t = np.dot(ap, ab) / norm_sq
        t = max(0.0, min(1.0, t)) # 線分内にクランプ
        proj = a + t * ab
        dist = np.linalg.norm(p - proj)
        return proj, t, dist

    def update(self):
        if not self.current_path_nodes or not self.sim.robot_states:
            return

        robots = self.sim.robot_states
        cx = sum(r.position[0] for r in robots) / len(robots)
        cy = sum(r.position[1] for r in robots) / len(robots)
        center_pos = np.array([cx, cy])

        path_points = np.array(self.current_path_points)
        
        # 1. チーム中心に最も近い線分を探す (平滑化のため)
        dists_to_nodes = np.linalg.norm(path_points - center_pos, axis=1)
        min_node_idx = np.argmin(dists_to_nodes)
        
        search_start = max(0, min_node_idx - 1)
        search_end = min(len(path_points) - 1, min_node_idx + 1)
        
        best_proj = path_points[min_node_idx]
        best_seg_idx = min_node_idx
        min_dist_to_segment = float('inf')
        
        for i in range(search_start, search_end):
            if i >= len(path_points) - 1: break
            p_a = path_points[i]
            p_b = path_points[i+1]
            proj, t, dist = self._project_point_on_segment(center_pos, p_a, p_b)
            if dist < min_dist_to_segment:
                min_dist_to_segment = dist
                best_proj = proj
                best_seg_idx = i

        # 2. 参照点(サブゴール)の計算
        target_dist_px = self.lookahead_dist_m * self.sim.env.PIXELS_PER_METER
        current_lookahead = target_dist_px
        
        ref_point = best_proj
        # デフォルトは現在のセグメントだが、ループ内で必ず更新されることを期待
        ref_fmt_idx = self.current_path_nodes[best_seg_idx].formation_idx
        ref_idx = best_seg_idx
        tangent = np.array([1.0, 0.0])
        
        curr_pt = best_proj
        curr_idx = best_seg_idx
        
        while True:
            if curr_idx >= len(path_points) - 1:
                ref_point = path_points[-1]
                ref_idx = len(path_points) - 1
                ref_fmt_idx = self.current_path_nodes[-1].formation_idx
                if len(path_points) > 1:
                    v = path_points[-1] - path_points[-2]
                    n = np.linalg.norm(v)
                    if n > 0: tangent = v / n
                break

            p_end = path_points[curr_idx + 1]
            seg_vec = p_end - curr_pt
            seg_len = np.linalg.norm(seg_vec)
            
            if current_lookahead <= seg_len:
                if seg_len > 0:
                    dir_vec = seg_vec / seg_len
                    ref_point = curr_pt + dir_vec * current_lookahead
                    tangent = dir_vec
                else:
                    ref_point = curr_pt
                
                # ここで「サブゴールがいる区間」のフォーメーションIDを取得
                ref_idx = curr_idx
                ref_fmt_idx = self.current_path_nodes[curr_idx].formation_idx
                # f1 = self.current_path_nodes[curr_idx].formation_idx
                # f2 = self.current_path_nodes[curr_idx+1].formation_idx
                # ref_fmt_idx = max(f1, f2)
                break
            else:
                current_lookahead -= seg_len
                curr_pt = p_end
                curr_idx += 1

        is_targeting_final_goal = (ref_idx == len(path_points) - 1) or \
                                  (np.linalg.norm(ref_point - path_points[-1]) < 1.0)

        # ---------------------------------------------------------------------
        # 3. フォーメーション展開
        # ref_fmt_idx (サブゴールの区間) に基づいて決定する
        # ---------------------------------------------------------------------
        direction_vec = np.array([0.0, 0.0])
        spacing_factor = 1.0
        path_direction = tangent 

        if ref_fmt_idx == 0:
            # Wide: 垂直展開
            direction_vec = np.array([-tangent[1], tangent[0]]) 
            spacing_factor = 1.0
        else:
            # Narrow
            if self.narrow_area_mode == "line":
                direction_vec = tangent
                spacing_factor = 0.8
            elif self.narrow_area_mode == "skip":
                found_next_wide = False
                for k in range(ref_idx + 1, len(self.current_path_nodes)):
                    if self.current_path_nodes[k].formation_idx == 0:
                        norm = np.linalg.norm(path_points[k+1] - path_points[k])
                        t_next = (path_points[k+1] - path_points[k]) / norm if norm > 0 else np.array([1.0, 0.0])
                        ref_point = path_points[k]
                        direction_vec = np.array([-t_next[1], t_next[0]])
                        path_direction = t_next 
                        found_next_wide = True
                        break
                if not found_next_wide:
                    direction_vec = tangent
                    spacing_factor = 0.8

        target_theta_rad = math.atan2(-path_direction[0], -path_direction[1])
        
        current_spacing_px = (self.robot_spacing_m * self.sim.env.PIXELS_PER_METER) * spacing_factor
        num_robots = len(robots)
        offsets = (np.arange(num_robots) - (num_robots - 1) / 2) * current_spacing_px
        
        robot_indices = list(range(num_robots))
        scores = []
        for i, robot in enumerate(robots):
            r_pos = np.array(robot.position)
            # 現在の位置関係に基づいてソート (近い順/前順)
            score = np.dot(r_pos, direction_vec)
            scores.append(score)
        
        robot_indices = [x for _, x in sorted(zip(scores, robot_indices))]

        path_theta = target_theta_rad

        for rank, robot_idx in enumerate(robot_indices):
            offset_val = offsets[rank]
            offset_vec = direction_vec * offset_val
            target_px = ref_point + offset_vec
            
            robot = self.sim.robot_states[robot_idx]
            
            # 中間角度の計算
            sin_avg = math.sin(path_theta) + math.sin(robot.theta)
            cos_avg = math.cos(path_theta) + math.cos(robot.theta)
            
            if abs(sin_avg) < 1e-6 and abs(cos_avg) < 1e-6:
                goal_theta = path_theta
            else:
                goal_theta = math.atan2(sin_avg, cos_avg)
            
            goal_theta = path_theta

            self.sim.set_robot_target(
                robot_idx, 
                (target_px[0], target_px[1]), 
                in_meters=False,
                target_theta=goal_theta 
            )
            
            self.sim.robot_states[robot_idx].is_final_target = is_targeting_final_goal

    def draw(self, surface):
        if len(self.current_path_nodes) < 2:
            return

        formation_colors = [
            (0, 255, 255),
            (255, 0, 255),
            (255, 255, 0),
        ]

        for i in range(len(self.current_path_nodes) - 1):
            node_start = self.current_path_nodes[i]
            node_end = self.current_path_nodes[i+1]
            
            fmt_idx = node_end.formation_idx
            color = formation_colors[fmt_idx % len(formation_colors)]
            
            start_pos = (int(node_start.point[0]), int(node_start.point[1]))
            end_pos = (int(node_end.point[0]), int(node_end.point[1]))
            
            pygame.draw.line(surface, color, start_pos, end_pos, 3)
        
        if self.team_goal:
            pygame.draw.circle(surface, (255, 0, 0), (int(self.team_goal[0]), int(self.team_goal[1])), 8)