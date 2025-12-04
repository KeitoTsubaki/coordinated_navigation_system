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
        Remove unnecessary curves from the path using line-of-sight checks.
        Replicates SmoothPath from VWAstar.cpp.
        Restrict smoothing to sections with the same formation index.
        """
        if not nodes:
            return []

        waypoints = [nodes[0]]
        current_idx = 0
        total_nodes = len(nodes)

        while current_idx < total_nodes - 1:
            # デフォルトは隣接点（最低でも1歩は進む）
            valid_idx = current_idx + 1
            current_node = nodes[current_idx]
            current_fmt = current_node.formation_idx
            current_pt = current_node.point

            # current_idx + 1 から順に見ていき、直線を引ける限界を探す
            for i in range(current_idx + 1, total_nodes):
                target_node = nodes[i]
                
                # 1. フォーメーションが異なる場合は、そこでセグメントを切る
                if target_node.formation_idx != current_fmt:
                    # 変更点の手前(i-1)までを採用
                    valid_idx = max(i - 1, current_idx + 1)
                    break

                # 2. 衝突判定
                if self.check_line_collision(current_pt, target_node.point, current_fmt):
                    # 衝突した場合、衝突した点(i)は採用せず、一つ手前(i-1)を採用する
                    # (ただし、もしiが隣接点(current_idx+1)だった場合は、
                    #  進まないと無限ループになるため、やむを得ずiを採用して進む)
                    valid_idx = max(i - 1, current_idx + 1)
                    break
                else:
                    # 衝突しなければ、ここまでは直線化可能として候補を更新
                    valid_idx = i
            
            waypoints.append(nodes[valid_idx])
            current_idx = valid_idx

        # ウェイポイント間を補間
        smoothed_nodes = []
        step = 20 # ピクセル単位の間引き間隔
        
        for i in range(len(waypoints) - 1):
            start_node = waypoints[i]
            end_node = waypoints[i+1]
            
            segment_points = bresenham_points(start_node.point, end_node.point)
            fmt_idx = start_node.formation_idx
            
            if i > 0:
                segment_points.pop(0) # 重複削除
                
            for pt in segment_points:
                smoothed_nodes.append(Node(pt, fmt_idx))

        # 間引き処理
        spaced_nodes = []
        for i in range(0, len(smoothed_nodes), step):
            spaced_nodes.append(smoothed_nodes[i])
        
        if smoothed_nodes and (not spaced_nodes or spaced_nodes[-1].point != smoothed_nodes[-1].point):
            spaced_nodes.append(smoothed_nodes[-1])

        return spaced_nodes

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
        return self.smooth_path(raw_path)

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
        path_nodes = self.planner.plan(start_pos, target_pos)
        
        if path_nodes:
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
        dists = np.linalg.norm(path_points - center_pos, axis=1)
        min_idx = np.argmin(dists)

        # ---------------------------------------------------------------------
        # 1. 最近傍セグメントの探索（前回のロジック維持）
        # ---------------------------------------------------------------------
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

        # ---------------------------------------------------------------------
        # 2. 参照点の計算
        # ---------------------------------------------------------------------
        target_dist_px = self.lookahead_dist_m * self.sim.env.PIXELS_PER_METER
        
        current_lookahead = target_dist_px
        ref_point = best_proj
        ref_fmt_idx = self.current_path_nodes[best_seg_idx].formation_idx
        ref_idx = best_seg_idx
        tangent = np.array([1.0, 0.0])
        
        curr_pt = best_proj
        curr_idx = best_seg_idx
        
        while True:
            if curr_idx >= len(path_points) - 1:
                ref_point = path_points[-1]
                ref_fmt_idx = self.current_path_nodes[-1].formation_idx
                ref_idx = len(path_points) - 1
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
                
                ref_fmt_idx = self.current_path_nodes[curr_idx].formation_idx
                ref_idx = curr_idx
                break
            else:
                current_lookahead -= seg_len
                curr_pt = p_end
                curr_idx += 1

        is_targeting_final_goal = (ref_idx == len(path_points) - 1) or \
                                  (np.linalg.norm(ref_point - path_points[-1]) < 1.0)

        # ---------------------------------------------------------------------
        # 3. サブゴール展開方向と向きの決定
        # ---------------------------------------------------------------------
        direction_vec = np.array([0.0, 0.0])
        spacing_factor = 1.0
        
        # 【追加】サブゴールの向き（接線ベクトル）を保持する変数
        path_direction = tangent 

        if ref_fmt_idx == 0:
            # Wide
            direction_vec = np.array([-tangent[1], tangent[0]]) 
            spacing_factor = 1.0
        else:
            # Narrow
            if self.narrow_area_mode == "line":
                direction_vec = tangent
                spacing_factor = 1.0
            elif self.narrow_area_mode == "skip":
                found_next_wide = False
                for k in range(ref_idx + 1, len(self.current_path_nodes)):
                    if self.current_path_nodes[k].formation_idx == 0:
                        p_curr = np.array(self.current_path_points[k])
                        if k < len(self.current_path_points) - 1:
                            p_next = np.array(self.current_path_points[k+1])
                            vec = p_next - p_curr
                        else:
                            p_prev = np.array(self.current_path_points[k-1])
                            vec = p_curr - p_prev
                        
                        norm = np.linalg.norm(vec)
                        t_next = vec / norm if norm > 0 else np.array([1.0, 0.0])
                        
                        ref_point = p_curr
                        direction_vec = np.array([-t_next[1], t_next[0]])
                        spacing_factor = 1.0
                        
                        # 【修正】スキップ先での進行方向をターゲットの向きとする
                        path_direction = t_next 
                        
                        found_next_wide = True
                        break
                
                if not found_next_wide:
                    direction_vec = tangent
                    spacing_factor = 0.8

        # 【追加】接線ベクトルから角度(rad)を計算
        # Simulatorの座標系: Y軸が下向き、θ=0が上向き(0, -1)
        # math.atan2(-x, -y) で変換可能 (simulator_v5.pyの_infer_target_headingと同様)
        target_theta_rad = math.atan2(-path_direction[0], -path_direction[1])

        # ---------------------------------------------------------------------
        # 4. 割り当てとターゲット設定
        # ---------------------------------------------------------------------
        # 経路の接線方向の角度 (Simulator座標系: 0=上, CW)
        path_theta = math.atan2(-path_direction[0], -path_direction[1])

        current_spacing_px = (self.robot_spacing_m * self.sim.env.PIXELS_PER_METER) * spacing_factor
        num_robots = len(robots)
        offsets = (np.arange(num_robots) - (num_robots - 1) / 2) * current_spacing_px
        
        robot_indices = list(range(num_robots))
        scores = []
        for i, robot in enumerate(robots):
            r_pos = np.array(robot.position)
            score = np.dot(r_pos, direction_vec)
            scores.append(score)
        
        robot_indices = [x for _, x in sorted(zip(scores, robot_indices))]

        for rank, robot_idx in enumerate(robot_indices):
            offset_val = offsets[rank]
            offset_vec = direction_vec * offset_val
            target_px = ref_point + offset_vec
            
            # 【修正】向きを「経路接線」と「現在のロボットの向き」の中間に設定
            robot = self.sim.robot_states[robot_idx]
            robot_theta = robot.theta
            
            # 角度の平均を計算（ベクトル和の角度をとることで、0/360度の境界も正しく扱える）
            sin_avg = math.sin(path_theta) + math.sin(robot_theta)
            cos_avg = math.cos(path_theta) + math.cos(robot_theta)
            
            # 正反対(180度差)でベクトル和がゼロになる場合は、経路方向を優先
            if abs(sin_avg) < 1e-6 and abs(cos_avg) < 1e-6:
                goal_theta = path_theta
            else:
                goal_theta = math.atan2(sin_avg, cos_avg)

            goal_theta = path_theta ###########
            
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