import numpy as np
import heapq
import math
import pygame
from scipy import ndimage

class VWAstarConfig:
    def __init__(self):
        self.Euclidean = True
        self.OccupyThresh = 128
        # インフレート半径（メートル）
        # [広い形態, 狭い形態]
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
        self.pixels_per_meter = pixels_per_meter # 追加: 変換用係数
        self.map_width = 0
        self.map_height = 0
        self.label_maps = []

    def update_map(self, obstacle_mask):
        self.map_height, self.map_width = obstacle_mask.shape
        self.label_maps = []
        
        # メートル設定をピクセルに変換して処理
        for r_m in self.config.InflateRadiusListM:
            r_px = r_m * self.pixels_per_meter
            
            if r_px > 0:
                structure = ndimage.generate_binary_structure(2, 1)
                # iterationsは整数である必要があるためintにキャスト
                dilated = ndimage.binary_dilation(obstacle_mask, structure, iterations=int(r_px))
                self.label_maps.append(dilated)
            else:
                self.label_maps.append(obstacle_mask.copy())

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
                
                # 参照するリスト名を InflateRadiusListM に合わせる（長さ取得用）
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

        return self._reconstruct_path(final_node)

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
        self.robot_spacing_m = 2.0

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
            self.current_path_nodes = path_nodes[::3] 
            self.current_path_points = [n.point for n in self.current_path_nodes]
        else:
            self.current_path_nodes = []
            self.current_path_points = []
            print("Path planning failed or empty.")

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

        target_dist_px = self.lookahead_dist_m * self.sim.env.PIXELS_PER_METER
        
        current_dist = 0.0
        ref_point = np.array(self.current_path_points[-1])
        ref_fmt_idx = self.current_path_nodes[-1].formation_idx
        tangent = np.array([1.0, 0.0])

        for i in range(min_idx, len(self.current_path_points) - 1):
            p1 = np.array(self.current_path_points[i])
            p2 = np.array(self.current_path_points[i+1])
            seg_len = np.linalg.norm(p2 - p1)
            
            if current_dist + seg_len >= target_dist_px:
                ratio = (target_dist_px - current_dist) / seg_len
                ref_point = p1 + (p2 - p1) * ratio
                ref_fmt_idx = self.current_path_nodes[i].formation_idx
                
                vec = p2 - p1
                norm = np.linalg.norm(vec)
                tangent = vec / norm if norm > 0 else np.array([1.0, 0.0])
                break
            
            current_dist += seg_len
        else:
            if len(self.current_path_points) > 1:
                vec = ref_point - np.array(self.current_path_points[-2])
                norm = np.linalg.norm(vec)
                tangent = vec / norm if norm > 0 else np.array([1.0, 0.0])

        normal = np.array([-tangent[1], tangent[0]])

        spacing_factor = 1.0
        current_spacing_px = (self.robot_spacing_m * self.sim.env.PIXELS_PER_METER) * spacing_factor

        num_robots = len(robots)
        offsets = (np.arange(num_robots) - (num_robots - 1) / 2) * current_spacing_px

        for i, robot in enumerate(robots):
            offset_vec = normal * offsets[i]
            target_px = ref_point + offset_vec
            self.sim.set_robot_target(i, (target_px[0], target_px[1]), in_meters=False)

    def draw(self, surface):
        if len(self.current_path_points) > 1:
            pygame.draw.lines(surface, (255, 0, 255), False, self.current_path_points, 3)
        
        if self.team_goal:
            pygame.draw.circle(surface, (255, 0, 0), (int(self.team_goal[0]), int(self.team_goal[1])), 8)