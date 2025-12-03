import os

import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import yaml
import math
import importlib.resources
from collections import deque

# Square用の環境をインポート
from simulation_core import environment_square


class EnvSquare(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, eval_mode=False):
        super().__init__()

        # ===== Initialize and load config files =====
        simulator_config_path = (
            importlib.resources.files("simulation_core")
            .joinpath("config")
            .joinpath("simulator_config_square.yaml")
        )
        ddpg_config_path = os.path.join(
            os.path.dirname(__file__), "config", "ddpg_config.yaml"
        )

        with open(ddpg_config_path, "r", encoding="utf-8") as f:
            self.ddpg_config = yaml.safe_load(f)

        reward_config = self.ddpg_config["reward"]

        # ===== Initialize render mode =====
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # SimCoreSquareEnvironment (四角いロボット) を使用
        self.env = environment_square.SimCoreSquareEnvironment(
            render_mode=self.render_mode,
            config_path=simulator_config_path,
            min_target_clearance_m=reward_config["rho_0"],
        )
        self.env.eval_mode = eval_mode

        # ===== Parameters from Environment =====
        self.V_MAX = self.env.V_MAX
        self.OMEGA_MAX = self.env.OMEGA_MAX
        self.A_MAX = self.env.A_MAX
        self.ALPHA_MAX = self.env.ALPHA_MAX
        self.K_D = self.env.K_D
        self.T_DF = self.env.T_DF
        self.MAX_D = self.env.MAX_D
        self.MAX_THETA = self.env.MAX_THETA
        self.LIDAR_RANGE_M = self.env.LIDAR_RANGE_M
        
        # ===== RL Parameters from DDPG Config =====
        self.N_L = self.ddpg_config["lidar"]["n_l"]
        self.N_E = self.ddpg_config["lidar"]["n_e"]
        self.LIDAR_HISTORY_LEN = self.ddpg_config["lidar"]["lidar_history_len"]
        self.DT = 1 / self.metadata["render_fps"]
        self.MAX_EPISODE_STEPS = self.ddpg_config["max_episode_steps"]
        self.RHO_2 = float(reward_config.get("rho_2", reward_config["rho_1"]))
        self.K_10 = float(reward_config.get("k_10", 0.0))
        self.ANGLE_DIFF_THRESHOLD = float(
            self.ddpg_config.get("angle_diff_threshold", math.radians(10.0))
        )

        # ===== Physical parameters to mimic Gazebo dynamics =====
        self.ROBOT_MASS_KG = 56.5
        self.ROTATIONAL_INERTIA = 8.16
        self.LINEAR_DAMPING = 1.0
        self.ANGULAR_DAMPING = 1.0
        self.ROLLING_RESISTANCE_COEFF = 0.8
        self.RESPONSE_TIME_CONSTANT = 1.0 / 12.0

        # Action Space
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        max_dist = (
            math.sqrt(self.env.SCREEN_W**2 + self.env.SCREEN_H**2)
            / self.env.PIXELS_PER_METER
        )

        # Observation Space
        self.observation_space = spaces.Dict(
            {
                "lidar": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.LIDAR_HISTORY_LEN, self.N_L + self.N_E),
                    dtype=np.float32,
                ),
                "goal": spaces.Box(
                    low=np.array([0.0, -1.0, -1.0], dtype=np.float32),
                    high=np.array([max_dist, 1.0, 1.0], dtype=np.float32),
                    dtype=np.float32,
                ),
                "velocity": spaces.Box(
                    low=np.array([0.0, -1.0], dtype=np.float32),
                    high=np.array([1.0, 1.0], dtype=np.float32),
                    dtype=np.float32,
                ),
                "closest_obstacle": spaces.Box(
                    low=np.array([0.0, -1.0], dtype=np.float32),
                    high=np.array([1.0, 1.0], dtype=np.float32),
                    dtype=np.float32,
                ),
            }
        )

        self.clock = None
        self.lidar_history = deque(maxlen=self.LIDAR_HISTORY_LEN)
        self.is_collision = False
        
        # State variables
        self.previous_distance_to_taret = None
        self.previous_angle_to_target = None
        self.current_distance_to_target = None
        self.current_angle_to_target = None
        self.current_angle_diff_robot_target = None
        self.previous_distance_to_closest_obstacle = None
        self.current_distance_to_closest_obstacle = None
        
        self.current_v = 0.0
        self.current_omega = 0.0
        self.previous_v = 0.0
        self.previous_omega = 0.0

        self._distance_condition_met = False
        self._orientation_condition_met = False

        self.current_step = 0

    def step(self, action):
        v_cmd, omega_cmd = self._action_to_command(action)
        self._update_robot_position(v_cmd, omega_cmd, dt=self.DT)
        self.env.update_sensors()
        self.render()

        self.previous_distance_to_taret = self.current_distance_to_target
        self.previous_angle_to_target = self.current_angle_to_target
        self.previous_distance_to_closest_obstacle = (
            self.current_distance_to_closest_obstacle
        )
        
        self.current_distance_to_target, self.current_angle_to_target = (
            self.env.get_goal_info()
        )
        self.current_distance_to_closest_obstacle = (
            self.env.get_closest_obstacle_info()[0]
        )
        self.current_angle_diff_robot_target = (
            self._compute_robot_target_orientation_diff()
        )

        self.current_step += 1
        
        (
            distance_condition_met,
            orientation_condition_met,
        ) = self._update_target_alignment_flags()
        
        is_target_reached = distance_condition_met and orientation_condition_met
        terminated = is_target_reached or self.is_collision
        truncated = self.current_step >= self.MAX_EPISODE_STEPS
        
        observation = self._get_obs()
        info = {
            "is_target_reached": is_target_reached,
            "distance_condition_met": distance_condition_met,
            "orientation_condition_met": orientation_condition_met,
        }
        reward = self._calculate_reward()
        
        if terminated:
            if is_target_reached:
                print(f"is_target_reached: True")
            else:
                print(f"is_target_reached: False (collision)")
        if truncated:
            print(f"is_target_reached: False (time limit)")

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if options is not None and "map_type" in options:
            map_type = options["map_type"]
        else:
            map_type = self._randomly_seletct_map()
        
        print("map_type: ", map_type)
        self.env.reset(map_type=map_type)
        self.env.update_sensors()

        self.current_step = 0
        self._reset_parameters()

        observation = self._get_obs()
        info = self._get_info()
        self.render()

        return observation, info

    def render(self):
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.render_mode == "human":
            self.env.render_robot_and_lidar()
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.update()

        elif self.render_mode == "rgb_array":
            self.env.render_robot_and_lidar()
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.env.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.env.screen is not None:
            pygame.display.quit()
            pygame.quit()

    def _reset_parameters(self):
        self.env.np_random = self.np_random
        self.is_collision = False

        self.lidar_history.clear()
        initial_lidar_obs = self._process_lidar_data(self.env.sensor_data)
        for _ in range(self.LIDAR_HISTORY_LEN):
            self.lidar_history.append(initial_lidar_obs)

        self.current_v = 0.0
        self.current_omega = 0.0
        self.previous_v = 0.0
        self.previous_omega = 0.0

        initial_dist_to_target, initial_angle_to_target = self.env.get_goal_info()
        self.previous_distance_to_taret = initial_dist_to_target
        self.current_distance_to_target = initial_dist_to_target
        self.previous_angle_to_target = initial_angle_to_target
        self.current_angle_to_target = initial_angle_to_target
        self.previous_distance_to_closest_obstacle = (
            self.env.get_closest_obstacle_info()[0]
        )
        self.current_distance_to_closest_obstacle = (
            self.env.get_closest_obstacle_info()[0]
        )
        self.current_angle_diff_robot_target = (
            self._compute_robot_target_orientation_diff()
        )
        self._distance_condition_met = False
        self._orientation_condition_met = self._is_orientation_condition_met()

    def _calculate_reward(self):
        r_safe = self.env.ROBOT_RADIUS_M
        r_arrival = self.ddpg_config["reward"]["r_arrival"]
        r_collision = self.ddpg_config["reward"]["r_collision"]
        eta_r = self.ddpg_config["reward"]["eta_r"]
        rho_0 = self.ddpg_config["reward"]["rho_0"]
        rho_1 = self.ddpg_config["reward"]["rho_1"]
        k_1 = self.ddpg_config["reward"]["k_1"]
        k_2 = self.ddpg_config["reward"]["k_2"]
        k_3 = self.ddpg_config["reward"]["k_3"]
        k_4 = self.ddpg_config["reward"]["k_4"]
        k_5 = self.ddpg_config["reward"]["k_5"]
        k_6 = self.ddpg_config["reward"]["k_6"]
        k_7 = self.ddpg_config["reward"]["k_7"]
        minor_penalty = self.ddpg_config["reward"]["minor_penalty"]
        k_10 = self.K_10

        r_s = 0.0
        if self._distance_condition_met and self._orientation_condition_met:
            r_s = r_arrival
        elif self.is_collision:
            r_s = r_collision

        r_g_d = (
            k_1
            * (self.previous_distance_to_taret - self.current_distance_to_target)
            / self.MAX_D
        )
        r_g_o1 = (
            k_2
            * (abs(self.previous_angle_to_target) - abs(self.current_angle_to_target))
            / self.MAX_THETA
        )
        r_g_o2 = -k_3 * abs(self.current_angle_to_target) / math.pi
        
        angle_diff_normalized = np.clip(
            (self.current_angle_diff_robot_target or 0.0) / math.pi, -1.0, 1.0
        )
        r_g_o3 = 0.0
        if (
            self.current_distance_to_closest_obstacle is not None
            and self.current_distance_to_closest_obstacle < self.RHO_2
        ):
            r_g_o3 = -k_10 * abs(angle_diff_normalized)
        r_g = r_g_d + r_g_o1 + r_g_o2 + r_g_o3

        r_c_1 = 0.0
        if self.current_distance_to_closest_obstacle < rho_0:
            r_c_1 = (
                -k_4
                * (
                    (1 / self.current_distance_to_closest_obstacle - 1 / rho_0)
                    / (1 / r_safe - 1 / rho_0)
                )
                ** eta_r
            )
        r_c_2 = 0.0
        if self.current_distance_to_closest_obstacle < rho_1:
            r_c_2 = (
                -k_5
                * (
                    self.previous_distance_to_closest_obstacle
                    - self.current_distance_to_closest_obstacle
                )
                / self.MAX_D
            )
        r_c = r_c_1 + r_c_2

        r_v_v = k_6 * (self.current_v / self.V_MAX)
        r_v_omega = -k_7 * (abs(self.current_omega) / self.OMEGA_MAX)
        r_v = r_v_v + r_v_omega

        r_m = minor_penalty

        return r_s + r_g + r_c + r_v + r_m

    def _compute_robot_target_orientation_diff(self):
        angle_diff = self.env.target_theta - self.env.robot_theta
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
        return angle_diff

    def _get_normalized_robot_target_orientation_diff(self):
        angle_diff = self.current_angle_diff_robot_target
        if angle_diff is None:
            angle_diff = self._compute_robot_target_orientation_diff()
        return float(np.clip(angle_diff / math.pi, -1.0, 1.0))

    def _is_orientation_condition_met(self):
        angle_diff = self.current_angle_diff_robot_target
        if angle_diff is None:
            angle_diff = self._compute_robot_target_orientation_diff()
        return abs(angle_diff) <= self.ANGLE_DIFF_THRESHOLD

    def _update_target_alignment_flags(self):
        distance_condition_met = self.env.is_target_reached()
        orientation_condition_met = self._is_orientation_condition_met()
        self._distance_condition_met = distance_condition_met
        self._orientation_condition_met = orientation_condition_met
        return distance_condition_met, orientation_condition_met

    def _action_to_command(self, action):
        v_cmd = (action[0] + 1.0) / 2.0 * self.V_MAX
        omega_cmd = action[1] * self.OMEGA_MAX
        return v_cmd, omega_cmd

    def _update_robot_position(self, v_cmd, omega_cmd, dt):
        """
        慣性（応答遅れ）と加速度制限を考慮して速度を更新する。
        摩擦項を削除し、純粋な一次遅れ系＋レートリミッタとして実装することで
        低速時のスタックを防ぐ。
        """
        prev_v = self.previous_v or 0.0
        prev_omega = self.previous_omega or 0.0

        # --- 1. Acceleration Limit (Rate Limiter) ---
        # 目標速度(v_cmd)に対して、最大加速度(A_MAX)の範囲内で近づける
        
        # 線形速度の更新
        v_diff = v_cmd - prev_v
        max_v_change = self.A_MAX * dt
        v_change = np.clip(v_diff, -max_v_change, max_v_change)
        target_v_limited = prev_v + v_change

        # 角速度の更新
        omega_diff = omega_cmd - prev_omega
        max_omega_change = self.ALPHA_MAX * dt
        omega_change = np.clip(omega_diff, -max_omega_change, max_omega_change)
        target_omega_limited = prev_omega + omega_change

        # --- 2. Response Lag (First-Order Lag) ---
        # 実機の「ぬるっとした動き（慣性）」を再現するための一次遅れフィルタ
        # response_time_constant (例: 1/12秒) を時定数として使用
        
        tau = self.RESPONSE_TIME_CONSTANT
        alpha = dt / (tau + dt)  # フィルタ係数 (0~1)

        new_v = prev_v * (1.0 - alpha) + target_v_limited * alpha
        new_omega = prev_omega * (1.0 - alpha) + target_omega_limited * alpha

        # 念のためクリップ (後退禁止はここでも保証)
        new_v = np.clip(new_v, 0.0, self.V_MAX)
        new_omega = np.clip(new_omega, -self.OMEGA_MAX, self.OMEGA_MAX)

        # --- 3. Update Position (Integration) ---
        # 台形積分ではなく、単純なオイラー積分でも十分精度は出る（安定性重視）
        v_pps = new_v * self.env.PIXELS_PER_METER
        
        dx = float(-v_pps * math.sin(self.env.robot_theta) * dt)
        dy = float(-v_pps * math.cos(self.env.robot_theta) * dt)
        dtheta = float(new_omega * dt)

        next_robot_position = (
            self.env.robot_position[0] + dx,
            self.env.robot_position[1] + dy,
        )

        if (
            0 < next_robot_position[0] < self.env.SCREEN_W
            and 0 < next_robot_position[1] < self.env.SCREEN_H
        ):
            if self.env.is_collision(next_robot_position):
                self.is_collision = True
                # 衝突したら速度をゼロにリセットする（めり込み防止）
                new_v = 0.0
                new_omega = 0.0
            else:
                self.env.robot_position = next_robot_position

        self.env.robot_theta += dtheta
        self.env.robot_theta = (self.env.robot_theta + math.pi) % (2 * math.pi) - math.pi

        # --- 4. Store State ---
        self.previous_v = float(new_v)
        self.previous_omega = float(new_omega)
        self.current_v = float(new_v)
        self.current_omega = float(new_omega)

        self.env.v = self.current_v
        self.env.omega = self.current_omega

        self.env.update_path(self.env.robot_position, self.env.robot_theta)

    def _process_lidar_data(self, sensor_data):
        """
        sensor_dataは [distance, angle] のリスト (angleは -pi ~ pi でソート済み)
        """
        if not sensor_data:
            return np.full(self.N_L + self.N_E, self.LIDAR_RANGE_M, dtype=np.float32)

        processed_lidar = []
        for data in sensor_data:
            # data[0] は距離。sensor.py は範囲外の場合 self.LIDAR_RANGE_M を返す
            if data[0] < self.LIDAR_RANGE_M:
                distance = data[0] / self.LIDAR_RANGE_M
            else:
                distance = 1.0  # No obstacle (range out)
            processed_lidar.append(distance)
            
        # N_E分だけデータを水増し（先頭データを末尾に追加）
        processed_lidar += processed_lidar[: self.N_E]

        lidar_in_meters = np.array(processed_lidar, dtype=np.float32)
        return lidar_in_meters

    def _get_obs(self):
        processed_lidar = self._process_lidar_data(self.env.sensor_data)
        self.lidar_history.append(processed_lidar)

        obs_lidar = np.array(list(self.lidar_history), dtype=np.float32)
        
        goal_dist, goal_angle = self.env.get_goal_info()
        goal_angle_n = np.clip(goal_angle / math.pi, -1.0, 1.0)
        orientation_diff_n = self._get_normalized_robot_target_orientation_diff()
        
        obs_goal = np.array(
            [goal_dist, goal_angle_n, orientation_diff_n], dtype=np.float32
        )
        
        v_n = np.clip((self.current_v or 0.0) / self.V_MAX, 0.0, 1.0)
        omega_n = np.clip((self.current_omega or 0.0) / self.OMEGA_MAX, -1.0, 1.0)
        obs_velocity = np.array([v_n, omega_n], dtype=np.float32)

        # sensor.py の get_closest_obstacle_info は [raw_dist, angle] を返す
        co_dist, co_angle = self.env.get_closest_obstacle_info()
        
        # 安全距離(半径)を引いて「表面までの距離」にする
        co_dist -= self.env.ROBOT_RADIUS_M
        
        co_dist_n = np.clip(co_dist / self.LIDAR_RANGE_M, 0.0, 1.0)
        # co_angle は -pi ~ pi なので pi で割って正規化
        co_angle_n = np.clip(co_angle / math.pi, -1.0, 1.0)
        
        obs_closest_obstacle = np.array([co_dist_n, co_angle_n], dtype=np.float32)

        observation = {
            "lidar": obs_lidar,
            "goal": obs_goal,
            "velocity": obs_velocity,
            "closest_obstacle": obs_closest_obstacle,
        }
        return observation

    def _get_info(self):
        return {}

    def _randomly_seletct_map(self):
        map_types = ["random", "empty"]
        probabilities = [0.9, 0.1]
        return self.np_random.choice(map_types, p=probabilities)