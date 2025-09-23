import os

import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import yaml
import math
import importlib.resources
from collections import deque

from simulation_core import environment


class Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()

        # ===== Initialize and load config files =====
        simulator_config_path = importlib.resources.files("simulation_core").joinpath(
            "config", "simulator_config.yaml"
        )
        ddpg_config_path = os.path.join(
            os.path.dirname(__file__), "config", "ddpg_config.yaml"  # rl_training/
        )

        with open(ddpg_config_path, "r", encoding="utf-8") as f:
            self.ddpg_config = yaml.safe_load(f)

        # ===== Initialize render mode =====
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.env = environment.SimCoreEnvironment(
            render_mode=self.render_mode, config_path=simulator_config_path
        )

        self.V_MAX = self.env.V_MAX
        self.OMEGA_MAX = self.env.OMEGA_MAX
        self.A_MAX = self.env.A_MAX
        self.ALPHA_MAX = self.env.ALPHA_MAX
        self.MAX_D = self.env.MAX_D
        self.MAX_THETA = self.env.MAX_THETA
        self.LIDAR_RANGE_M = self.env.LIDAR_RANGE_M
        self.N_L = self.ddpg_config["lidar"]["n_l"]
        self.N_E = self.ddpg_config["lidar"]["n_e"]
        self.LIDAR_HISTORY_LEN = self.ddpg_config["lidar"]["lidar_history_len"]
        self.DT = 1 / self.metadata["render_fps"]
        self.MAX_EPISODE_STEPS = self.ddpg_config["max_episode_steps"]

        # DDPG allows continuous action space, and the action of differential-
        # driven robots includes the linear and angular velocity.
        # v_n: 0 ~ V_MAX
        # omega_n: -OMEGA_MAX ~ OMEGA_MAX
        # action = [v_n, omega_n]
        self.action_space = spaces.Box(
            # Normalize action space to [-1, 1] for better compatibility with RL algorithms.
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        # We design a robot-centric observation space to remove the dependency
        # on localization and global map.
        # observation space includes o_l, o_g, o_v, o_o
        # o_l: processed LiDAR data
        # o_g:
        # o_v:
        # o_o:
        # Define a reasonable maximum distance (e.g., map diagonal) instead of infinity
        max_dist = (
            math.sqrt(self.env.SCREEN_W**2 + self.env.SCREEN_H**2)
            / self.env.PIXELS_PER_METER
        )
        self.observation_space = spaces.Dict(
            {
                "lidar": spaces.Box(
                    low=0,
                    high=self.LIDAR_RANGE_M * 100,  # LiDAR values are already in meters
                    shape=(4, self.N_L + self.N_E),
                    dtype=np.float32,
                ),
                "goal": spaces.Box(
                    low=np.array([0, -np.pi], dtype=np.float32),
                    high=np.array([max_dist, np.pi], dtype=np.float32),
                    dtype=np.float32,
                ),
                "velocity": spaces.Box(
                    low=np.array([0, -self.OMEGA_MAX], dtype=np.float32),
                    high=np.array([self.V_MAX, self.OMEGA_MAX], dtype=np.float32),
                    dtype=np.float32,
                ),
                "closest_obstacle": spaces.Box(
                    low=np.array([0, -np.pi], dtype=np.float32),
                    high=np.array([max_dist, np.pi], dtype=np.float32),
                    dtype=np.float32,
                ),
            }
        )

        self.clock = None

        # ===== Parameters that need to reset =====
        # 4フレーム分のLiDARデータを保持するためのdeque
        self.lidar_history = deque(maxlen=self.LIDAR_HISTORY_LEN)
        self.is_collision = False
        self.previous_distance_to_taret = None
        self.previous_angle_to_target = None
        self.current_distance_to_target = None
        self.current_angle_to_target = None
        self.previous_distance_to_closest_obstacle = None
        self.current_distance_to_closest_obstacle = None
        self.current_v = None
        self.current_omega = None
        self.previous_v = 0.0
        self.previous_omega = 0.0

        self.current_step = 0

    def step(self, action):
        # Map the action to the executable action
        v_e, omega_e = self._action_to_executable_action(action, self.DT)

        # Update robot position
        self._update_robot_position(v_e, omega_e, dt=self.DT)
        # Get new sensor readings from the new position
        self.env.update_sensors()

        self.render()

        # 報酬計算のために、現在の値を前の値として保存
        self.previous_distance_to_taret = self.current_distance_to_target
        self.previous_angle_to_target = self.current_angle_to_target
        self.previous_distance_to_closest_obstacle = (
            self.current_distance_to_closest_obstacle
        )
        # 新しい現在の値を取得
        self.current_distance_to_target, self.current_angle_to_target = (
            self.env.get_goal_info()
        )
        self.current_distance_to_closest_obstacle = (
            self.env.get_closest_obstacle_info()[0]
        )

        self.current_v = v_e
        self.current_omega = omega_e

        # An eppisode is done if the robot has reached the target or collided with an obstacle
        # Terminated: Reached the target or collided with an obstacle
        # Truncated: Time limit exceeded
        self.current_step += 1
        terminated = self.env.is_target_reached() or self.is_collision
        truncated = self.current_step >= self.MAX_EPISODE_STEPS
        observation = self._get_obs()
        info = self._get_info()
        reward = self._calculate_reward()

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Reset the state of the environment, and then render screen."""
        super().reset(seed=seed)  # sets self.np_random

        # render map and initialize robot ant target positions
        self.env.render_obstacle_and_robots()
        # Perform an initial sensor scan to populate observation data
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

            pygame.event.pump()  # to update the event state
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.update()

        elif self.render_mode == "rgb_array":
            # rgb_arrayモードでも、配列に変換する前に描画内容を更新する必要がある
            self.env.render_robot_and_lidar()

            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.env.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.env.screen is not None:
            pygame.display.quit()
            pygame.quit()

    def _reset_parameters(self):
        # super().reset(seed=seed) で定義された self.np_random をenvに渡す
        self.env.np_random = self.np_random

        self.is_collision = False

        # 最初の観測でLiDAR履歴を初期化
        self.lidar_history.clear()
        initial_lidar_obs = self._process_lidar_data(self.env.sensor_data)
        for _ in range(4):
            self.lidar_history.append(initial_lidar_obs)

        # 速度情報を初期化
        self.current_v = 0.0
        self.current_omega = 0.0
        self.previous_v = 0.0
        self.previous_omega = 0.0

        # 報酬計算用の変数を初期化
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

    def _calculate_reward(self):
        # r_s: spare reward corresponding to: robot reachs the target, collides with an obstacle
        # r_d: dense reward to accelerate the training and seconadary tasks such as smoothness
        r_safe = self.env.ROBOT_RADIUS_M  # robot radius in meters
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
        k_8 = self.ddpg_config["reward"]["k_8"]
        k_9 = self.ddpg_config["reward"]["k_9"]

        r_s = 0.0
        r_d = 0.0

        r_g = 0.0
        r_c = 0.0
        r_v = 0.0
        r_m = 0.0

        if self.env.is_target_reached():
            r_s = r_arrival
        elif self.is_collision:
            r_s = r_collision

        # The robot is awarded when it approaches the target and turns to the target
        # The robot is punished according to the angle deviation to the target
        r_g_d = (
            k_1
            * (self.previous_distance_to_taret - self.current_distance_to_target)
            / self.MAX_D
        )
        r_g_o1 = (
            k_2
            * (self.current_angle_to_target - self.previous_angle_to_target)
            / self.MAX_THETA
        )
        r_g_o2 = -k_3 * abs(self.current_angle_to_target) / math.pi
        r_g = r_g_d + r_g_o1 + r_g_o2

        # The robot is punished when it is aproaching and being close to the obstacles
        # r_c_1: APF formulation. it becomes zero when the distance to the closest obstacle is larger than rho_0 (Fig.3a)
        # r_c_2: the robot is awarded if leaving the obstacle, and punished if approaching the obstacle (Fig.3b)
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

        # r_v: the robot is encouraging the robot to move quickly and smoothly
        # The robot is awarded when the linear velocity is large, and punished when the angular velocity is large
        r_v_v = k_6 * (self.current_v / self.V_MAX)
        r_v_omega = -k_7 * (abs(self.current_omega) / self.OMEGA_MAX)
        r_v = r_v_v + r_v_omega

        # r_m: minor penalty to encourage the robot to actively explore the environment and shorten the trajectory. penalty for time passage
        r_m = -0.1

        r_d = r_g + r_c + r_v + r_m

        reward = r_s + r_d
        return reward

    def _action_to_executable_action(self, action, dt):
        # De-normalize the action from [-1, 1] to the robot's physical limits.
        # Linear velocity: maps [-1, 1] to [0, V_MAX]
        # Note: action[0] is used. A value of -1 maps to 0, and 1 maps to V_MAX.
        v_n = (action[0] + 1.0) / 2.0 * self.V_MAX
        # Angular velocity: maps [-1, 1] to [-OMEGA_MAX, OMEGA_MAX]
        omega_n = action[1] * self.OMEGA_MAX

        v_kmin = max(0, self.previous_v - self.A_MAX * dt)
        v_kmax = min(self.V_MAX, self.previous_v + self.A_MAX * dt)
        omega_kmin = max(-self.OMEGA_MAX, self.previous_omega - self.ALPHA_MAX * dt)
        omega_kmax = min(self.OMEGA_MAX, self.previous_omega + self.ALPHA_MAX * dt)

        v_e = np.clip(v_n, v_kmin, v_kmax)
        omega_e = np.clip(omega_n, omega_kmin, omega_kmax)

        self.previous_v = v_e
        self.previous_omega = omega_e
        return v_e, omega_e

    def _update_robot_position(self, v_e, omega_e, dt):
        # ロボットの位置と向きを更新する関数
        # robot_theta = 0: y軸負方向
        # robot_theta = π/2: x軸正方向
        # robot_theta = π: y軸正方向
        # robot_theta = 3π/2 or -π/2: x軸負方向
        # v_e は m/s, omega_e は rad/s
        # 位置はピクセル単位で計算するため、v_eをピクセル/秒に変換
        v_e_pps = v_e * self.env.PIXELS_PER_METER
        dx = (
            -v_e_pps * math.sin(self.env.robot_theta) * dt
        )  # 反時計回りを正とするためsinの符号を反転
        dy = (
            -v_e_pps * math.cos(self.env.robot_theta) * dt
        )  # Pygameのy軸は下向きが正のため、符号を反転. cosは不変
        dtheta = omega_e * dt

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
            else:
                self.env.robot_position = next_robot_position

        # self.env.robot_position[0] += dx
        # self.env.robot_position[1] += dy
        self.env.robot_theta += dtheta

        # 角度を-πからπの範囲に正規化
        self.env.robot_theta = (self.env.robot_theta + math.pi) % (
            2 * math.pi
        ) - math.pi

    def _process_lidar_data(self, sensor_data):
        """LiDARスキャンデータの最初のn_e個の点を後ろにコピーし、n_l+n_e次元のベクトルに処理する"""
        if not sensor_data:
            # センサーデータがない場合は、最大距離で埋める
            return np.full(self.N_L + self.N_E, self.LIDAR_RANGE_M, dtype=np.float32)

        processed_lidar = []
        for data in sensor_data:
            if data[0] < self.LIDAR_RANGE_M:
                distance = data[0]
            else:
                distance = self.LIDAR_RANGE_M * 100
            processed_lidar.append(distance)
        processed_lidar += processed_lidar[: self.N_E]

        # print("sensor_data_distance length:", len(sensor_data_distance))
        # print("processed_lidar length:", len(processed_lidar))

        # The sensor data is in pixels, convert to meters
        lidar_in_meters = np.array(processed_lidar, dtype=np.float32)
        # Clip values to the defined observation space range
        return lidar_in_meters

    def _get_obs(self):
        processed_lidar = self._process_lidar_data(self.env.sensor_data)
        self.lidar_history.append(processed_lidar)

        obs_lidar = np.array(list(self.lidar_history), dtype=np.float32)
        # print("obs_lidar shape:", obs_lidar.shape)
        obs_goal = np.array(self.env.get_goal_info(), dtype=np.float32)
        # print("robot theta: ", self.env.robot_theta)
        # print("obs_goal shape:", obs_goal.shape)
        # print("obs_goal:", obs_goal)
        obs_velocity = np.array([self.current_v, self.current_omega], dtype=np.float32)
        # print("obs_velocity shape:", obs_velocity.shape)
        # print("obs_velocity:", obs_velocity)
        obs_closest_obstacle = np.array(
            self.env.get_closest_obstacle_info(), dtype=np.float32
        )

        # # --- Debug Print ---
        # print(f"Robot Theta: {self.env.robot_theta:.2f} rad (0:UP)")
        # print(f"LiDAR shape: {obs_lidar.shape}")
        # print(f"Goal (Dist, Angle(relative to robot)): ({obs_goal[0]:.2f} m, {obs_goal[1]:.2f} rad)")
        # print(f"Velocity (v, omega): ({obs_velocity[0]:.2f} m/s, {obs_velocity[1]:.2f} rad/s)")
        # print(f"Closest Obstacle (Dist, Angle(relative to robot)): ({obs_closest_obstacle[0]:.2f} m, {obs_closest_obstacle[1]:.2f} rad)")

        observation = {
            "lidar": obs_lidar,
            "goal": obs_goal,
            "velocity": obs_velocity,
            "closest_obstacle": obs_closest_obstacle,
        }
        # observation = None
        return observation

    def _get_info(self):
        return {}
