import os

import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import yaml
import math

from simulation_core import environment

class Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    def __init__(self, render_mode=None):
        super().__init__()

        # Resolve the config file path relative to this file
        # This allows running this script without sourcing the ROS 2 workspace
        config_path = os.path.join(
            os.path.dirname(__file__), # rl_training/
            '..', # coordinated_navigation_system/
            'simulation_core', 'config', 'simulator_config.yaml'
        )
        self.env = environment.SimCoreEnvironment(config_path=config_path)

        # Define action and observation space
        self.v_max = self.env.v_max
        self.omega_max = self.env.omega_max
        self.a_max = self.env.a_max
        self.alpha_max = self.env.alpha_max
        self.max_d = self.env.max_d
        self.max_theta = self.env.max_theta
        self.r_lidar = self.env.r_lidar
        self.n_l = 10
        self.n_e = 2

        self.v_previous = 0.0
        self.omega_previous = 0.0

        # DDPG allows continuous action space, and the action of differential-
        # driven robots includes the linear and angular velocity.
        # v_n: 0 ~ v_max
        # omega_n: -omega_max ~ omega_max
        # action = [v_n, omega_n]
        self.action_space = spaces.Box(
            low=np.array([0, -self.omega_max]),
            high=np.array([self.v_max, self.omega_max]),
            dtype=np.float32,
        )
        # We design a robot-centric observation space to remove the dependency
        # on localization and global map.
        # observation space includes o_l, o_g, o_v, o_o
        # o_l: processed LiDAR data
        # o_g:
        # o_v:
        # o_o:
        self.observation_space = spaces.Dict(
            {
                "lidar": spaces.Box(
                    low=0,
                    high=100 * self.r_lidar,
                    shape=(4, self.n_l + self.n_e),
                    dtype=np.float32,
                ),
                "goal": spaces.Box(
                    low=np.array([0, -np.pi]), high=np.array([np.inf, np.pi]), dtype=np.float32
                ),
                "velocity": spaces.Box(
                    low=np.array([0, 0]), high=np.array([self.v_max, self.omega_max]), dtype=np.float32
                ),
                "closest_obstacle": spaces.Box(
                    low=np.array([0, -np.pi]), high=np.array([np.inf, np.pi]), dtype=np.float32
                ),
            }
        )
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        # for human-rendering
        self.window = None
        self.clock = None
        
        
    def step(self, action):
        # Execute one time step within the environment
        
        dt = 1 / self.metadata["render_fps"]
        # Map the action to the executable action
        v_e, omega_e = self._action_to_executable_action(action, dt)
        self.update_robot_position(v_e, omega_e, dt=dt)
        print(f"Robot Info: {self.env.get_goal_info()}")
        
        # An eppisode is done if the robot has reached the target or collided with an obstacle
        # Terminated: Reached the target or collided with an obstacle
        # Truncated: Time limit exceeded
        terminated = self.env.check_reaching_target()
        truncated = False
        observation = self._get_obs()
        info = self._get_info()
        reward = self._calculate_reward()
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state
        super().reset(seed=seed)
        pygame.init()
        pygame.display.init()
        
        self.env.np_random = self.np_random
        self.env.render_obstacles()
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, info

    def _calculate_reward(self):
        #! TODO: reward関数を作る
        reward = 0.0
        return reward
    
    def _action_to_executable_action(self, action, dt):
        v_n, omega_n = action

        v_kmin = max(0, self.v_previous - self.a_max * dt)
        v_kmax = min(self.v_max, self.v_previous + self.a_max * dt)
        omega_kmin = max(-self.omega_max, self.omega_previous - self.alpha_max * dt)
        omega_kmax = min(self.omega_max, self.omega_previous + self.alpha_max * dt)
        
        v_e = np.clip(v_n, v_kmin, v_kmax)
        omega_e = np.clip(omega_n, omega_kmin, omega_kmax)
        
        self.v_previous = v_e
        self.omega_previous = omega_e
        return v_e, omega_e
    
    def update_robot_position(self, v_e, omega_e, dt):
        # ロボットの位置と向きを更新する関数
        # robot_theta = 0: y軸負方向
        # robot_theta = π/2: x軸正方向
        # robot_theta = π: y軸正方向
        # robot_theta = 3π/2 or -π/2: x軸負方向
        # v_e は m/s, omega_e は rad/s
        # 位置はピクセル単位で計算するため、v_eをピクセル/秒に変換
        v_e_pps = v_e * self.env.pixels_per_meter
        dx = v_e_pps * math.sin(self.env.robot_theta) * dt
        dy = -v_e_pps * math.cos(self.env.robot_theta) * dt  # Pygameのy軸は下向きが正のため、符号を反転
        dtheta = omega_e * dt
        
        
        next_robot_position = (
            self.env.robot_position[0] + dx,
            self.env.robot_position[1] + dy,
        )
        
        if 0 < next_robot_position[0] < self.env.map_w and 0 < next_robot_position[1] < self.env.map_h:
            if not self.env.is_collision(next_robot_position):
                self.env.robot_position = next_robot_position
                
        # self.env.robot_position[0] += dx
        # self.env.robot_position[1] += dy
        self.env.robot_theta += dtheta

        # 角度を0から2πの範囲に正規化
        if self.env.robot_theta > 2 * math.pi:
            self.env.robot_theta -= 2 * math.pi
        elif self.env.robot_theta < 0:
            self.env.robot_theta += 2 * math.pi
    
    def _get_obs(self):
        observation = {
                "lidar", "goal", "velocity", "closest_obstacle"
            }
        return observation

    def _get_info(self):
        info = None
        return info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            self.window = self.env.map
            
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        if self.render_mode == "human":
            self.env.render_robot_and_lidar()
            
            pygame.event.pump() # to update the event state
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            )
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            
            
