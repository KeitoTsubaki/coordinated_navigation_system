import os

import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import yaml

import environment

class Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    def __init__(self, render_mode=None):
        super().__init__()
        
        config_path = "config/simulator_config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        # Define action and observation space
        self.v_max = config["robot"]["v_max"]
        self.omega_max = config["robot"]["omega_max"]
        self.a_max = config["robot"]["a_max"]
        self.alpha_max = config["robot"]["alpha_max"]
        self.max_d = config["robot"]["max_d"]
        self.max_theta = config["robot"]["max_theta"]
        self.r_lidar = config["lidar"]["range_m"]
        self.n_l = 10
        self.n_e = 2
        
        self.v_previous = 0.0
        self.omega_previous = 0.0
        
        
        self.env = environment.Environment()
        
        

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
        
        # Map the action to the executable action
        v_e, omega_e = self._action_to_executable_action(action)
        
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
        #! TODO: 正しい初期化に変更。この時点でランダムな値にする？
        
        self.env.np_random = self.np_random
        self.env.render_window()
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, info

    def _calculate_reward(self):
        #! TODO: reward関数を作る
        reward = 0.0
        return reward
    
    def _action_to_executable_action(self, action):
        v_n, omega_n = action

        dt = 0.1  # 仮の値
        v_kmin = max(0, self.v_previous - self.a_max * dt)
        v_kmax = min(self.v_max, self.v_previous + self.a_max * dt)
        omega_kmin = max(-self.omega_max, self.omega_previous - self.alpha_max * dt)
        omega_kmax = min(self.omega_max, self.omega_previous + self.alpha_max * dt)
        
        v_e = np.clip(v_n, v_kmin, v_kmax)
        omega_e = np.clip(omega_n, omega_kmin, omega_kmax)
        
        self.v_previous = v_e
        self.omega_previous = omega_e
        return v_e, omega_e
    
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
            
            # env = environment.Environment()
            
            self.window = self.env.map
            
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        if self.render_mode == "human":
            self.env.clear_lidar_layer()  # Reset the lidar line while remaining the map and obstacles
            
            self.env.change_map_by_event()

            self.env._operate_moving()  # to handle robot movement

            sensor_data = self.env.lidar_scan()
            self.env.show_sensorData(sensor_data)
            
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
            
            
