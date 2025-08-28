import gymnasium as gym
from gymnasium import spaces
import numpy as np


class Env(gym.Env):
    def __init__(self, v_max, omega_max, r_lidar, n_l, n_e):
        super().__init__()
        # Define action and observation space

        self.v_max = v_max
        self.omega_max = omega_max
        self.r_lidar = r_lidar
        self.n_l = n_l
        self.n_e = n_e

        # DDPG allows continuous action space, and the action of differential-
        # driven robots includes the linear and angular velocity.
        # v_n: 0 ~ v_max
        # omega_n: -omega_max ~ omega_max
        self.action_space = spaces.Box(
            low=np.array([0, -omega_max]),
            high=np.array([v_max, omega_max]),
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
                    low=[0, -np.pi], high=[np.inf, np.pi], dtype=np.float32
                ),
                "velocity": spaces.Box(
                    low=[0, 0], high=[self.v_max, self.omega_max], dtype=np.float32
                ),
                "closest_obstacle": spaces.Box(
                    low=[0, -np.pi], high=[np.inf, np.pi], dtype=np.float32
                ),
            }
        )
        self.reset()

    def step(self, action):
        # Execute one time step within the environment
        #! TODO: actionの出力値に変更を加える
        #! TODO: reward関数を作る
        v, omega = action
        dt = 0.1
        self.robot_theta += omega * dt
        self.robot_pos += v * dt * np.array([np.cos(self.robot_theta), np.sin(self.robot_theta)])

        reward = -np.linalg.norm(self.robot_pos - self.goal_pos)
        terminated = np.linalg.norm(self.robot_pos - self.goal_pos) < 0.5
        truncated = self.step_count > 200
        self.step_count += 1
        observation = None
        info = None
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state
        super().reset(seed=seed)
        #! TODO: 正しい初期化に変更。この時点でランダムな値にする？
        self.robot_pos = np.array([1.0, 1.0])
        self.robot_theta = 0.0
        self.goal_pos = np.array([8.0, 8.0])
        self.obstacles = [((4, 4), (2, 1))]  # (center), (width, height)
        observation = None
        info = None
        return observation, info

