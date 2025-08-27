import gymnasium as gym
from gymnasium import spaces
import numpy as np


class Env(gym.Env):
    def __init__(self, v_max, omega_max):
        super().__init__()
        # Define action and observation space

        self.v_max = v_max
        self.omega_max = omega_max

        # DDPG allows continuous action space, and the action of differential-
        # driven robots includes the linear and angular velocity.
        # v_n: 0 ~ v_max
        # omega_n: -omega_max ~ omega_max
        self.action_space = spaces.Box(
            low=np.array([0, -omega_max]),
            high=np.array([v_max, omega_max]),
            dtype=np.float32,
        )
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 3), dtype=np.uint8
        )

    def step(self, action):
        # Execute one time step within the environment
        pass

    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state
        super().reset(seed=seed)
        pass

    def render(self):
        # Render the environment to the screen
        pass

    def close(self):
        # close the environment
        pass
