import torch
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise


def split_obs(obs_dict):
    # obs_dict: Dict with keys ["lidar", "goal", "velocity", "closest_obstacle"]
    discrete_state = torch.cat(
        [obs_dict["goal"], obs_dict["velocity"], obs_dict["closest_obstacle"]], dim=1
    )  # (B,6)
    lidar = obs_dict["lidar"]  # (B,4,640)
    return lidar, discrete_state


class DecayOUActionNoise(OrnsteinUhlenbeckActionNoise):
    def __init__(self, mean, theta, sigma, initial_e, decay_rate):
        super().__init__(mean=mean, theta=theta, sigma=sigma)
        self.initial_e = initial_e
        self.decay_rate = decay_rate

        self.episode_count = 0
        self.current_e = initial_e

    def __call__(self):
        return self.current_e * super().__call__()

    def reset(self):
        super().reset()
        self.episode_count += 1
        self.current_e = self.initial_e / (1 + self.episode_count / self.decay_rate)
