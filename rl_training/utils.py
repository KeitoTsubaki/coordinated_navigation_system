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
        self.initial_e = initial_e
        self.decay_rate = decay_rate
        self.episode_count = 0
        self.current_e = initial_e
        # Call super().__init__() after initializing attributes
        # because the parent's __init__ calls self.reset()
        super().__init__(mean=mean, sigma=sigma, theta=theta)

    def __call__(self):
        return self.current_e * super().__call__()

    def reset(self):
        super().reset()
        self.current_e = self.initial_e / (1 + self.episode_count / self.decay_rate)

    def step_episode(self):
        self.episode_count += 1
    
    def get_current_e(self):
        return self.current_e


class CustomLRSchedule:
    def __init__(self, initial_lr, decay_rate):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.decay_rate = decay_rate
        self.episode_count = 0

    def __call__(self, progress_remaining):
        self.current_lr = self.initial_lr / (1 + self.episode_count / self.decay_rate)
        return self.current_lr

    def step_episode(self):
        self.episode_count += 1

    def get_current_lr(self):
        return self.current_lr
