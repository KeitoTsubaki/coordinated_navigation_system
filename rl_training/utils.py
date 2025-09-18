import torch
def split_obs(obs_dict):
    # obs_dict: Dict with keys ["lidar", "goal", "velocity", "closest_obstacle"]
    discrete_state = torch.cat([obs_dict["goal"], obs_dict["velocity"], obs_dict["closest_obstacle"]], dim=1)  # (B,6)
    lidar = obs_dict["lidar"]  # (B,4,640)
    return lidar, discrete_state