import argparse
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.monitor import Monitor


if __package__ is None or __package__ == "":
    # If it is not running as a package
    package_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(package_root.parent))
    from rl_training import custom_ddpg_policy  # type: ignore
    from rl_training import ddpg  # type: ignore
    from rl_training import callbacks_v2  # type: ignore
    from rl_training import utils  # type: ignore
else:  # pragma: no cover - executed when run as module
    # If it is running directly
    from . import custom_ddpg_policy
    from . import ddpg
    from . import callbacks_v2
    from . import utils

# Ensure the custom environment is registered
import rl_training  # noqa: F401


EpisodeLogger = callbacks_v2.EpisodeLoggerV2

ENVIRONMENT_NAME = "simulation_env-square-v1"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a DDPG agent in the simulation environment."
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=10_000_000,
        help="Number of training timesteps.",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default="none",
        choices=["none", "human", "rgb_array"],
        help="Rendering mode to use during training. 'none' disables rendering.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "logs"),
        help="Directory where training logs and models will be saved.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Number of completed episodes between console log updates.",
    )
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=25_000,
        help="Timesteps before training starts.",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=50_000,
        help="Replay buffer size.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training updates.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.005,
        help="Soft update coefficient for target networks.",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Logging interval for Stable-Baselines3.",
    )
    return parser.parse_args()


def make_run_directory(base_dir: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(base_dir, f"ddpg_run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "videos"), exist_ok=True)
    return run_dir


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()

    render_mode = None if args.render_mode == "none" else args.render_mode

    set_random_seed(args.seed)

    gym_env = gym.make(ENVIRONMENT_NAME, render_mode=render_mode)
    gym_env.reset(seed=args.seed)
    gym_env = Monitor(gym_env)
    
    eval_env = gym.make(ENVIRONMENT_NAME, render_mode="rgb_array", eval_mode=True)
    eval_env.reset(seed=args.seed + 1000)


    n_actions = gym_env.action_space.shape[-1]
    action_noise = utils.DecayOUActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.2 * np.ones(n_actions),
        theta=0.15 * np.ones(n_actions),
        initial_e=1.0,
        decay_rate=4000,
    )
    run_dir = make_run_directory(os.path.abspath(args.log_dir))

    action_lr_schedule = utils.CustomLRSchedule(initial_lr=0.0002, decay_rate=2000)
    critic_lr_schedule = utils.CustomLRSchedule(initial_lr=0.0005, decay_rate=2000)

    callback = EpisodeLogger(
        action_lr_schedule=action_lr_schedule,
        critic_lr_schedule=critic_lr_schedule,
        action_noise=action_noise,
        gym_env=gym_env,
        eval_env=eval_env,
        eval_starts_episode=100,
        eval_freq_episodes=100,
        fps=10,
        video_dir=os.path.join(run_dir, "videos"),
        log_dir=run_dir,
        log_every=args.log_every,
        verbose=1,
    )

    model = ddpg.CustomDDPG(
        policy=custom_ddpg_policy.CustomDDPGPolicy,
        env=gym_env,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        action_noise=action_noise,
        tau=args.tau,
        gamma=args.gamma,
        verbose=1,
        tensorboard_log=run_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
        action_lr_schedule=action_lr_schedule,
        critic_lr_schedule=critic_lr_schedule,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        log_interval=args.log_interval,
        progress_bar=True,  # 気になったからTrueにしてみた
    )

    model_path = os.path.join(run_dir, "ddpg_model")
    model.save(model_path)

    gym_env.close()

    print(f"Training completed. Model saved to: {model_path}")
    print(f"Training logs and TensorBoard data are available in: {run_dir}")


if __name__ == "__main__":
    main()
