import argparse
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor

if __package__ is None or __package__ == "":
    # If it is not running as a package
    package_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(package_root.parent))
    from rl_training import ddpg_td3  # type: ignore
    from rl_training import callbacks  # type: ignore
else:  # pragma: no cover - executed when run as module
    # If it is running directly
    from . import ddpg_td3
    from . import callbacks

# Ensure the custom environment is registered
import rl_training  # noqa: F401

CustomDDPGPolicy = ddpg_td3.CustomDDPGPolicy
EpisodeLogger = callbacks.EpisodeLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DDPG agent in the simulation environment.")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000, help="Number of training timesteps.")
    parser.add_argument(
        "--render-mode",
        type=str,
        default="none",
        choices=["none", "human", "rgb_array"],
        help="Rendering mode to use during training. 'none' disables rendering.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
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
    parser.add_argument("--learning-starts", type=int, default=10_000, help="Timesteps before training starts.")
    parser.add_argument("--buffer-size", type=int, default=100_000, help="Replay buffer size.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training updates.")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update coefficient for target networks.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--log-interval", type=int, default=10, help="Logging interval for Stable-Baselines3.")
    return parser.parse_args()


def make_run_directory(base_dir: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(base_dir, f"ddpg_run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
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

    env = gym.make("simulation_env-v0", render_mode=render_mode)
    env.reset(seed=args.seed)
    env = Monitor(env)

    run_dir = make_run_directory(os.path.abspath(args.log_dir))

    callback = EpisodeLogger(log_dir=run_dir, log_every=args.log_every, verbose=1)

    model = DDPG(
        policy=CustomDDPGPolicy,
        env=env,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        tau=args.tau,
        gamma=args.gamma,
        verbose=1,
        tensorboard_log=run_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    model.learn(total_timesteps=args.total_timesteps, callback=callback, log_interval=args.log_interval)

    model_path = os.path.join(run_dir, "ddpg_model")
    model.save(model_path)


    env.close()

    print(f"Training completed. Model saved to: {model_path}")
    print(f"Training logs and TensorBoard data are available in: {run_dir}")

if __name__ == "__main__":
    main()