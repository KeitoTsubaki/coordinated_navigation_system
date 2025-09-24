import argparse
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import torch as th
import numpy as np
from torch.nn import functional as F
from stable_baselines3.common.utils import polyak_update

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise


if __package__ is None or __package__ == "":
    # If it is not running as a package
    package_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(package_root.parent))
    from rl_training import custom_ddpg_policy  # type: ignore
    from rl_training import callbacks  # type: ignore
    from rl_training import utils  # type: ignore
else:  # pragma: no cover - executed when run as module
    # If it is running directly
    from . import custom_ddpg_policy
    from . import callbacks
    from . import utils

# Ensure the custom environment is registered
import rl_training  # noqa: F401


EpisodeLogger = callbacks.EpisodeLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a DDPG agent in the simulation environment."
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
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
        default=10_000,
        help="Timesteps before training starts.",
    )
    parser.add_argument(
        "--buffer-size", type=int, default=100_000, help="Replay buffer size."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for training updates."
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
    return run_dir


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class CustomDDPG(DDPG):
    def __init__(self, *args, action_lr_schedule=None, critic_lr_schedule=None,**kwargs):
        super().__init__(*args, **kwargs)
        self.actor_lr_schedule = action_lr_schedule
        self.critic_lr_schedule = critic_lr_schedule
        
    def _update_learning_rate(self, optimizers, lr_schedule):
        new_lr = lr_schedule(self._current_progress_remaining)
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr
    
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        # self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
        self._update_learning_rate([self.actor.optimizer], self.actor_lr_schedule)
        self._update_learning_rate([self.critic.optimizer], self.critic_lr_schedule)

        actor_losses, critic_losses = [], []
        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            # For n-step replay, discount factor is gamma**n_steps (when no early termination)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))

def main() -> None:
    args = parse_args()

    render_mode = None if args.render_mode == "none" else args.render_mode

    set_random_seed(args.seed)

    gym_env = gym.make("simulation_env-v0", render_mode=render_mode)
    gym_env.reset(seed=args.seed)
    gym_env = Monitor(gym_env)

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
        log_dir=run_dir,
        log_every=args.log_every,
        verbose=1,
    )

    model = CustomDDPG(
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
