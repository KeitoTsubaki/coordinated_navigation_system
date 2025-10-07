import torch as th
import numpy as np
from torch.nn import functional as F
from stable_baselines3.common.utils import polyak_update
from stable_baselines3 import DDPG


class CustomDDPG(DDPG):
    def __init__(
        self, *args, action_lr_schedule=None, critic_lr_schedule=None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.actor_lr_schedule = action_lr_schedule
        self.critic_lr_schedule = critic_lr_schedule

    # --- Helper: モデル内から LiDARNetwork を取り出す ---
    @staticmethod
    def _find_lidar_module(module):
        # サブモジュールを総なめにして LiDARNetwork を見つける
        for m in module.modules():
            if m.__class__.__name__ == "LiDARNetwork":
                return m
        raise RuntimeError("LiDARNetwork module not found in given model.")

    @staticmethod
    def _copy_params(dst_module, src_module):
        with th.no_grad():
            for p_dst, p_src in zip(dst_module.parameters(), src_module.parameters()):
                p_dst.copy_(p_src)

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
            discounts = (
                replay_data.discounts
                if replay_data.discounts is not None
                else self.gamma
            )

            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(
                    0, self.target_policy_noise
                )
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                # next_actions = (
                #     self.actor_target(replay_data.next_observations) + noise
                # ).clamp(-1, 1)
                # 環境のアクション空間に合わせて clamp
                action_low = th.as_tensor(
                    self.action_space.low, device=self.device, dtype=noise.dtype
                )
                action_high = th.as_tensor(
                    self.action_space.high, device=self.device, dtype=noise.dtype
                )

                next_actions = self.actor_target(replay_data.next_observations) + noise
                next_actions = th.max(th.min(next_actions, action_high), action_low)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(
                    self.critic_target(replay_data.next_observations, next_actions),
                    dim=1,
                )
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * discounts * next_q_values
                )

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(
                replay_data.observations, replay_data.actions
            )

            # Compute critic loss
            critic_loss = sum(
                F.smooth_l1_loss(current_q, target_q_values) for current_q in current_q_values
            )
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            critic_lidar = self._find_lidar_module(self.critic)
            actor_lidar  = self._find_lidar_module(self.actor)
            self._copy_params(actor_lidar, critic_lidar)

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                actor_loss = -self.critic.q1_forward(
                    replay_data.observations, self.actor(replay_data.observations)
                ).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                critic_lidar = self._find_lidar_module(self.critic)
                actor_lidar  = self._find_lidar_module(self.actor)
                self._copy_params(critic_lidar, actor_lidar)


                polyak_update(
                    self.critic.parameters(), self.critic_target.parameters(), self.tau
                )
                polyak_update(
                    self.actor.parameters(), self.actor_target.parameters(), self.tau
                )
                # Copy running stats, see GH issue #996
                polyak_update(
                    self.critic_batch_norm_stats,
                    self.critic_batch_norm_stats_target,
                    1.0,
                )
                polyak_update(
                    self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0
                )

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
