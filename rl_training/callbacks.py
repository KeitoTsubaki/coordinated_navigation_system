import csv
import os
from collections import deque
from datetime import datetime
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeLogger(BaseCallback):
    """Callback to log per-episode statistics to the console and a CSV file.

    This logger relies on the environment being wrapped with :class:`Monitor` so
    that each ``info`` dictionary includes an ``"episode"`` key containing the
    cumulative reward (``"r"``) and episode length (``"l"``).
    """

    def __init__(
        self,
        action_lr_schedule,
        critic_lr_schedule,
        log_dir: str,
        log_every: int = 10,
        verbose: int = 1,
        filename: Optional[str] = None,
    ) -> None:
        super().__init__(verbose)
        self.action_lr_schedule = action_lr_schedule
        self.critic_lr_schedule = critic_lr_schedule
        self.log_dir = log_dir
        self.log_every = max(1, log_every)
        self.filename = filename or "episode_log.csv"

        self._log_file: Optional[object] = None
        self._csv_writer: Optional[csv.writer] = None
        self._episode_counter = 0
        self._recent_rewards: deque[float] = deque(maxlen=self.log_every)
        self._recent_lengths: deque[int] = deque(maxlen=self.log_every)

    def _init_logger(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        log_path = os.path.join(self.log_dir, self.filename)
        self._log_file = open(log_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._log_file)
        self._csv_writer.writerow(
            ["timestamp", "episode", "timesteps", "episode_reward", "episode_length"]
        )

    def _close_logger(self) -> None:
        if self._log_file is not None:
            self._log_file.close()
        self._log_file = None
        self._csv_writer = None

    def _on_training_start(self) -> None:
        self._init_logger()

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")

        if dones is None or infos is None:
            return True

        dones = np.array(dones).flatten()
        for idx, done in enumerate(dones):
            if not done:
                continue

            info = infos[idx]
            episode_info = info.get("episode")
            if episode_info is None:
                # ``Monitor`` was not used around the environment
                continue

            self._episode_counter += 1
            self.action_lr_schedule.step_episode()
            self.critic_lr_schedule.step_episode()

            episode_reward = float(episode_info.get("r", 0.0))
            episode_length = int(episode_info.get("l", 0))

            self._recent_rewards.append(episode_reward)
            self._recent_lengths.append(episode_length)

            if self._csv_writer is not None:
                timestamp = datetime.utcnow().isoformat()
                self._csv_writer.writerow(
                    [
                        timestamp,
                        self._episode_counter,
                        self.num_timesteps,
                        episode_reward,
                        episode_length,
                    ]
                )
                self._log_file.flush()

            # Record metrics for Stable-Baselines3's logger (e.g., TensorBoard)
            self.logger.record("rollout/episode_reward", episode_reward)
            self.logger.record("rollout/episode_length", episode_length)

            if self.verbose > 0 and self._episode_counter % self.log_every == 0:
                reward_array = np.array(self._recent_rewards, dtype=np.float32)
                length_array = np.array(self._recent_lengths, dtype=np.float32)
                mean_reward = float(reward_array.mean())
                min_reward = float(reward_array.min())
                max_reward = float(reward_array.max())
                mean_length = float(length_array.mean())

                print(
                    (
                        f"Episode {self._episode_counter:05d} | Steps {self.num_timesteps:08d} | "
                        f"Reward avg/min/max: {mean_reward:+.2f}/{min_reward:+.2f}/{max_reward:+.2f} | "
                        f"Length avg: {mean_length:.1f}"
                    )
                )

        return True

    def _on_training_end(self) -> None:
        self._close_logger()
