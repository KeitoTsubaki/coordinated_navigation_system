import csv
import os
import sys
import select
from collections import deque
from datetime import datetime
from typing import Optional
import imageio
import torch

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder


class EpisodeLoggerV2(BaseCallback):
    """Callback to log per-episode statistics to the console and a CSV file.

    This logger relies on the environment being wrapped with :class:`Monitor` so
    that each ``info`` dictionary includes an ``"episode"`` key containing the
    cumulative reward (``"r"``) and episode length (``"l"``).
    """

    def __init__(
        self,
        action_lr_schedule,
        critic_lr_schedule,
        action_noise,
        gym_env,
        eval_env,
        eval_starts_episode: int,
        eval_freq_episodes: int,
        fps: int,
        video_dir: str,
        log_dir: str,
        log_every: int = 10,
        verbose: int = 1,
        filename: Optional[str] = None,
        decay_patience: int = 500,
        early_stop_patience: int = 1000,
    ) -> None:
        super().__init__(verbose)
        self.action_lr_schedule = action_lr_schedule
        self.critic_lr_schedule = critic_lr_schedule
        self.action_noise = action_noise
        self.gym_env = gym_env
        self.eval_env = eval_env
        self.eval_starts_episode = eval_starts_episode
        self.eval_freq_episodes = eval_freq_episodes
        self.fps = fps
        self.video_dir = video_dir
        self.log_dir = log_dir
        self.log_every = max(1, log_every)
        self.filename = filename or "episode_log.csv"
        self.decay_patience = decay_patience
        self.early_stop_patience = early_stop_patience
        self.best_success_rate = -np.inf
        self.no_improve_counter = 0
        self._evaluation_history: list[float] = []
        self._decay_event_count = 0

        self.periodic_test_count = 0
        self._should_decay_parameters = False
        self._log_file: Optional[object] = None
        self._csv_writer: Optional[csv.writer] = None
        self._eval_log_file: Optional[object] = None
        self._eval_csv_writer: Optional[csv.writer] = None
        self._velocity_log_file: Optional[object] = None
        self._velocity_csv_writer: Optional[csv.writer] = None
        self._episode_counter = 0
        self._recent_rewards: deque[float] = deque(maxlen=self.log_every)
        self._recent_lengths: deque[int] = deque(maxlen=self.log_every)

        self.in_eval = False
        
        
        
        self.n_eval_episodes = {"random": 100, "empty": 10}
        
    def _init_logger(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        log_path = os.path.join(self.log_dir, self.filename)
        self._log_file = open(log_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._log_file)
        self._csv_writer.writerow(
            [
                "timestamp",
                "episode",
                "timesteps",
                "episode_reward",
                "episode_length",
                "action_lr",
                "critic_lr",
                "action_noise_epsilon",
            ]
        )
        eval_log_path = os.path.join(self.log_dir, "eval_log.csv")
        self._eval_log_file = open(eval_log_path, "w", newline="", encoding="utf-8")
        self._eval_csv_writer = csv.writer(self._eval_log_file)
        self._eval_csv_writer.writerow(
            [
                "timestamp",
                "episode",
                "type",
                "mean_reward",
                "success_rate",
            ]
        )
        velocity_log_path = os.path.join(self.log_dir, "velocity_log.csv")
        self._velocity_log_file = open(velocity_log_path, "w", newline="", encoding="utf-8")
        self._velocity_csv_writer = csv.writer(self._velocity_log_file)
        self._velocity_csv_writer.writerow(
            [
                "timestamp",
                "episode",
                "linear_velocity",
                "angular_velocity"
            ]
        )

    def _close_logger(self) -> None:
        if self._log_file is not None:
            self._log_file.close()
        self._log_file = None
        self._csv_writer = None
        if self._eval_log_file is not None:
            self._eval_log_file.close()
        self._eval_log_file = None
        self._eval_csv_writer = None
        if self._velocity_log_file is not None:
            self._velocity_log_file.close()
        self._velocity_log_file = None
        self._velocity_csv_writer = None

    def _on_training_start(self) -> None:
        self._init_logger()

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")

        if dones is None or infos is None:
            return True

        dones = np.array(dones).flatten()
        for idx, done in enumerate(dones):

            timestamp = datetime.utcnow().isoformat()
            self._velocity_csv_writer.writerow(
                [
                    timestamp,
                    self._episode_counter,
                    self.gym_env.unwrapped.current_v,
                    self.gym_env.unwrapped.current_omega,
                ]
            )
            self._velocity_log_file.flush()

            if not done:
                continue

            info = infos[idx]
            episode_info = info.get("episode")
            if episode_info is None:
                # ``Monitor`` was not used around the environment
                continue

            self._episode_counter += 1

            episode_reward = float(episode_info.get("r", 0.0))
            episode_length = int(episode_info.get("l", 0))

            self._recent_rewards.append(episode_reward)
            self._recent_lengths.append(episode_length)

            action_lr = self.action_lr_schedule.get_current_lr()
            critic_lr = self.critic_lr_schedule.get_current_lr()
            action_noise_epsilon = self.action_noise.get_current_e()

            if self._csv_writer is not None:
                timestamp = datetime.utcnow().isoformat()
                self._csv_writer.writerow(
                    [
                        timestamp,
                        self._episode_counter,
                        self.num_timesteps,
                        episode_reward,
                        episode_length,
                        action_lr,
                        critic_lr,
                        action_noise_epsilon,
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
            
            if (
                self._episode_counter % self.eval_freq_episodes == 0
                and self._episode_counter >= self.eval_starts_episode
            ):
                (
                    overall_success_rate,
                    success_rate_by_map,
                ) = self.periodic_test_and_record()
                self._evaluation_history.append(overall_success_rate)
                running_best = max(self._evaluation_history)
                self.logger.record("evaluation/overall_success_rate", overall_success_rate)
                self.logger.record("evaluation/running_best_success_rate", running_best)
                random_success_rate = success_rate_by_map.get("random", 0.0)
                if random_success_rate > 0.8 and not self._should_decay_parameters:
                    self._should_decay_parameters = True
                    if self.verbose > 0:
                        print(
                            "🔁 Plateau-triggered parameter decay enabled: "
                            f"random map success={random_success_rate:.3f}"
                        )

                if overall_success_rate > self.best_success_rate:
                    if self.verbose > 0 and self.best_success_rate > -np.inf:
                        print(
                            "✅ Evaluation success improved from "
                            f"{self.best_success_rate:.3f} to {overall_success_rate:.3f}"
                        )
                    self.best_success_rate = overall_success_rate
                    self.no_improve_counter = 0
                else:
                    self.no_improve_counter += self.eval_freq_episodes
                    self.logger.record(
                        "evaluation/episodes_without_improvement", self.no_improve_counter
                    )

                if (
                    self._should_decay_parameters
                    and self.no_improve_counter >= self.decay_patience
                ):
                    plateau_episodes = self.no_improve_counter
                    self.action_lr_schedule.step_episode(plateau_episodes)
                    self.critic_lr_schedule.step_episode(plateau_episodes)
                    self.action_noise.step_episode(plateau_episodes)
                    new_action_lr = self.action_lr_schedule.decay_once()
                    new_critic_lr = self.critic_lr_schedule.decay_once()
                    new_noise_epsilon = self.action_noise.decay_once()
                    self._decay_event_count += 1
                    self.logger.record("train/plateau_decay_events", self._decay_event_count)
                    self.logger.record("train/action_lr", new_action_lr)
                    self.logger.record("train/critic_lr", new_critic_lr)
                    self.logger.record("train/action_noise_epsilon", new_noise_epsilon)
                    if self.verbose > 0:
                        print(
                            "📉 Plateau detected: "
                            f"{plateau_episodes} episodes without improvement. "
                            f"Decayed action LR to {new_action_lr:.6f}, "
                            f"critic LR to {new_critic_lr:.6f}, "
                            f"noise epsilon to {new_noise_epsilon:.6f}."
                        )
                    self.no_improve_counter = 0

                if (
                    not self._should_decay_parameters
                    and self.no_improve_counter >= self.early_stop_patience
                ):
                    if self.verbose > 0:
                        print("⏹ Early stopping triggered!")
                    return False
                
        
        return True

    def _on_training_end(self) -> None:
        self._close_logger()

    

    def periodic_test_and_record(self) -> tuple[float, dict[str, float]]:
        print("\nStarted periodic evaluation...")
        # self.in_eval = True
        self.periodic_test_count += 1
        total_episodes = sum(self.n_eval_episodes.values())
        episode_count = 0
        total_successes = []
        success_rate_by_map = {}
        for map_name, n_episodes in self.n_eval_episodes.items():
            rewards, successes = [], []
            for ep in range(n_episodes):
                is_record = False
                # 1環境につき1つの動画を保存
                if ep == 0:
                    is_record = True
                writer=None
                
                obs, _ = self.eval_env.reset(options={"map_type": map_name})
                total_reward = 0.0
                success = 0
                
                if is_record:
                    print(f"Recording evaluation video for map: {map_name}, episode: {ep + 1}")
                    video_path = os.path.join(self.video_dir, f"{map_name}_test_{self.periodic_test_count}.mp4")
                    writer = imageio.get_writer(video_path, fps=self.fps, codec="libx264", format="mp4")
                
                try:
                    while True:
                        action, _states = self.model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = self.eval_env.step(action)
                        total_reward += reward
                        
                        if terminated or truncated:
                            if info.get("is_target_reached", True):
                                success = 1
                            break

                        if writer is not None:                 
                            frame = self.eval_env.render()
                            if frame is not None and frame.ndim == 3:
                                writer.append_data(frame)
                finally:   
                    if writer is not None:
                        writer.close()
                        print(f"Saved evaluation video to: {video_path}")
                
                rewards.append(total_reward)
                successes.append(success)
                total_successes.append(success)
                episode_count += 1
                if episode_count % 10 == 0:
                    self.show_test_progress(episode_count, total_episodes)
                
            mean_reward = np.mean(rewards)
            success_rate = np.mean(successes)
            overall_success_rate = np.mean(total_successes)
            success_rate_by_map[map_name] = success_rate

            if self._eval_csv_writer is not None:
                timestamp = datetime.utcnow().isoformat()
                self._eval_csv_writer.writerow(
                    [
                        timestamp,
                        self._episode_counter,
                        map_name,
                        mean_reward,
                        success_rate,
                    ]
                )
                self._eval_log_file.flush()
        if self._eval_csv_writer is not None:
            timestamp = datetime.utcnow().isoformat()
            self._eval_csv_writer.writerow(
                [
                    timestamp,
                    self._episode_counter,
                    "overall",
                    mean_reward,
                    overall_success_rate,
                ]
            )
            self._eval_log_file.flush()
        
        save_path = os.path.join(self.log_dir, f"model_ep{self._episode_counter}.zip")
        self.model.save(save_path)
        print(f"[PeriodicTest] Saved model to {save_path}")
        
        # self.in_eval = False
        return overall_success_rate, success_rate_by_map
    
    def show_test_progress(self, episode: int, total_episodes: int) -> None:
        progress = (episode / total_episodes) * 100
        progress_bar = ""
        for i in range(10):
            if i < (progress / 10):
                bar = "#"
            else:
                bar = "-"
            progress_bar += bar
        print(f"[{progress_bar}] {progress:.2f}%")

        