import gymnasium as gym
# rl_trainingパッケージをインポートすることで、__init__.pyが実行され、環境が登録される
import rl_training
from gymnasium.utils.env_checker import check_env

# env = gym.make('matplotllib_2d_simulator-v0', render_mode='human')

# gymnasium.make() を使って環境をインスタンス化する（推奨される方法）
env = gym.make('simulation_env-v0', render_mode='human')

# check_env関数で環境がGymnasiumのAPIに準拠しているかテスト
print("Checking environment...")
check_env(env.unwrapped)
print("Environment check passed!")

# obs, info = env.reset()

# for _, i in enumerate(range(300)):
#     print(i)
#     action = env.action_space.sample()  # Replace with your action
#     obs, reward, terminated, truncated, info = env.step(action)
#     env.render()
#     if terminated:
#         print("Episode finished")
#         break
# env.close()