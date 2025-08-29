import gymnasium as gym
import gym_environment

# env = gym.make('matplotllib_2d_simulator-v0', render_mode='human')

env = gym_environment.Env(render_mode='human')

obs, info = env.reset()

for _, i in enumerate(range(300)):
    print(i)
    action = env.action_space.sample()  # Replace with your action
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated:
        print("Episode finished")
        break
env.close()