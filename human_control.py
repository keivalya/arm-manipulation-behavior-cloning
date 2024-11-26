import gymnasium as gym
import gymnasium_robotics
import numpy as np

gym.register_envs(gymnasium_robotics)

if __name__ == '__main__':
    env_name = "FrankaKitchen-v1"
    max_episode_steps=500

    task = 'microwave'

    env = gym.make(env_name, max_episode_steps=max_episode_steps, tasks_to_complete=[task], render_mode='human', autoreset=False)

    state, _ = env.reset()

    print(state)