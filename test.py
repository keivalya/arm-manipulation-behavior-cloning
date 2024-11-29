import time
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
from gym_robotics_custom import RoboGymObservationWrapper
from replay_buffer import ReplayBuffer
from model import *
from agent import *

import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/Users/keivalya/Desktop/CS5180/project/robot-arm-manipulation-HRL/ffmpeg"

if __name__ == '__main__':

    env_name = "FrankaKitchen-v1"
    max_episode_steps = 500
    replay_buffer_size = 1000000
    task = 'slide cabinet'
    task_no_spaces = task.replace(" ", "_")
    gamma = 0.99
    tau = 0.005
    alpha = 0.1
    target_update_interval = 1
    updates_per_step = 4
    hidden_size = 512
    learning_rate = 0.0001
    batch_size = 64

    # env = gym.make(env_name, max_episode_steps=max_episode_steps, tasks_to_complete=[task], render_mode='human')
    env = gym.make(env_name, max_episode_steps=max_episode_steps, tasks_to_complete=[task], render_mode='rgb_array')
    env = RecordVideo(env, video_folder="./save_videos/", name_prefix=f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}", disable_logger=True)
    env = RoboGymObservationWrapper(env, goal=task)


    observation, info = env.reset()

    observation_size = observation.shape[0]
    
    agent = Agent(observation_size, env.action_space, gamma=gamma, tau=tau,
                  alpha=alpha, target_update_interval=target_update_interval,
                  hidden_size=hidden_size, learning_rate=learning_rate, goal=task_no_spaces)
    
    memory = ReplayBuffer(replay_buffer_size, input_size=observation_size,
                          n_actions=env.action_space.shape[0], augment_rewards=True,
                          augment_data=True)
    
    agent.load_checkpoint(evaluate=True)

    agent.test(env=env, episodes=3, max_episode_steps=max_episode_steps)

    env.close()