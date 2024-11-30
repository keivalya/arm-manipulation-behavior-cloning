import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper


class RoboGymObservationWrapper(ObservationWrapper):
    def __init__(self, env: gym.Env, goal):
        super(RoboGymObservationWrapper, self).__init__(env)
        env_model = env.env.env.env.model
        env_model.opt.gravity[:] = [0, 0, -1]
        self.goal = goal

    def set_goal(self, goal):
        self.goal = goal

    def reset(self):
        observation, info = self.env.reset()
        observation = self.process_observation(observation)
        return observation, info
    
    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        observation = self.process_observation(observation)
        return observation, reward, done, truncated, info
    
    def process_observation(self, observation):
        obs_pos = observation['observation']
        obs_achieved_goal = observation['achieved_goal']
        obs_desired_goal = observation['desired_goal']

        obs_concatenated = np.concatenate((obs_pos, obs_achieved_goal[self.goal], obs_desired_goal[self.goal]))

        return obs_concatenated