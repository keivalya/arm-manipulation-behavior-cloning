import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from model import *
from torch.utils.tensorboard import SummaryWriter
import datetime
from replay_buffer import ReplayBuffer
import time
from gym_robotics_custom import RoboGymObservationWrapper

class Agent(object):
    def __init__(self, n_inputs, action_space, gamma, tau, alpha, target_update_interval, hidden_size, learning_rate, goal):
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.target_update_interval = target_update_interval
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing device. Running on {self.device}")

        self.critic = Critic(n_inputs, action_space.shape[0], hidden_size, name=f"critic_{goal}").to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=learning_rate)

        self.critic_target = Critic(n_inputs, action_space.shape[0], hidden_size, name=f"critic_target_{goal}").to(device=self.device)
        self.hard_update(self.critic_target, self.critic)
        
        self.actor = Actor(n_inputs, action_space.shape[0], hidden_size, action_space, name=f"actor_{goal}").to(device=self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=learning_rate)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.actor.sample(state)
        else:
            _, _, action = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory : ReplayBuffer, batch_size, updates):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample_buffer(batch_size=batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        # compute critic loss
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target

        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        # update critic network
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # compute actor policy loss
        pi, log_pi, _ = self.actor.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        # update actor network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        alpha_loss = torch.tensor(0.).to(self.device)
        alpha_tlogs = torch.tensor(self.alpha)

        if updates % self.target_update_interval == 0:
            self.soft_update(self.critic_target, self.critic)

        return qf1_loss.item(), qf2_loss.item(), actor_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def train(self, env, memory, episodes=1000, batch_size=64, updates_per_step=1, summary_writer_name="", max_episode_steps=100):
        # tensorboard
        summary_writer_name = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_'+summary_writer_name
        writer = SummaryWriter(summary_writer_name)

        total_numsteps = 0
        updates = 0

        for episode in range(episodes):
            episode_reward = 0
            episode_steps = 0
            done = False
            state, _ = env.reset()

            while not done and episode_steps < max_episode_steps:
                action = self.select_action(state)
                if memory.can_sample(batch_size=batch_size):
                    for i in range(updates_per_step):
                        critic_1_loss, critic_2_loss, actor_loss, ent_loss, alpha = self.update_parameters(memory, batch_size, updates)
                        writer.add_scalar('loss/critic_1',critic_1_loss,updates)
                        writer.add_scalar('loss/critic_2',critic_2_loss,updates)
                        writer.add_scalar('loss/actor',actor_loss,updates)
                        writer.add_scalar('loss/entropy_loss',ent_loss,updates)
                        updates += 1

                next_state, reward, done, _, _ = env.step(action)
                
                episode_steps += 1
                total_numsteps += 1
                episode_reward += reward

                mask = 1 if episode_steps == max_episode_steps else float(not done)

                memory.store_transition(state, action, reward, next_state, mask)

                state = next_state

            writer.add_scalar('reward/train', episode_reward, episode)
            print("Episode: {}, Total numsteps: {}, episode steps: {}, reward:{}".format(episode, total_numsteps, episode_steps, round(episode_reward, 2)))

            if episode % 10 == 0:
                self.save_checkpoint()

    def test(self, env : RoboGymObservationWrapper, episodes=1, max_episode_steps=500, prev_action=None):
        for episode in range(episodes):
            episode_reward = 0
            episode_steps = 0
            done = False

            if prev_action is not None:
                state, reward, done, _, _ = env.step(prev_action)
            else:
                state, _ = env.reset()

            while not done and episode_steps < max_episode_steps:
                action = self.select_action(state, evaluate=True)
                next_state, reward, done, _, _ = env.step(action)
                episode_steps += 1
                if reward == 1:
                    done = True
                    prev_action = action
                episode_reward += reward
                mask = 1 if episode_steps == max_episode_steps else float(not done)
                state = next_state
                if env.env.render_mode == "human":
                    time.sleep(0.05)
            print("Episode: {}, Episode steps: {}, reward: {}".format(episode,
                                                                      episode_steps,
                                                                      round(episode_reward, 2)))
            return prev_action, episode_reward

    def save_checkpoint(self):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        print("Saving models...")
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.critic_target.save_checkpoint()

    
    def load_checkpoint(self, evaluate=False):

        try:
            print("Loading models...")
            self.actor.load_checkpoint()
            self.critic.load_checkpoint()
            self.critic_target.load_checkpoint()
            print("Successfully loaded models")
        except:
            if evaluate:
                raise Exception("Unable to load models. Can't evaluate model")
            else:
                print("Unable to load models. Starting from scratch")
        
        if evaluate:
            self.actor.eval()
            self.critic.eval()
            self.critic_target.eval()
        else:
            self.actor.train()
            self.critic.train()
            self.critic_target.train()