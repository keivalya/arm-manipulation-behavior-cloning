import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_size, n_actions, augment_data=False, augment_rewards=False,
                expert_data_ratio=0.1, augment_noise_ratio=0.1):
        self.mem_size = max_size
        self.mem_counter = 0
        self.state_memory = np.zeros((self.mem_size, input_size))
        self.new_state_memory = np.zeros((self.mem_size, input_size))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        self.augment_data = augment_data
        self.augment_rewards = augment_rewards
        self.augment_noise_ratio = augment_noise_ratio
        self.expert_data_ratio = expert_data_ratio
        self.expert_data_cutoff =  0
    
    def __len__(self):
        return self.mem_counter

    def can_sample(self, batch_size):
        if self.mem_counter > (batch_size * 500):
            return True
        else:
            return False

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_counter % self.mem_size

        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        if self.expert_data_ratio > 0:
            expert_data_quantity = int(batch_size * self.expert_data_ratio)
            random_batch = np.random.choice(max_mem, batch_size - expert_data_quantity)
            expert_batch = np.random.choice(self.expert_data_cutoff, expert_data_quantity)
            batch = np.concatenate((random_batch, expert_batch))
        else:
            batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        next_states = self.next_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        if self.augment_data:
            state_noise_std = self.augment_noise_ratio * np.mean(np.abs(states))
            action_noise_std = self.augment_noise_ratio * np.mean(np.abs(actions))

            states = states + np.random.normal(0, state_noise_std, states.shape)
            actions = actions + np.random.normal(0, action_noise_std, actions.shape)

        if self.augment_rewards:
            rewards = rewards * 100

        return states, actions, rewards, next_states, dones

    def save_to_csv(self, filename):
        np.savez(filename, 
                state = self.state_memory[:self.mem_counter],
                action = self.action_memory[:self.mem_counter],
                reward = self.reward_memory[:self.mem_counter],
                next_state = self.next_state_memory[:self.mem_counter],
                done = self.terminal_memory[:self.mem_counter])
        print(f"Saved {filename}")

    def load_from_csv(self, filename, expert_data=True):
        try:
            data = np.load(filename)
            self.mem_counter = len(data['state'])
            self.state_memory[:self.mem_counter] = data['state']
            self.action_memory[:self.mem_counter] = data['action']
            self.reward_memory[:self.mem_counter] = data['reward']
            self.next_state_memory[:self.mem_counter] = data['next_state']
            self.terminal_memory[:self.mem_counter] = data['done']
            print(f"Successfully loaded {filename} in the memory.")
            print(f"{self.mem_counter} memories loaded.")

            if expert_data:
                self.expert_data_cutoff = self.mem_counter
        
        except:
            print(f"Unable to load memory from {filename}")