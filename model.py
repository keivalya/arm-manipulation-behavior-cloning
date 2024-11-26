import torch
import torch.nn as nn
import torch.nn. functional as F
from torch.distributions import Normal
import os

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant(m.bias, 0)

class Actor(nn.Module):
    def __init__(self, n_inputs, n_actions, hidden_dim, action_space=None, checkpoint_dir='checkpoints',name='policy_network'):
        super(Actor, self).__init__()

        self.linear1 = nn.Linear(n_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, n_actions)
        self.log_std_linear = nn.Linear(hidden_dim, n_actions)

        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        