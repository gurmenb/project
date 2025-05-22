# pipetting actor-critic algorithm 
import hydra
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

# define the actor network 35D Observations -> 10 Actions 
class Actor(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_dim, std=0.1):
        super().__init__()

        self.std = std
        self.policy = nn.Sequential(
            # first layer 
            nn.Linear(obs_shape[0], hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),

            # second layer 
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), 
            nn.ReLU(inplace=True),

            # third layer start reducing dimensionals towards output 
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.ReLU(inplace=True),
            
            # output layer 
            nn.Linear(hidden_dim//2, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs):
        mu = self.policy(obs)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * self.std

        dist = utils.TruncatedNormal(mu, std)
        return dist

class Critic(nn.Module):
    def __init__(self, obs_shape, action_shape, num_critics,
                 hidden_dim):
        super().__init__()

        self.critics = nn.ModuleList([nn.Sequential(
            # concatenate observation + action [batch_size, 35+10]
            nn.Linear(obs_shape[0] + action_shape[0], hidden_dim), 
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True), 

            # 512 -> 512
            nn.Linear(hidden_dim, hidden_dim), 
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True), 

            # 512 → 256
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.ReLU(inplace=True),

            # 256 -> 1 Q-value 
            nn.Linear(hidden_dim//2, 1))
            
            for _ in range(num_critics)])

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h_action = torch.cat([obs, action], dim=-1)
        return [critic(h_action) for critic in self.critics]
    
class Reward_Function: 
    def __init__(self, 
                 volume=10.0,      
                 time=0.1,         
                 completion=20.0, 
                 collisions=5.0,    
                 splashes=3.0,      
                 contamination=8.0, 
                 air_bubbles=2.0): 

        self.volume = volume
        self.time = time
        self.completion = completion
        self.collisions = collisions
        self.splashes = splashes
        self.contamination = contamination
        self.air_bubbles = air_bubbles

def Compute_Reward(self, state, action, next_state, target_volume = 100):
    
    # get state information from our 35D oberservation vector
    volume = next_state[20]
    tip_position = next_state[0:3] 
    container_boundaries = next_state[21:24]
    liquid_properties = next_state[24:27]
    force_
    
    total_reward = volume_reward + time_reward + completion_reward + collisions_reward 
                   splashes_reward + contamination_reward + air_bubbles_reward
    return total_reward


# functions in hw2 needed to implement 
class Actor(nn.Module):
class Critic(nn.Module):
class ACAgent:
def update_critic(self, replay_iter):
def update_actor(self, replay_iter):
def bc(self, replay_iter):



