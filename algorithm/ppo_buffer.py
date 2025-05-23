import numpy as np
import torch
from collections import defaultdict


class PPOBuffer:
    """
    Buffer for collecting trajectories and computing advantages for PPO
    """
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        
        self.gamma = gamma
        self.lam = lam
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = size
        
    def store(self, obs, act, rew, val, logp):
        """
        Store one timestep of agent-environment interaction
        """
        assert self.ptr < self.max_size
        
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        
        self.ptr += 1
    
    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by a timeout. This looks back in the buffer to where the trajectory
        started, and uses rewards and value estimates from the whole trajectory
        to compute advantage estimates with GAE-Lambda, as well as compute
        the rewards-to-go for each state, to use as the targets for the value function.
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self._discount_cumsum(deltas, self.gamma * self.lam)
        
        # Compute rewards-to-go (targets for value function)
        self.ret_buf[path_slice] = self._discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr
    
    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        
        # Normalize advantages
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        
        data = dict(
            observations=self.obs_buf,
            actions=self.act_buf,
            returns=self.ret_buf,
            advantages=self.adv_buf,
            log_probs=self.logp_buf
        )
        return data
    
    def _discount_cumsum(self, x, discount):
        """
        Magic from rllab for computing discounted cumulative sums of vectors.
        Input: vector x = [x0, x1, x2]
        Output: [x0 + discount * x1 + discount^2 * x2,  
                 x1 + discount * x2,
                 x2]
        """
        return np.array(list(reversed(np.cumsum(list(reversed(x * discount ** np.arange(len(x))))))))


class SimpleBuffer:
    """
    Simpler buffer for collecting data during training
    Used when you don't need full PPO buffer functionality
    """
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        
    def store(self, obs, action, reward, done, value=None, log_prob=None):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        if value is not None:
            self.values.append(value)
        if log_prob is not None:
            self.log_probs.append(log_prob)
    
    def compute_returns_and_advantages(self, last_value=0, gamma=0.99, lam=0.95):
        """
        Compute returns and advantages using GAE
        """
        values = np.array(self.values + [last_value])
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        
        # Compute returns
        returns = []
        ret = last_value
        for i in reversed(range(len(rewards))):
            ret = rewards[i] + gamma * ret * (1 - dones[i])
            returns.insert(0, ret)
        
        # Compute advantages using GAE
        advantages = []
        adv = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values[i + 1] * (1 - dones[i]) - values[i]
            adv = delta + gamma * lam * (1 - dones[i]) * adv
            advantages.insert(0, adv)
        
        return np.array(returns), np.array(advantages)
    
    def get_batch(self, normalize_advantages=True):
        """
        Get all data as a batch for training
        """
        returns, advantages = self.compute_returns_and_advantages()
        
        if normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return {
            'observations': np.array(self.observations),
            'actions': np.array(self.actions),
            'returns': returns,
            'advantages': advantages,
            'log_probs': np.array(self.log_probs) if self.log_probs else None,
            'rewards': np.array(self.rewards)
        }
    
    def clear(self):
        """Clear all stored data"""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.log_probs.clear()
    
    def __len__(self):
        return len(self.observations)