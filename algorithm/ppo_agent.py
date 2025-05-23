import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import utils


class Actor(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_dim, log_std_bounds=(-20, 2)):
        super().__init__()
        
        # Handle discrete task_phase by expanding observation size
        # obs_shape[0] assumes task_phase is one-hot encoded (4 values -> 4 dims)
        obs_dim = obs_shape[0] - 1 + 4  # Remove 1D task_phase, add 4D one-hot
        
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_shape[0] * 2)  # mean and log_std
        )
        
        self.log_std_bounds = log_std_bounds
        self.apply(utils.weight_init)

    def forward(self, obs):
        # Process observation - one-hot encode task_phase
        obs_processed = self._process_observation(obs)
        
        output = self.policy(obs_processed)
        mu, log_std = output.chunk(2, dim=-1)
        
        # Bound log_std
        log_std = torch.clamp(log_std, self.log_std_bounds[0], self.log_std_bounds[1])
        std = log_std.exp()
        
        dist = Normal(mu, std)
        return dist
    
    def _process_observation(self, obs):
        """Convert task_phase to one-hot and concatenate with other observations"""
        # Assuming obs structure: [liquid_in_plunger, balls_in_plunger, source_amount, 
        # target_amount, source_pos(3), target_pos(3), asp_pressure, disp_pressure, 
        # task_phase, submerged]
        
        batch_size = obs.shape[0] if obs.dim() > 1 else 1
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        # Extract task_phase (assuming it's at index 12)
        task_phase = obs[:, 12].long()
        
        # Create one-hot encoding for task_phase (4 phases: 0,1,2,3)
        task_phase_onehot = F.one_hot(task_phase, num_classes=4).float()
        
        # Concatenate everything except original task_phase
        obs_without_phase = torch.cat([obs[:, :12], obs[:, 13:]], dim=1)
        processed_obs = torch.cat([obs_without_phase, task_phase_onehot], dim=1)
        
        if squeeze_output and batch_size == 1:
            processed_obs = processed_obs.squeeze(0)
            
        return processed_obs


class Critic(nn.Module):
    def __init__(self, obs_shape, hidden_dim):
        super().__init__()
        
        # Same observation processing as Actor
        obs_dim = obs_shape[0] - 1 + 4  # One-hot encoded task_phase
        
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        
        self.apply(utils.weight_init)

    def forward(self, obs):
        obs_processed = self._process_observation(obs)
        return self.value_net(obs_processed)
    
    def _process_observation(self, obs):
        """Same observation processing as Actor"""
        batch_size = obs.shape[0] if obs.dim() > 1 else 1
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        task_phase = obs[:, 12].long()
        task_phase_onehot = F.one_hot(task_phase, num_classes=4).float()
        obs_without_phase = torch.cat([obs[:, :12], obs[:, 13:]], dim=1)
        processed_obs = torch.cat([obs_without_phase, task_phase_onehot], dim=1)
        
        if squeeze_output and batch_size == 1:
            processed_obs = processed_obs.squeeze(0)
            
        return processed_obs


class PPOAgent:
    def __init__(self, obs_shape, action_shape, device, lr, hidden_dim, 
                 clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01, 
                 max_grad_norm=0.5, ppo_epochs=10, batch_size=64):
        
        self.device = device
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        # Networks
        self.actor = Actor(obs_shape, action_shape, hidden_dim).to(device)
        self.critic = Critic(obs_shape, hidden_dim).to(device)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        self.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, eval_mode=False):
        """Sample action from policy"""
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        
        with torch.no_grad():
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            
            dist = self.actor(obs)
            
            if eval_mode:
                action = dist.mean
            else:
                action = dist.sample()
            
            # Clamp actions to [-1, 1] range
            action = torch.clamp(action, -1.0, 1.0)
            
        return action.cpu().numpy()[0]

    def get_value(self, obs):
        """Get value estimate from critic"""
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        
        with torch.no_grad():
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            value = self.critic(obs)
        
        return value.cpu().numpy()[0, 0]

    def evaluate_actions(self, obs, actions):
        """Evaluate actions for PPO update"""
        dist = self.actor(obs)
        values = self.critic(obs)
        
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return values, log_probs, entropy

    def update(self, buffer):
        """PPO update using collected buffer data"""
        
        # Get all data from buffer
        obs_batch = torch.as_tensor(buffer['observations'], device=self.device, dtype=torch.float32)
        actions_batch = torch.as_tensor(buffer['actions'], device=self.device, dtype=torch.float32)
        returns_batch = torch.as_tensor(buffer['returns'], device=self.device, dtype=torch.float32)
        advantages_batch = torch.as_tensor(buffer['advantages'], device=self.device, dtype=torch.float32)
        old_log_probs_batch = torch.as_tensor(buffer['log_probs'], device=self.device, dtype=torch.float32)
        
        # Normalize advantages
        advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
        
        # PPO update for multiple epochs
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        for epoch in range(self.ppo_epochs):
            # Create mini-batches
            batch_size = min(self.batch_size, len(obs_batch))
            indices = torch.randperm(len(obs_batch))
            
            for start in range(0, len(obs_batch), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                obs_mini = obs_batch[batch_indices]
                actions_mini = actions_batch[batch_indices]
                returns_mini = returns_batch[batch_indices]
                advantages_mini = advantages_batch[batch_indices]
                old_log_probs_mini = old_log_probs_batch[batch_indices]
                
                # Evaluate current policy
                values, log_probs, entropy = self.evaluate_actions(obs_mini, actions_mini)
                
                # PPO actor loss
                ratio = torch.exp(log_probs - old_log_probs_mini)
                surr1 = ratio * advantages_mini
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_mini
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic loss - FIX: Ensure shapes match
                values_flat = values.squeeze(-1)  # Remove last dimension if present
                critic_loss = F.mse_loss(values_flat, returns_mini)
                
                # Total loss
                entropy_loss = -entropy.mean()
                
                # Update actor
                self.actor_optimizer.zero_grad()
                actor_total_loss = actor_loss + self.entropy_coef * entropy_loss
                actor_total_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
        
        # Return metrics for logging
        num_updates = self.ppo_epochs * (len(obs_batch) // batch_size + (1 if len(obs_batch) % batch_size != 0 else 0))
        metrics = {
            'loss': total_actor_loss / num_updates,  # Changed from 'actor_loss' to 'loss'
            'value_loss': total_critic_loss / num_updates,  # Changed from 'critic_loss' to 'value_loss'
            'policy_entropy': total_entropy / num_updates,  # Changed from 'entropy' to 'policy_entropy'
        }
        
        return metrics