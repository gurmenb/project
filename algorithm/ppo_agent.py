import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import random

from actor_critic import Actor, Critic, PipettingDimensions
import utils


class PPOAgent:
    """Proximal Policy Optimization agent for pipetting task."""
    
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        lr,
        critic_lr,
        hidden_dim,
        num_layers,
        num_critics,
        critic_target_tau,
        stddev_clip,
        use_tb,
        ppo_epoch,
        ppo_clip,
        value_loss_coef,
        entropy_coef,
        max_grad_norm,
        gae_lambda,
        std_min,
        std_max,
        **kwargs
    ):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.stddev_clip = stddev_clip
        self.use_tb = use_tb
        
        # PPO specific parameters
        self.ppo_epoch = ppo_epoch
        self.ppo_clip = ppo_clip
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda
        
        # Create networks
        self.actor = Actor(
            obs_shape[0], 
            action_shape[0],
            hidden_dim, 
            num_layers,
            std_min,
            std_max
        ).to(device)
        
        self.critic = Critic(
            obs_shape[0],
            action_shape[0],
            num_critics,
            hidden_dim,
            num_layers
        ).to(device)
        
        self.critic_target = Critic(
            obs_shape[0],
            action_shape[0],
            num_critics,
            hidden_dim,
            num_layers
        ).to(device)
        
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.train()
        self.critic_target.train()
        
    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        
    def act(self, obs, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        dist = self.actor(obs.unsqueeze(0))
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=self.stddev_clip)
        return action.cpu().numpy()[0]
    
    def compute_gae(
        self, 
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 0.99
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        returns = advantages + values
        return advantages, returns
    
    def update_ppo(self, replay_buffer):
        """Update actor and critic using PPO."""
        metrics = dict()
        
        # Sample batch from replay buffer
        batch = replay_buffer.sample()
        obs, actions, rewards, dones, next_obs = utils.to_torch(batch, self.device)
        
        # Get old log probs and values
        with torch.no_grad():
            old_dist = self.actor(obs)
            old_log_probs = old_dist.log_prob(actions).sum(-1, keepdim=True)
            old_values = self.critic.q_min(obs, actions)
            next_values = self.critic_target.q_min(next_obs, actions)
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, old_values, next_values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update loop
        for epoch in range(self.ppo_epoch):
            # Actor update
            dist = self.actor(obs)
            log_probs = dist.log_prob(actions).sum(-1, keepdim=True)
            entropy = dist.entropy().sum(-1, keepdim=True).mean()
            
            # PPO surrogate loss
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
            
            self.actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_opt.step()
            
            # Critic update
            values = self.critic.q_min(obs, actions)
            value_loss = F.mse_loss(values, returns)
            
            self.critic_opt.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_opt.step()
        
        # Update target critic
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)
        
        # Logging
        metrics['actor_loss'] = actor_loss.item()
        metrics['value_loss'] = value_loss.item()
        metrics['entropy'] = entropy.item()
        metrics['advantages'] = advantages.mean().item()
        metrics['ppo_ratio'] = ratio.mean().item()
        
        return metrics
    
    def update_critic(self, replay_iter):
        """Update critic using TD learning (for compatibility with HW2 structure)."""
        metrics = dict()
        
        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)
        
        # Compute Bellman targets
        with torch.no_grad():
            next_action_dist = self.actor(next_obs)
            next_action = next_action_dist.sample(clip=self.stddev_clip)
            
            target_Q_values = self.critic_target(next_obs, next_action)
            
            # Randomly select two critics for double Q-learning
            if len(target_Q_values) >= 2:
                idx1, idx2 = random.sample(range(len(target_Q_values)), 2)
                target_Q = torch.min(target_Q_values[idx1], target_Q_values[idx2])
            else:
                target_Q = target_Q_values[0]
            
            y = reward + discount * target_Q
        
        # Update all critics
        current_Q_values = self.critic(obs, action)
        critic_loss = 0
        for q in current_Q_values:
            critic_loss += F.mse_loss(q, y.detach())
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)
        
        metrics['critic_loss'] = critic_loss.item()
        metrics['critic_target'] = y.mean().item()
        
        return metrics
    
    def update_actor(self, replay_iter):
        """Update actor using policy gradient (for compatibility)."""
        metrics = dict()
        
        batch = next(replay_iter)
        obs, _, _, _, _ = utils.to_torch(batch, self.device)
        
        dist = self.actor(obs)
        actions = dist.sample(clip=self.stddev_clip)
        log_probs = dist.log_prob(actions).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True)
        
        q_values = self.critic(obs, actions)
        q_value = torch.stack(q_values, dim=0).mean(dim=0)
        
        actor_loss = -(q_value + self.entropy_coef * entropy).mean()
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_opt.step()
        
        metrics['actor_loss'] = actor_loss.item()
        metrics['entropy'] = entropy.mean().item()
        
        return metrics
    
    def save(self, path):
        """Save model parameters."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
        }, path)
    
    def load(self, path):
        """Load model parameters."""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_opt.load_state_dict(checkpoint['actor_opt'])
        self.critic_opt.load_state_dict(checkpoint['critic_opt'])