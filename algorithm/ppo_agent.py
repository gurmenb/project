import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


class Actor(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_dim, std=0.1):
        super().__init__()

        self.std = std
        self.policy = nn.Sequential(
            nn.Linear(obs_shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_shape[0])
        )

        self.apply(utils.weight_init)

    def forward(self, obs):
        mu = self.policy(obs)
        mu = torch.tanh(mu) 
        std = torch.ones_like(mu) * self.std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, obs_shape, hidden_dim):
        super().__init__()

        # PPO critic takes only observations
        self.value_net = nn.Sequential(
            nn.Linear(obs_shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1) 
        )

        self.apply(utils.weight_init)

    def forward(self, obs):
        return self.value_net(obs)


class PPOAgent:
    def __init__(self, obs_shape, action_shape, device, lr, hidden_dim, 
                 clip_ratio=0.2, stddev_clip=0.3, use_tb=True):
        self.device = device
        self.clip_ratio = clip_ratio
        self.stddev_clip = stddev_clip
        self.use_tb = use_tb

        # Create models 
        self.actor = Actor(obs_shape, action_shape, hidden_dim).to(device)
        self.critic = Critic(obs_shape, hidden_dim).to(device)

        # Optimizers 
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.train()

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

    def get_value(self, obs):
        """Get value estimate (new for PPO)"""
        obs = torch.as_tensor(obs, device=self.device)
        with torch.no_grad():
            value = self.critic(obs.unsqueeze(0))
        return value.cpu().numpy()[0, 0]

    def update_critic(self, batch_data):
        """
        Update critic network (HW2-style function signature)
        
        Args:
        batch_data: Dictionary containing:
            - observations: array of shape [batch, obs_dim]
            - returns: array of shape [batch,] (computed returns)
        
        Returns:
        metrics: dictionary of metrics for logging
        """
        metrics = dict()

        # Extract data from batch
        obs = torch.as_tensor(batch_data['observations'], device=self.device)
        returns = torch.as_tensor(batch_data['returns'], device=self.device)

        # Compute current value estimates
        current_values = self.critic(obs).squeeze(-1)

        # Critic loss (MSE between predicted values and actual returns)
        critic_loss = F.mse_loss(current_values, returns)

        # Update critic
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        metrics['critic_loss'] = critic_loss.item()
        metrics['critic_target'] = returns.mean().item()

        return metrics

    def update_actor(self, batch_data):
        """
        Update actor network with PPO objective (HW2-style function signature)
        
        Args:
        batch_data: Dictionary containing:
            - observations: array of shape [batch, obs_dim]
            - actions: array of shape [batch, action_dim]
            - advantages: array of shape [batch,] (computed advantages)
            - old_log_probs: array of shape [batch,] (log probs from data collection)
        
        Returns:
        metrics: dictionary of metrics for logging
        """
        metrics = dict()

        # Extract data from batch
        obs = torch.as_tensor(batch_data['observations'], device=self.device)
        actions = torch.as_tensor(batch_data['actions'], device=self.device)
        advantages = torch.as_tensor(batch_data['advantages'], device=self.device)
        old_log_probs = torch.as_tensor(batch_data['old_log_probs'], device=self.device)

        # Normalize advantages 
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Get current policy distribution
        action_distribution = self.actor(obs)
        
        # Compute log probabilities of the actions under current policy
        new_log_probs = action_distribution.log_prob(actions).sum(-1)

        # PPO algorithm 
        # Prob ratio between new and old policy 
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        # clip between 0.8 and 1.2 so the policy does not change too much in one update step
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        
        # PPO loss (take minimum for conservative policy update)
        actor_loss = -torch.min(surr1, surr2).mean()

        # Update actor
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        metrics['actor_loss'] = actor_loss.item()
        metrics['policy_ratio'] = ratio.mean().item()

        return metrics

    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions under current policy (for collecting data)
        
        Args:
        obs: observations
        actions: actions taken
        
        Returns:
        log_probs: log probabilities of actions
        values: state values
        """
        # Get action distribution
        dist = self.actor(obs)
        
        # Get log probabilities and values
        log_probs = dist.log_prob(actions).sum(-1)
        values = self.critic(obs).squeeze(-1)
        
        return log_probs, values

    def ppo_update(self, buffer_data, ppo_epochs=10, batch_size=64):
        """
        Complete PPO update (combines critic and actor updates)
        
        Args:
        buffer_data: Dictionary with all collected data
        ppo_epochs: Number of epochs to train on the data
        batch_size: Mini-batch size
        
        Returns:
        metrics: Combined metrics from both updates
        """
        all_metrics = {'actor_loss': 0, 'critic_loss': 0, 'policy_ratio': 0, 'critic_target': 0}
        
        # Convert data to tensors
        obs = torch.as_tensor(buffer_data['observations'], device=self.device)
        actions = torch.as_tensor(buffer_data['actions'], device=self.device)
        returns = torch.as_tensor(buffer_data['returns'], device=self.device)
        advantages = torch.as_tensor(buffer_data['advantages'], device=self.device)
        old_log_probs = torch.as_tensor(buffer_data['old_log_probs'], device=self.device)
        
        dataset_size = len(obs)
        
        # Multiple epochs over the same data (PPO's key feature)
        for epoch in range(ppo_epochs):
            # Shuffle data
            indices = torch.randperm(dataset_size)
            
            # Mini-batch updates
            for start in range(0, dataset_size, batch_size):
                end = min(start + batch_size, dataset_size)
                batch_indices = indices[start:end]
                
                # Create mini-batch
                batch = {
                    'observations': obs[batch_indices],
                    'actions': actions[batch_indices],
                    'returns': returns[batch_indices],
                    'advantages': advantages[batch_indices],
                    'old_log_probs': old_log_probs[batch_indices]
                }
                
                # Update critic
                critic_metrics = self.update_critic(batch)
                
                # Update actor  
                actor_metrics = self.update_actor(batch)
                
                # Accumulate metrics
                for key in all_metrics:
                    if key in critic_metrics:
                        all_metrics[key] += critic_metrics[key]
                    if key in actor_metrics:
                        all_metrics[key] += actor_metrics[key]
        
        # Average metrics over all updates
        num_updates = ppo_epochs * ((dataset_size + batch_size - 1) // batch_size)
        for key in all_metrics:
            all_metrics[key] /= num_updates
            
        return all_metrics

    def bc(self, replay_iter):
        """
        Behavior cloning update (kept from HW2 for compatibility)
        """
        metrics = dict()
        batch = next(replay_iter)

        obs, action, _, _, _ = utils.to_torch(batch, self.device)
        action_distribution = self.actor(obs)

        # Behavior cloning loss
        loss_function = -action_distribution.log_prob(action).sum(-1, keepdim=True).mean()

        self.actor_opt.zero_grad()
        loss_function.backward()
        self.actor_opt.step()

        metrics['actor_loss'] = loss_function.item()
        return metrics