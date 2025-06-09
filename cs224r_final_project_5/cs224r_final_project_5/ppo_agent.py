import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from utils import weight_init
from ppo_buffer import PPOBuffer

# Define the actor network
class Actor(nn.Module):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256)):
        super().__init__()
        
        # Build the main network
        sizes = [obs_dim] + list(hidden_sizes)
        layers = []
        for i in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            layers.append(nn.ReLU())
        
        self.net = nn.Sequential(*layers)
        
        # Output layers
        self.mu_layer = nn.Linear(sizes[-1], act_dim)  # Mean of actions
        self.log_std = nn.Parameter(torch.ones(act_dim) * -0.5)  # Learnable log std
        
        # Initialize weights
        self.apply(weight_init)
    
    def forward(self, obs):
        """
        Forward pass through network.
        
        Args:
            obs: Observations tensor [batch_size, obs_dim]
            
        Returns:
            Normal distribution over actions
        """
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        std = torch.exp(self.log_std)
        return Normal(mu, std)
    
    def act(self, obs):
        """
        Sample action from policy (for environment interaction).
        
        Args:
            obs: Single observation tensor [obs_dim] 
            
        Returns:
            action: Action array [act_dim]
            logp: Log probability of action
        """
        with torch.no_grad():
            dist = self.forward(obs)
            action = dist.sample()
            logp = dist.log_prob(action).sum(axis=-1)
        return action.numpy(), logp.numpy()


class Critic(nn.Module):
    """
    PPO Critic (Value Function) Network.
    Estimates state values for advantage computation.
    """
    def __init__(self, obs_dim, hidden_sizes=(256, 256)):
        super().__init__()
        
        # Build value network
        sizes = [obs_dim] + list(hidden_sizes) + [1]
        layers = []
        for i in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes)-2:  # No activation on final layer
                layers.append(nn.ReLU())
        
        self.v_net = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(weight_init)
    
    def forward(self, obs):
        """
        Forward pass through value network.
        
        Args:
            obs: Observations tensor [batch_size, obs_dim]
            
        Returns:
            values: State values [batch_size]
        """
        return torch.squeeze(self.v_net(obs), -1)


class PPOAgent:
    """
    Proximal Policy Optimization agent for continuous control.
    Designed to work with PipetteEnv and config file.
    """
    def __init__(self, cfg):
        # Extract config parameters
        self.obs_dim = cfg.agent.obs_dim
        self.act_dim = cfg.agent.act_dim
        self.lr = cfg.lr
        self.clip_ratio = cfg.agent.clip_ratio
        self.target_kl = cfg.agent.target_kl
        self.entropy_coef = cfg.agent.entropy_coef
        self.hidden_sizes = cfg.agent.hidden_sizes
        
        # Training parameters
        self.gamma = cfg.discount
        self.lam = cfg.agent.gae_lambda
        self.steps_per_epoch = cfg.steps_per_epoch
        self.train_iters = cfg.train_iterations
        
        # Create networks
        self.actor = Actor(self.obs_dim, self.act_dim, tuple(self.hidden_sizes))
        self.critic = Critic(self.obs_dim, tuple(self.hidden_sizes))
        
        # Create optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        
        # Create experience buffer
        self.buffer = PPOBuffer(self.obs_dim, self.act_dim, self.steps_per_epoch, 
                               self.gamma, self.lam)
        
        # For logging training stats
        self.training_stats = {}
    
    def act(self, obs):

        # Convert observation to tensor
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        
        # Get action from policy
        action, logp = self.actor.act(obs_tensor)
        
        # Get value estimate
        with torch.no_grad():
            value = self.critic(obs_tensor).item()
        
        return action[0], value, logp[0]
    
    def store_experience(self, obs, act, rew, val, logp):
        self.buffer.store(obs, act, rew, val, logp)
    
    def finish_episode(self, last_val=0):
        """Call at end of episode to compute advantages."""
        self.buffer.finish_path(last_val)
    
    def ready_to_update(self):
        """Check if buffer is full and ready for training."""
        return self.buffer.is_full()
    
    def update(self):
        """
        Update actor and critic networks using PPO.
        Call this when buffer is full.
        """
        # Get all data from buffer
        data = self.buffer.get()
        
        obs, act = data['obs'], data['act']
        old_logp = data['logp']
        
        # Training loop
        for i in range(self.train_iters):
            
            # === ACTOR UPDATE ===
            # Get current policy distribution
            dist = self.actor(obs)
            logp = dist.log_prob(act).sum(axis=-1)
            
            # PPO ratio
            ratio = torch.exp(logp - old_logp)
            
            # PPO clipped objective
            clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * data['adv']
            actor_loss = -(torch.min(ratio * data['adv'], clip_adv)).mean()
            
            # Add entropy bonus for exploration (using config value)
            entropy = dist.entropy().sum(axis=-1).mean()
            actor_loss -= self.entropy_coef * entropy
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # === EARLY STOPPING CHECK ===
            with torch.no_grad():
                kl = (old_logp - logp).mean().item()
                if kl > 1.5 * self.target_kl:
                    break
            
            # === CRITIC UPDATE ===
            values = self.critic(obs)
            critic_loss = F.mse_loss(values, data['ret'])
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
        
        # Store training statistics
        with torch.no_grad():
            clip_fraction = ((ratio > (1 + self.clip_ratio)) | 
                           (ratio < (1 - self.clip_ratio))).float().mean().item()
        
        self.training_stats = {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(), 
            'kl_divergence': kl,
            'entropy': entropy.item(),
            'clip_fraction': clip_fraction,
            'training_iterations': i + 1
        }
    
    def get_training_stats(self):
        """Get training statistics for logging."""
        return self.training_stats.copy()
    
    def save(self, filepath):
        """Save agent to file."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filepath)
        print(f"Agent saved to {filepath}")
    
    def load(self, filepath):
        """Load agent from file."""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        print(f"Agent loaded from {filepath}")


# Quick integration test with pipette environment
if __name__ == "__main__":
    from pipette_env import PipetteEnv
    from omegaconf import OmegaConf
    
    # Load config for testing
    cfg = OmegaConf.create({
        'agent': {
            'obs_dim': 14,
            'act_dim': 4,
            'clip_ratio': 0.2,
            'target_kl': 0.01,
            'gae_lambda': 0.95,
            'entropy_coef': 0.01,
            'hidden_sizes': [256, 256]
        },
        'lr': 3e-4,
        'discount': 0.99,
        'steps_per_epoch': 100,  # Small for testing
        'train_iterations': 5,
        'env': {
            'max_episode_steps': 200,
            'environment_size': 10.0,
            'well_radius': 1.0,
            'well_depth': 2.0
        }
    })
    
    # Create environment and agent
    env = PipetteEnv(cfg)
    agent = PPOAgent(cfg)
    
    print("Testing PPO Agent with Pipette Environment...")
    
    # Test episode
    obs, _ = env.reset()
    total_reward = 0
    
    for step in range(10):
        # Agent selects action
        action, value, logp = agent.act(obs)
        
        # Environment step
        next_obs, reward, terminated, truncated, _ = env.step(action)
        
        # Store experience
        agent.store_experience(obs, action, reward, value, logp)
        
        total_reward += reward
        obs = next_obs
        
        print(f"Step {step}: reward={reward:.2f}, value={value:.2f}")
        
        if terminated or truncated:
            agent.finish_episode()
            break
    
    print(f"Episode total reward: {total_reward:.2f}")
    print(f"Buffer size: {agent.buffer.size()}")
    print("Integration test completed successfully!")