#!/usr/bin/env python3
"""
Example training script for the pipette environment using actor-critic algorithms.
Demonstrates how to integrate the physics simulation with popular RL libraries.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
from collections import deque
import os
import time

from integrated_pipette_environment import IntegratedPipetteEnv

class SimpleActorCritic(nn.Module):
    """Simple Actor-Critic network for pipette control"""
    
    def __init__(self, obs_dim=25, action_dim=4, hidden_size=256):
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor network (policy)
        self.actor_mean = nn.Linear(hidden_size, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Critic network (value function)
        self.critic = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        shared_features = self.shared(x)
        
        # Actor output
        action_mean = torch.tanh(self.actor_mean(shared_features))
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
        
        # Critic output
        value = self.critic(shared_features)
        
        return action_mean, action_std, value
    
    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            action_mean, action_std, value = self.forward(state)
            
            if deterministic:
                action = action_mean
            else:
                dist = Normal(action_mean, action_std)
                action = dist.sample()
            
            # Scale actions to environment bounds
            # [x, y, z, plunger] -> [[-0.4,0.4], [-0.4,0.4], [-0.3,0.1], [0,1]]
            action_scaled = torch.zeros_like(action)
            action_scaled[:, 0] = action[:, 0] * 0.4  # x: -0.4 to 0.4
            action_scaled[:, 1] = action[:, 1] * 0.4  # y: -0.4 to 0.4
            action_scaled[:, 2] = action[:, 2] * 0.2 - 0.1  # z: -0.3 to 0.1
            action_scaled[:, 3] = (action[:, 3] + 1) * 0.5  # plunger: 0 to 1
            
        return action_scaled, value
    
    def evaluate_actions(self, states, actions):
        action_mean, action_std, values = self.forward(states)
        
        # Reverse action scaling for log probability calculation
        actions_unscaled = torch.zeros_like(actions)
        actions_unscaled[:, 0] = actions[:, 0] / 0.4
        actions_unscaled[:, 1] = actions[:, 1] / 0.4
        actions_unscaled[:, 2] = (actions[:, 2] + 0.1) / 0.2
        actions_unscaled[:, 3] = actions[:, 3] * 2 - 1
        
        dist = Normal(action_mean, action_std)
        log_probs = dist.log_prob(actions_unscaled).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return values, log_probs, entropy

class PPOTrainer:
    """PPO trainer for the pipette environment"""
    
    def __init__(self, env, model, lr=3e-4, gamma=0.99, tau=0.95, 
                 clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01):
        self.env = env
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # PPO hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Training statistics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.training_stats = {
            'episode': [],
            'reward': [],
            'length': [],
            'particles_transferred': []
        }
    
    def collect_trajectories(self, num_steps=2048):
        """Collect trajectories for PPO update"""
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        particles_transferred = 0
        
        for step in range(num_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, value = self.model.get_action(state_tensor)
            
            # Execute action
            next_state, reward, done, info = self.env.step(action.numpy()[0])
            
            # Store trajectory data
            states.append(state)
            actions.append(action.numpy()[0])
            rewards.append(reward)
            values.append(value.item())
            dones.append(done)
            
            # Calculate log probability for this action
            with torch.no_grad():
                _, log_prob, _ = self.model.evaluate_actions(state_tensor, action)
                log_probs.append(log_prob.item())
            
            # Update episode statistics
            episode_reward += reward
            episode_length += 1
            
            # Track task completion
            if 'particles_held' in info and info['particles_held'] > particles_transferred:
                particles_transferred = info['particles_held']
            
            # Handle episode end
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                # Log episode statistics
                self.training_stats['episode'].append(len(self.episode_rewards))
                self.training_stats['reward'].append(episode_reward)
                self.training_stats['length'].append(episode_length)
                self.training_stats['particles_transferred'].append(particles_transferred)
                
                # Reset for next episode
                state = self.env.reset()
                episode_reward = 0
                episode_length = 0
                particles_transferred = 0
            else:
                state = next_state
        
        return {
            'states': torch.FloatTensor(states),
            'actions': torch.FloatTensor(actions),
            'rewards': torch.FloatTensor(rewards),
            'values': torch.FloatTensor(values),
            'log_probs': torch.FloatTensor(log_probs),
            'dones': torch.FloatTensor(dones)
        }
    
    def compute_advantages(self, rewards, values, dones):
        """Compute advantages using Generalized Advantage Estimation (GAE)"""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.tau * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update_policy(self, trajectories, num_epochs=4, batch_size=64):
        """Update policy using PPO"""
        states = trajectories['states']
        actions = trajectories['actions']
        old_log_probs = trajectories['log_probs']
        values = trajectories['values']
        rewards = trajectories['rewards']
        dones = trajectories['dones']
        
        # Compute advantages and returns
        advantages, returns = self.compute_advantages(rewards, values, dones)
        
        # PPO update
        for epoch in range(num_epochs):
            # Generate random indices for minibatches
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate current policy
                batch_values, batch_log_probs, batch_entropy = self.model.evaluate_actions(
                    batch_states, batch_actions
                )
                
                # Calculate ratio (π_θ / π_θ_old)
                ratio = torch.exp(batch_log_probs - batch_old_log_probs.unsqueeze(-1))
                
                # Calculate surrogate losses
                surr1 = ratio * batch_advantages.unsqueeze(-1)
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages.unsqueeze(-1)
                
                # Policy loss
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.MSELoss()(batch_values, batch_returns.unsqueeze(-1))
                
                # Entropy loss (for exploration)
                entropy_loss = -batch_entropy.mean()
                
                # Total loss
                total_loss = (policy_loss + 
                             self.value_coef * value_loss + 
                             self.entropy_coef * entropy_loss)
                
                # Update model
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
    
    def train(self, total_timesteps=100000, log_interval=10, save_interval=50):
        """Main training loop"""
        print("Starting PPO training for pipette environment...")
        print(f"Total timesteps: {total_timesteps}")
        
        timesteps_collected = 0
        episode_count = 0
        
        while timesteps_collected < total_timesteps:
            # Collect trajectories
            trajectories = self.collect_trajectories(num_steps=2048)
            timesteps_collected += len(trajectories['states'])
            
            # Update policy
            self.update_policy(trajectories)
            
            episode_count += 1
            
            # Logging
            if episode_count % log_interval == 0:
                avg_reward = np.mean(list(self.episode_rewards)[-10:]) if self.episode_rewards else 0
                avg_length = np.mean(list(self.episode_lengths)[-10:]) if self.episode_lengths else 0
                
                print(f"Episode {episode_count}")
                print(f"  Timesteps: {timesteps_collected}/{total_timesteps}")
                print(f"  Avg Reward (last 10): {avg_reward:.2f}")
                print(f"  Avg Length (last 10): {avg_length:.1f}")
                print(f"  Total Episodes: {len(self.episode_rewards)}")
            
            # Save model
            if episode_count % save_interval == 0:
                self.save_model(f"pipette_model_episode_{episode_count}.pt")
        
        print("Training completed!")
        self.save_model("pipette_model_final.pt")
        self.plot_training_stats()
    
    def save_model(self, filename):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats
        }, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename):
        """Load a trained model"""
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
        print(f"Model loaded from {filename}")
    
    def plot_training_stats(self):
        """Plot training statistics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Episode rewards
        if self.training_stats['reward']:
            axes[0, 0].plot(self.training_stats['episode'], self.training_stats['reward'])
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
        
        # Episode lengths
        if self.training_stats['length']:
            axes[0, 1].plot(self.training_stats['episode'], self.training_stats['length'])
            axes[0, 1].set_title('Episode Lengths')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Steps')
        
        # Particles transferred
        if self.training_stats['particles_transferred']:
            axes[1, 0].plot(self.training_stats['episode'], self.training_stats['particles_transferred'])
            axes[1, 0].set_title('Particles Transferred')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Count')
        
        # Running average reward
        if len(self.training_stats['reward']) > 10:
            window = min(50, len(self.training_stats['reward']) // 10)
            running_avg = np.convolve(self.training_stats['reward'], 
                                    np.ones(window)/window, mode='valid')
            axes[1, 1].plot(range(window-1, len(self.training_stats['reward'])), running_avg)
            axes[1, 1].set_title(f'Running Average Reward (window={window})')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Reward')
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.show()

def evaluate_model(model_path, env, num_episodes=10):
    """Evaluate a trained model"""
    # Load model
    model = SimpleActorCritic()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    total_rewards = []
    success_count = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, _ = model.get_action(state_tensor, deterministic=True)
            
            state, reward, done, info = env.step(action.numpy()[0])
            episode_reward += reward
            episode_length += 1
            
            # Optionally render
            if episode < 3:  # Render first few episodes
                env.render()
                time.sleep(0.01)
            
            if done:
                if info.get('task_completed', False):
                    success_count += 1
                break
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    avg_reward = np.mean(total_rewards)
    success_rate = success_count / num_episodes
    
    print(f"\nEvaluation Results:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Reward Std: {np.std(total_rewards):.2f}")

def main():
    """Main training script"""
    # Create environment
    try:
        env = IntegratedPipetteEnv("particle_pipette_system.xml")
        print("✓ Environment created successfully")
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        return
    
    # Create model
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    model = SimpleActorCritic(obs_dim, action_dim)
    
    print(f"✓ Model created: {obs_dim} obs -> {action_dim} actions")
    
    # Create trainer
    trainer = PPOTrainer(env, model)
    
    # Train model
    try:
        trainer.train(total_timesteps=50000, log_interval=5, save_interval=25)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save_model("pipette_model_interrupted.pt")
    except Exception as e:
        print(f"Training failed: {e}")
        return
    
    # Evaluate final model
    print("\nEvaluating trained model...")
    evaluate_model("pipette_model_final.pt", env, num_episodes=5)
    
    env.close()

if __name__ == "__main__":
    main()