import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
import numpy as np
import torch
from pathlib import Path
from collections import deque

# Your project imports
from ppo_agent import PPOAgent
from ppo_buffer import PPOBuffer, SimpleBuffer
from reward_functions import PipettingRewardFunction
import utils

# For now, we'll create a dummy environment that your partner can replace
class DummyPipettingEnv:
    """
    Dummy environment for testing - replace with your partner's MuJoCo environment
    """
    def __init__(self):
        self.obs_dim = 14
        self.act_dim = 6
        self.max_episode_steps = 200
        self.current_step = 0
        self.target_volume = 20
        
        # Dummy state
        self.source_amount = 50
        self.target_amount = 0
        self.balls_in_plunger = 0
        self.task_phase = 0  # 0=approach, 1=aspirate, 2=transfer, 3=dispense
        
    def reset(self):
        self.current_step = 0
        self.source_amount = 50
        self.target_amount = 0
        self.balls_in_plunger = 0
        self.task_phase = 0
        
        return self._get_observation()
    
    def step(self, action):
        self.current_step += 1
        
        # Dummy dynamics - replace with actual pipetting physics
        # Simulate task progression based on actions (more realistic)
        
        # Extract action components
        pipette_pos = action[:3]  # x, y, z position
        plunger_pos = action[3]   # plunger position
        plunger_force = action[4] # plunger force
        plunger_speed = action[5] # plunger speed
        
        # Simple state machine with action-dependent transitions
        if self.current_step < 50:
            self.task_phase = 0  # approach
        elif self.current_step < 100:
            self.task_phase = 1  # aspirate
            # Simulate aspiration based on plunger action
            if plunger_pos < -0.5 and plunger_force > 0.3:  # Pulling plunger out
                if self.balls_in_plunger == 0:  # Only aspirate once
                    transfer_amount = min(self.target_volume, self.source_amount)
                    # Add some randomness based on action quality
                    noise = np.random.normal(0, 2)
                    actual_transfer = max(0, min(50, transfer_amount + noise))
                    self.balls_in_plunger = actual_transfer
                    self.source_amount -= actual_transfer
        elif self.current_step < 150:
            self.task_phase = 2  # transfer
        else:
            self.task_phase = 3  # dispense
            # Simulate dispensing based on plunger action
            if plunger_pos > 0.5 and plunger_force > 0.3:  # Pushing plunger in
                if self.balls_in_plunger > 0:  # Only dispense if have balls
                    # Check if positioned over target (simplified)
                    target_pos = np.array([0.2, 0.5, 0.1])
                    pipette_distance = np.linalg.norm(pipette_pos[:2] - target_pos[:2])
                    
                    if pipette_distance < 0.15:  # Close enough to target
                        self.target_amount += self.balls_in_plunger
                        self.balls_in_plunger = 0
                    else:
                        # Missed the target - balls fall somewhere else
                        self.balls_in_plunger = 0
        
        obs = self._get_observation()
        done = self.current_step >= self.max_episode_steps
        
        # Simple environment reward (your custom reward will be computed separately)
        reward = 0
        if done:
            # Success if close to target volume
            reward = 10 if abs(self.target_amount - self.target_volume) <= 3 else -5
        
        # Add some intermediate rewards to help learning
        if self.task_phase == 1 and self.balls_in_plunger > 0:
            reward += 1  # Good, aspirated something
        if self.task_phase == 3 and self.target_amount > 0:
            reward += 2  # Good, dispensed something
        
        info = {
            'success': abs(self.target_amount - self.target_volume) <= 3 if done else False,
            'collision': False,  # Could add collision detection later
            'task_phase': self.task_phase,
            'balls_transferred': self.target_amount
        }
        
        return obs, reward, done, info
    
    def _get_observation(self):
        """
        Return 14D observation:
        [liquid_in_plunger, balls_in_plunger, source_amount, target_amount,
         source_x, source_y, source_z, target_x, target_y, target_z,
         aspiration_pressure, dispersion_pressure, task_phase, submerged]
        """
        obs = np.array([
            float(self.balls_in_plunger > 0),  # liquid_in_plunger
            float(self.balls_in_plunger),      # balls_in_plunger
            float(self.source_amount),         # source_amount
            float(self.target_amount),         # target_amount
            0.0, 0.5, 0.1,                    # source position
            0.2, 0.5, 0.1,                    # target position
            1.0,                              # aspiration_pressure
            1.0,                              # dispersion_pressure
            float(self.task_phase),           # task_phase
            0.0                               # submerged
        ], dtype=np.float32)
        
        return obs


class PipettingTrainer:
    def __init__(self, config):
        self.config = config
        device_name = config.get('device', 'auto')
        if device_name == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_name)
        
        # Set seeds
        utils.set_seed_everywhere(config.get('seed', 42))
        
        # Create environment (replace with your partner's environment)
        self.env = DummyPipettingEnv()
        
        # Create agent
        obs_shape = (self.env.obs_dim,)
        action_shape = (self.env.act_dim,)
        
        self.agent = PPOAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=self.device,
            lr=config.get('learning_rate', 3e-4),
            hidden_dim=config.get('hidden_dim', 256),
            clip_ratio=config.get('clip_ratio', 0.2),
            value_coef=config.get('value_coef', 0.5),
            entropy_coef=config.get('entropy_coef', 0.01),
            ppo_epochs=config.get('ppo_epochs', 10),
            batch_size=config.get('batch_size', 64)
        )
        
        # Create reward function
        self.reward_fn = PipettingRewardFunction(
            target_volume=config.get('target_volume', 20),
            w_volume=config.get('w_volume', 10.0),
            w_time=config.get('w_time', 0.1),
            w_completion=config.get('w_completion', 5.0),
            w_collision=config.get('w_collision', 2.0),
            w_drop=config.get('w_drop', 10.0),
            w_contamination=config.get('w_contamination', 3.0),
            w_miss=config.get('w_miss', 5.0),
            w_jerk=config.get('w_jerk', 1.0)
        )
        
        # Create buffer
        self.buffer_size = config.get('buffer_size', 2048)
        self.buffer = SimpleBuffer()
        
        # Simple logging (no CSV logger)
        self.work_dir = Path.cwd()
        
        # Training parameters
        self.num_episodes = config.get('num_episodes', 100)
        self.eval_frequency = config.get('eval_frequency', 10)
        self.save_frequency = config.get('save_frequency', 25)
        
        # Metrics tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.success_rate = deque(maxlen=100)
        
        self.global_step = 0
        self.episode_count = 0

    def collect_trajectories(self):
        """Collect trajectories for PPO update"""
        self.buffer.clear()
        episode_count = 0
        
        while len(self.buffer) < self.buffer_size:
            obs = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            # Reset reward function for new episode
            self.reward_fn.reset()
            
            while not done and len(self.buffer) < self.buffer_size:
                # Get action and value
                action = self.agent.act(obs, eval_mode=False)
                value = self.agent.get_value(obs)
                
                # Get log probability for this action
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
                    action_tensor = torch.as_tensor(action, device=self.device, dtype=torch.float32).unsqueeze(0)
                    dist = self.agent.actor(obs_tensor)
                    log_prob = dist.log_prob(action_tensor).sum(dim=-1).item()
                
                # Take step in environment
                next_obs, env_reward, done, info = self.env.step(action)
                
                # Compute custom reward
                reward = self.reward_fn.compute_reward(obs, action, info, done)
                
                # Store in buffer
                self.buffer.store(obs, action, reward, done, value, log_prob)
                
                obs = next_obs
                episode_reward += reward
                episode_length += 1
                self.global_step += 1
            
            # End of episode
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.success_rate.append(float(info.get('success', False)))
            episode_count += 1
        
        return episode_count

    def train_step(self):
        """Perform one training step with collected data"""
        # Get batch data
        batch_data = self.buffer.get_batch()
        
        # Update agent
        metrics = self.agent.update(batch_data)
        
        return metrics

    def evaluate(self, num_episodes=10):
        """Evaluate the current policy"""
        eval_rewards = []
        eval_lengths = []
        eval_successes = []
        
        for _ in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            # Reset reward function for evaluation
            self.reward_fn.reset()
            
            while not done:
                action = self.agent.act(obs, eval_mode=True)
                next_obs, env_reward, done, info = self.env.step(action)
                
                # Use custom reward for evaluation too
                reward = self.reward_fn.compute_reward(obs, action, info, done)
                
                obs = next_obs
                episode_reward += reward
                episode_length += 1
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            eval_successes.append(float(info.get('success', False)))
        
        return {
            'eval_reward_mean': np.mean(eval_rewards),
            'eval_reward_std': np.std(eval_rewards),
            'eval_length_mean': np.mean(eval_lengths),
            'eval_success_rate': np.mean(eval_successes)
        }

    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.num_episodes} episodes...")
        print(f"Device: {self.device}")
        print(f"Buffer size: {self.buffer_size}")
        print(f"Environment: {type(self.env).__name__}")
        print("="*50)
        
        for episode in range(self.num_episodes):
            # Collect trajectories
            episodes_collected = self.collect_trajectories()
            self.episode_count += episodes_collected
            
            # Train agent
            train_metrics = self.train_step()
            
            # Simple console logging (no CSV)
            if len(self.episode_rewards) > 0:
                avg_reward = np.mean(self.episode_rewards)
                avg_length = np.mean(self.episode_lengths)
                success_rate = np.mean(self.success_rate)
                
                print(f"Episode {episode:4d} | "
                      f"Steps: {self.global_step:6d} | "
                      f"Reward: {avg_reward:8.2f} | "
                      f"Length: {avg_length:6.1f} | "
                      f"Success: {success_rate:5.2f} | "
                      f"Loss: {train_metrics.get('loss', 0):7.4f}")
            
            # Evaluation
            if episode % self.eval_frequency == 0 and episode > 0:
                eval_metrics = self.evaluate()
                
                print(f"EVAL    {episode:4d} | "
                      f"Reward: {eval_metrics['eval_reward_mean']:8.2f} | "
                      f"Success: {eval_metrics['eval_success_rate']:5.2f} | "
                      f"Length: {eval_metrics['eval_length_mean']:6.1f}")
                print("-" * 70)
            
            # Save model
            if episode % self.save_frequency == 0 and episode > 0:
                self.save_checkpoint(episode)
        
        print("Training completed!")
        print("="*50)
        
        # Final evaluation
        final_eval = self.evaluate(num_episodes=20)
        print(f"Final Performance:")
        print(f"  Average Reward: {final_eval['eval_reward_mean']:.2f}")
        print(f"  Success Rate: {final_eval['eval_success_rate']:.2f}")
        print(f"  Average Length: {final_eval['eval_length_mean']:.1f}")
        
        self.save_checkpoint(self.num_episodes)

    def save_checkpoint(self, episode):
        """Save model checkpoint"""
        checkpoint = {
            'episode': episode,
            'global_step': self.global_step,
            'episode_count': self.episode_count,
            'actor_state_dict': self.agent.actor.state_dict(),
            'critic_state_dict': self.agent.critic.state_dict(),
            'actor_optimizer_state_dict': self.agent.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.agent.critic_optimizer.state_dict(),
            'config': self.config
        }
        
        checkpoint_path = self.work_dir / f'checkpoint_episode_{episode}.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        self.episode_count = checkpoint['episode_count']
        self.global_step = checkpoint['global_step']
        
        print(f"Loaded checkpoint from episode {checkpoint['episode']}")


def main():
    """Main training function"""
    
    # Configuration - using values from your config.py
    config = {
        'device': 'auto',  # Will auto-detect GPU/CPU
        'seed': 42,
        'learning_rate': 3e-4,
        'hidden_dim': 256,
        'clip_ratio': 0.2,
        'value_coef': 0.5,
        'entropy_coef': 0.01,
        'ppo_epochs': 10,
        'batch_size': 64,
        'buffer_size': 2048,
        'num_episodes': 50,  # Reduced for testing
        'eval_frequency': 10,  # More frequent evaluation for testing
        'save_frequency': 25,
        'target_volume': 20,
        # Reward weights
        'w_volume': 10.0,
        'w_time': 0.1,
        'w_completion': 5.0,
        'w_collision': 2.0,
        'w_drop': 10.0,
        'w_contamination': 3.0,
        'w_miss': 5.0,
        'w_jerk': 1.0
    }
    
    print("="*50)
    print("PIPETTING RL TRAINING")
    print("="*50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    print("="*50)
    
    # Create trainer and start training
    trainer = PipettingTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()