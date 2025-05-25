#!/usr/bin/env python3

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
from pathlib import Path
import numpy as np
import torch
from collections import deque

# Import your PPO modules (unchanged)
from ppo_agent import PPOAgent
from ppo_buffer import PPOBufferSimple, PPODataLoader
from reward_function import make_pipetting_reward_function
from config import get_config
import utils
from logger import Logger

# Import partner's MuJoCo environment (NEW)
from integrated_pipette_environment import IntegratedPipetteEnv

torch.backends.cudnn.benchmark = True


class PipettingWorkspace:
    """
    Main training workspace - UPDATED for MuJoCo integration
    """
    
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')
        
        self.cfg = cfg
        utils.set_seed_everywhere(cfg['seed'])
        self.device = torch.device(cfg['device'])
        self.setup()
        
        # Create agent with MuJoCo specs
        obs_spec = self.train_env.observation_space
        action_spec = self.train_env.action_space
        
        print(f"MuJoCo Observation Space: {obs_spec.shape}")  # Should be (25,)
        print(f"MuJoCo Action Space: {action_spec.shape}")    # Should be (4,)
        
        self.agent = PPOAgent(
            obs_shape=obs_spec.shape,
            action_shape=action_spec.shape,
            device=self.device,
            lr=cfg['lr'],
            hidden_dim=cfg['hidden_dim'],
            clip_ratio=cfg['clip_ratio'],
            stddev_clip=cfg.get('stddev_clip', 0.3)
        )
        
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
    
    def setup(self):
        """Setup environment, logger, and training components"""
        # Create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg['use_tb'])
        
        # Create MuJoCo environments (UPDATED)
        xml_path = "particle_pipette_system.xml"  # Make sure this file exists
        self.train_env = IntegratedPipetteEnv(xml_path)
        self.eval_env = IntegratedPipetteEnv(xml_path)
        
        # Create PPO buffer with MuJoCo dimensions (UPDATED)
        obs_dim = self.train_env.observation_space.shape[0]  # 25
        act_dim = self.train_env.action_space.shape[0]       # 4
        
        self.ppo_buffer = PPOBufferSimple(
            obs_dim=obs_dim,
            act_dim=act_dim,
            max_size=self.cfg['buffer_size'],
            gamma=self.cfg.get('gamma', 0.99),
            lam=self.cfg.get('lam', 0.95)
        )
        
        # Keep your custom reward function for additional rewards (OPTIONAL)
        self.reward_fn = make_pipetting_reward_function(
            target_volume=self.cfg['target_volume'],
            w_volume=self.cfg['w_volume'],
            w_completion=self.cfg['w_completion'],
            w_time=self.cfg['w_time'],
            w_collision=self.cfg['w_collision'],
            w_drop=self.cfg['w_drop'],
            w_contamination=self.cfg['w_contamination'],
            w_miss=self.cfg['w_miss'],
            w_jerk=self.cfg['w_jerk']
        )
        
        # Training metrics
        self.episode_rewards = deque(maxlen=50)
        self.success_rates = deque(maxlen=50)
        self.episode_lengths = deque(maxlen=50)
    
    @property
    def global_step(self):
        return self._global_step
    
    @property
    def global_episode(self):
        return self._global_episode
    
    @property
    def global_frame(self):
        return self.global_step
    
    def collect_episode(self):
        """Collect one episode of experience - UPDATED for MuJoCo"""
        obs = self.train_env.reset()  # Now returns 25D observation
        self.reward_fn.reset()
        
        episode_reward = 0
        episode_length = 0
        
        while True:
            # Check if buffer is full
            if self.ppo_buffer.is_full():
                break
                
            # Get action and value from agent (no changes needed)
            with torch.no_grad():
                action = self.agent.act(obs, eval_mode=False)  # 4D action: [x, y, z, plunger]
                value = self.agent.get_value(obs)
                
                # Get log probability
                obs_tensor = torch.as_tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
                action_tensor = torch.as_tensor(action, device=self.device, dtype=torch.float32).unsqueeze(0)
                dist = self.agent.actor(obs_tensor)
                log_prob = dist.log_prob(action_tensor).sum(dim=-1).item()
            
            # Take environment step (UPDATED for physics-primary rewards)
            next_obs, mujoco_reward, done, info = self.train_env.step(action)
            
            # IMPORTANT: Physics rewards are primary (aspiration +2.0, dispensing +1.5, etc.)
            physics_info = info.get('physics_info', {})
            physics_reward = sum(physics_info.get('reward_components', {}).values())
            
            # Combined reward: Physics primary + MuJoCo supplementary  
            reward = physics_reward + 0.3 * mujoco_reward  # Physics dominates
            
            # Store in buffer
            stored = self.ppo_buffer.store(obs, action, reward, value, log_prob)
            if not stored:
                break  # Buffer is full
            
            episode_reward += reward
            episode_length += 1
            self._global_step += 1
            
            obs = next_obs
            
            if done:
                # Finish trajectory in buffer
                last_val = self.agent.get_value(obs) if not done else 0
                self.ppo_buffer.finish_path(last_val)
                self._global_episode += 1
                break
        
        # Extract success information from MuJoCo environment
        success = info.get('task_completed', False)
        return episode_reward, episode_length, success
    
    def update_agent(self):
        """Update PPO agent with collected data (NO CHANGES NEEDED)"""
        # Get data from buffer
        data = self.ppo_buffer.get()
        if data is None:
            return {'actor_loss': 0, 'critic_loss': 0, 'policy_ratio': 0}
        
        # Create data loader
        data_loader = PPODataLoader(data, batch_size=self.cfg['batch_size'])
        
        # Update for multiple epochs
        total_metrics = {'actor_loss': 0, 'critic_loss': 0, 'policy_ratio': 0}
        update_count = 0
        
        for epoch in range(self.cfg['ppo_epochs']):
            for batch in data_loader:
                # Update critic
                critic_metrics = self.agent.update_critic(batch)
                
                # Update actor
                actor_metrics = self.agent.update_actor(batch)
                
                # Accumulate metrics
                for key in total_metrics:
                    if key in critic_metrics:
                        total_metrics[key] += critic_metrics[key]
                    if key in actor_metrics:
                        total_metrics[key] += actor_metrics[key]
                
                update_count += 1
        
        # Average metrics
        if update_count > 0:
            for key in total_metrics:
                total_metrics[key] /= update_count
        
        # Clear buffer after training
        self.ppo_buffer.clear()
        
        return total_metrics
    
    def eval(self, num_eval_episodes=None):
        """Evaluate current policy - UPDATED for MuJoCo"""
        num_episodes = num_eval_episodes or self.cfg['num_eval_episodes']
        
        total_reward = 0
        total_success = 0
        total_steps = 0
        particles_transferred = 0
        
        for episode in range(num_episodes):
            obs = self.eval_env.reset()
            
            episode_reward = 0
            episode_steps = 0
            
            while True:
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(obs, eval_mode=True)
                
                next_obs, reward, done, info = self.eval_env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                total_steps += 1
                
                obs = next_obs
                
                if done:
                    total_success += float(info.get('task_completed', False))
                    particles_transferred += info.get('particles_held', 0)
                    break
            
            total_reward += episode_reward
        
        # Log evaluation results with MuJoCo-specific metrics
        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / num_episodes)
            log('episode_success', total_success / num_episodes)
            log('episode_length', total_steps / num_episodes)
            log('particles_transferred', particles_transferred / num_episodes)
            log('episode', self.global_episode)
            log('step', self.global_step)
            log('eval_total_time', self.timer.total_time())
    
    def train(self):
        """Main training loop (NO CHANGES NEEDED)"""
        print("Starting PPO Training with MuJoCo Physics!")
        print(f"Device: {self.device}")
        print(f"Observation dim: {self.train_env.observation_space.shape}")
        print(f"Action dim: {self.train_env.action_space.shape}")
        print(f"Buffer size: {self.cfg['buffer_size']}")
        print("Physics Simulation Features:")
        print("  - Realistic particle aspiration/dispensing")
        print("  - 4 pipette states: Idle → Aspirating → Holding → Dispensing") 
        print("  - RL controls plunger directly (not MuJoCo)")
        print("  - Rich physics rewards: +2.0 aspiration, +1.5 dispensing")
        print("="*50)
        
        train_until_step = utils.Until(self.cfg['num_train_frames'])
        eval_every_step = utils.Every(self.cfg['eval_every_frames'])
        
        while train_until_step(self.global_step):
            # Collect episodes until buffer is full or we have enough data
            episodes_this_round = 0
            while not self.ppo_buffer.is_full() and episodes_this_round < 10:
                episode_reward, episode_length, success = self.collect_episode()
                
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.success_rates.append(float(success))
                episodes_this_round += 1
            
            # Update agent
            update_metrics = self.update_agent()
            
            # Log training metrics
            elapsed_time, total_time = self.timer.reset()
            with self.logger.log_and_dump_ctx(self.global_frame, ty='train') as log:
                log('fps', episodes_this_round / max(elapsed_time, 0.001))
                log('total_time', total_time)
                log('episode_reward', np.mean(self.episode_rewards))
                log('episode_length', np.mean(self.episode_lengths))
                log('success_rate', np.mean(self.success_rates))
                log('episode', self.global_episode)
                log('step', self.global_step)
                
                # PPO metrics
                for key, value in update_metrics.items():
                    log(key, value)
            
            # Evaluation
            if eval_every_step(self.global_step):
                self.eval()
            
            # Save snapshot
            if self.cfg.get('save_snapshot', True):
                self.save_snapshot()
    
    def save_snapshot(self):
        """Save training snapshot (NO CHANGES NEEDED)"""
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)
    
    def load_snapshot(self):
        """Load training snapshot (NO CHANGES NEEDED)"""
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


def main():
    """Main training function"""
    
    # Get configuration
    cfg = get_config()
    
    print("PIPETTING PPO TRAINING WITH MUJOCO")
    print("="*50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    print("="*50)
    
    # Create workspace and start training
    root_dir = Path.cwd()
    workspace = PipettingWorkspace(cfg)
    
    # Resume from snapshot if exists
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    
    workspace.train()


if __name__ == '__main__':
    main()