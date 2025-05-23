#!/usr/bin/env python3

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
from pathlib import Path
import numpy as np
import torch
from collections import deque

# Import your modules
from ppo_agent import PPOAgent
from ppo_buffer import PPOBufferSimple, PPODataLoader
from reward_function import make_pipetting_reward_function
from config import get_config
import utils
from logger import Logger

torch.backends.cudnn.benchmark = True


class PipettingEnvironment:
    """
    Pipetting Environment (replace with your partner's MuJoCo environment)
    
    Observation Space (14D): liquid_in_plunger, balls_in_plunger, source_amount, target_amount,
                            source_pos(xyz), target_pos(xyz), aspiration_pressure, dispersion_pressure,
                            task_phase, submerged
    Action Space (6D): pipette_pos(xyz), plunger_pos, plunger_force, plunger_speed
    """
    
    def __init__(self, max_episode_steps=200):
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        
        # Environment state
        self.source_amount = 50
        self.target_amount = 0
        self.balls_in_plunger = 0
        self.task_phase = 0
        
    def observation_spec(self):
        return {'shape': (14,), 'dtype': np.float32}
    
    def action_spec(self):
        return {'shape': (6,), 'dtype': np.float32, 'minimum': -1.0, 'maximum': 1.0}
    
    def reset(self):
        self.current_step = 0
        self.source_amount = 50
        self.target_amount = 0
        self.balls_in_plunger = 0
        self.task_phase = 0
        return self._get_observation()
    
    def step(self, action):
        self.current_step += 1
        
        # Parse action
        pipette_pos = action[0:3]
        plunger_pos = action[3]
        plunger_force = action[4]
        plunger_speed = action[5]
        
        # Dummy physics simulation (replace with MuJoCo)
        if self.current_step < 50:
            self.task_phase = 0  # approach
        elif self.current_step < 100:
            self.task_phase = 1  # aspirate
            if plunger_pos < -0.5 and plunger_force > 0.3:
                if self.balls_in_plunger == 0:
                    transfer_amount = min(20, self.source_amount)
                    noise = np.random.normal(0, 2)
                    actual_transfer = max(0, min(50, transfer_amount + noise))
                    self.balls_in_plunger = actual_transfer
                    self.source_amount -= actual_transfer
        elif self.current_step < 150:
            self.task_phase = 2  # transfer
        else:
            self.task_phase = 3  # dispense
            if plunger_pos > 0.5 and plunger_force > 0.3:
                if self.balls_in_plunger > 0:
                    target_pos_xy = np.array([0.2, 0.5])
                    distance = np.linalg.norm(pipette_pos[:2] - target_pos_xy)
                    if distance < 0.15:
                        self.target_amount += self.balls_in_plunger
                        self.balls_in_plunger = 0
                    else:
                        self.balls_in_plunger = 0
        
        obs = self._get_observation()
        done = self.current_step >= self.max_episode_steps
        
        info = {
            'success': abs(self.target_amount - 20) <= 3 if done else False,
            'collision': False,
            'task_phase': self.task_phase,
            'balls_transferred': self.target_amount
        }
        
        return obs, 0.0, done, info
    
    def _get_observation(self):
        """Get 14D observation"""
        return np.array([
            float(self.balls_in_plunger > 0),  # liquid_in_plunger
            float(self.balls_in_plunger),      # balls_in_plunger
            float(self.source_amount),         # source_amount
            float(self.target_amount),         # target_amount
            0.0, 0.5, 0.1,                    # source_pos (x, y, z)
            0.2, 0.5, 0.1,                    # target_pos (x, y, z)
            1.0,                              # aspiration_pressure
            1.0,                              # dispersion_pressure
            float(self.task_phase),           # task_phase
            0.0                               # submerged
        ], dtype=np.float32)


class PipettingWorkspace:
    """
    Main training workspace (HW2 style)
    """
    
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')
        
        self.cfg = cfg
        utils.set_seed_everywhere(cfg['seed'])
        self.device = torch.device(cfg['device'])
        self.setup()
        
        # Create agent
        obs_spec = self.train_env.observation_spec()
        action_spec = self.train_env.action_spec()
        
        self.agent = PPOAgent(
            obs_shape=obs_spec['shape'],
            action_shape=action_spec['shape'],
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
        
        # Create environments
        self.train_env = PipettingEnvironment(max_episode_steps=self.cfg['max_episode_steps'])
        self.eval_env = PipettingEnvironment(max_episode_steps=self.cfg['max_episode_steps'])
        
        # Create PPO buffer
        self.ppo_buffer = PPOBufferSimple(
            obs_dim=14,
            act_dim=6,
            max_size=self.cfg['buffer_size'],
            gamma=self.cfg.get('gamma', 0.99),
            lam=self.cfg.get('lam', 0.95)
        )
        
        # Create reward function
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
        """Collect one episode of experience"""
        obs = self.train_env.reset()
        self.reward_fn.reset()
        
        episode_reward = 0
        episode_length = 0
        
        while True:
            # Check if buffer is full
            if self.ppo_buffer.is_full():
                break
                
            # Get action and value from agent
            with torch.no_grad():
                action = self.agent.act(obs, eval_mode=False)
                value = self.agent.get_value(obs)
                
                # Get log probability
                obs_tensor = torch.as_tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
                action_tensor = torch.as_tensor(action, device=self.device, dtype=torch.float32).unsqueeze(0)
                dist = self.agent.actor(obs_tensor)
                log_prob = dist.log_prob(action_tensor).sum(dim=-1).item()
            
            # Take environment step
            next_obs, env_reward, done, info = self.train_env.step(action)
            
            # Compute custom reward
            reward = self.reward_fn.compute_reward(obs, action, info, done)
            
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
        
        return episode_reward, episode_length, info.get('success', False)
    
    def update_agent(self):
        """Update PPO agent with collected data"""
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
        """Evaluate current policy"""
        num_episodes = num_eval_episodes or self.cfg['num_eval_episodes']
        
        total_reward = 0
        total_success = 0
        total_steps = 0
        
        for episode in range(num_episodes):
            obs = self.eval_env.reset()
            eval_reward_fn = make_pipetting_reward_function(target_volume=self.cfg['target_volume'])
            eval_reward_fn.reset()
            
            episode_reward = 0
            episode_steps = 0
            
            while True:
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(obs, eval_mode=True)
                
                next_obs, _, done, info = self.eval_env.step(action)
                reward = eval_reward_fn.compute_reward(obs, action, info, done)
                
                episode_reward += reward
                episode_steps += 1
                total_steps += 1
                
                obs = next_obs
                
                if done:
                    total_success += float(info.get('success', False))
                    break
            
            total_reward += episode_reward
        
        # Log evaluation results
        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / num_episodes)
            log('episode_success', total_success / num_episodes)
            log('episode_length', total_steps / num_episodes)
            log('episode', self.global_episode)
            log('step', self.global_step)
            log('eval_total_time', self.timer.total_time())
    
    def train(self):
        """Main training loop"""
        print("Starting PPO Training...")
        print(f"Device: {self.device}")
        print(f"Buffer size: {self.cfg['buffer_size']}")
        print("="*50)
        
        train_until_step = utils.Until(self.cfg['num_train_frames'])
        eval_every_step = utils.Every(self.cfg['eval_every_frames'])
        
        while train_until_step(self.global_step):
            # Collect episodes until buffer is full or we have enough data
            episodes_this_round = 0
            while not self.ppo_buffer.is_full() and episodes_this_round < 10:  # Limit episodes per round
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
        """Save training snapshot"""
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)
    
    def load_snapshot(self):
        """Load training snapshot"""
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


def main():
    """Main training function"""
    
    # Get configuration
    cfg = get_config()
    
    print("PIPETTING PPO TRAINING")
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