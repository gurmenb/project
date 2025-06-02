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
#from reward_function import make_pipetting_reward_function   # <— REMOVE this import
from config import get_config
import utils
from logger import Logger

# Import the new MuJoCo environment
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
        obs_spec    = self.train_env.observation_space
        action_spec = self.train_env.action_space

        print(f"MuJoCo Observation Space: {obs_spec.shape}")  # Should be (26,)
        print(f"MuJoCo Action Space: {action_spec.shape}")    # Should be (4,)

        self.agent = PPOAgent(
            obs_shape   = obs_spec.shape,
            action_shape= action_spec.shape,
            device      = self.device,
            lr          = cfg['lr'],
            hidden_dim  = cfg['hidden_dim'],
            clip_ratio  = cfg['clip_ratio'],
            stddev_clip = cfg.get('stddev_clip', 0.3)
        )

        self.timer = utils.Timer()
        self._global_step    = 0
        self._global_episode = 0

    def setup(self):
        """Setup environment, logger, and training components"""
        # 1) Create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg['use_tb'])

        # 2) Create MuJoCo environments
        xml_path = "particle_pipette_system.xml"
        self.train_env = IntegratedPipetteEnv(xml_path)
        self.eval_env  = IntegratedPipetteEnv(xml_path)

        # 3) Create PPO buffer
        obs_dim = self.train_env.observation_space.shape[0]  # 26
        act_dim = self.train_env.action_space.shape[0]       # 4

        self.ppo_buffer = PPOBufferSimple(
            obs_dim = obs_dim,
            act_dim = act_dim,
            max_size= self.cfg['buffer_size'],
            gamma   = self.cfg.get('gamma', 0.99),
            lam     = self.cfg.get('lam', 0.95)
        )

        # We no longer need a separate reward function
        # self.reward_fn = make_pipetting_reward_function(...)

        # Training metrics
        self.episode_rewards = deque(maxlen=50)
        self.success_rates   = deque(maxlen=50)
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
        obs = self.train_env.reset()  # 26D observation

        episode_reward = 0.0
        episode_length = 0

        while True:
            if self.ppo_buffer.is_full():
                break

            # 1) Get action & value
            with torch.no_grad():
                action = self.agent.act(obs, eval_mode=False)  # 4D action
                value  = self.agent.get_value(obs)

                obs_tensor = torch.as_tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
                action_tensor = torch.as_tensor(action, device=self.device, dtype=torch.float32).unsqueeze(0)
                dist = self.agent.actor(obs_tensor)
                log_prob = dist.log_prob(action_tensor).sum(dim=-1).item()

            # 2) Step environment
            next_obs, physics_reward, done, info = self.train_env.step(action)

            # 3) Store in buffer
            stored = self.ppo_buffer.store(obs, action, physics_reward, value, log_prob)
            if not stored:
                break

            episode_reward += physics_reward
            episode_length += 1
            self._global_step += 1

            obs = next_obs

            if done:
                last_val = 0.0 if done else self.agent.get_value(obs)
                self.ppo_buffer.finish_path(last_val)
                self._global_episode += 1
                break

        success = info.get('task_completed', False)
        return episode_reward, episode_length, success

    def update_agent(self):
        """Update PPO agent with collected data"""
        data = self.ppo_buffer.get()
        if data is None:
            return {'actor_loss':0, 'critic_loss':0, 'policy_ratio':0}

        data_loader = PPODataLoader(data, batch_size=self.cfg['batch_size'])
        total_metrics = {'actor_loss':0, 'critic_loss':0, 'policy_ratio':0}
        update_count = 0

        for _ in range(self.cfg['ppo_epochs']):
            for batch in data_loader:
                critic_metrics = self.agent.update_critic(batch)
                actor_metrics  = self.agent.update_actor(batch)
                for key in total_metrics:
                    if key in critic_metrics:
                        total_metrics[key] += critic_metrics[key]
                    if key in actor_metrics:
                        total_metrics[key] += actor_metrics[key]
                update_count += 1

        if update_count > 0:
            for key in total_metrics:
                total_metrics[key] /= update_count

        self.ppo_buffer.clear()
        return total_metrics

    def eval(self, num_eval_episodes=None):
        """Evaluate current policy - UPDATED for MuJoCo"""
        num_episodes = num_eval_episodes or self.cfg['num_eval_episodes']
        total_reward    = 0.0
        total_success   = 0.0
        total_steps     = 0
        particles_trans = 0

        for _ in range(num_episodes):
            obs = self.eval_env.reset()
            ep_reward = 0.0
            ep_steps  = 0

            while True:
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(obs, eval_mode=True)
                next_obs, reward, done, info = self.eval_env.step(action)

                ep_reward += reward
                ep_steps  += 1
                total_steps += 1
                obs = next_obs

                if done:
                    total_success += float(info.get('task_completed', False))
                    particles_trans += info.get('held_particle_count',0)
                    break

            total_reward += ep_reward

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / num_episodes)
            log('episode_success', total_success / num_episodes)
            log('episode_length', total_steps / num_episodes)
            log('particles_transferred', particles_trans / num_episodes)
            log('episode', self.global_episode)
            log('step',    self.global_step)
            log('eval_total_time', self.timer.total_time())

    def train(self):
        """Main training loop (NO CHANGES NEEDED)"""
        print("Starting PPO Training with MuJoCo Physics!")
        print(f"Device: {self.device}")
        print(f"Observation dim: {self.train_env.observation_space.shape}")
        print(f"Action dim: {self.train_env.action_space.shape}")
        print(f"Buffer size: {self.cfg['buffer_size']}")
        print("="*50)

        train_until_step = utils.Until(self.cfg['num_train_frames'])
        eval_every_step = utils.Every(self.cfg['eval_every_frames'])

        while train_until_step(self.global_step):
            episodes_this_round = 0
            while (not self.ppo_buffer.is_full()) and (episodes_this_round < 10):
                ep_r, ep_len, success = self.collect_episode()
                self.episode_rewards.append(ep_r)
                self.episode_lengths.append(ep_len)
                self.success_rates.append(float(success))
                episodes_this_round += 1

            update_metrics = self.update_agent()
            elapsed_time, total_time = self.timer.reset()

            with self.logger.log_and_dump_ctx(self.global_frame, ty='train') as log:
                log('fps', episodes_this_round / max(elapsed_time, 0.001))
                log('total_time', total_time)
                log('episode_reward', np.mean(self.episode_rewards))
                log('episode_length', np.mean(self.episode_lengths))
                log('success_rate', np.mean(self.success_rates))
                log('episode', self.global_episode)
                log('step', self.global_step)

                for key, val in update_metrics.items():
                    log(key, val)

            if eval_every_step(self.global_step):
                self.eval()

            if self.cfg.get('save_snapshot', True):
                self.save_snapshot()

    def save_snapshot(self):
        """Save training snapshot (unchanged)"""
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent','timer','_global_step','_global_episode']
        payload = {k:self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        """Load training snapshot (unchanged)"""
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k,v in payload.items():
            self.__dict__[k] = v


def main():
    """Main training function"""

    cfg = get_config()
    print("PIPETTING PPO TRAINING WITH MUJOCO")
    print("="*50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    print("="*50)

    root_dir = Path.cwd()
    workspace = PipettingWorkspace(cfg)

    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()

    workspace.train()

if __name__ == '__main__':
    main()
