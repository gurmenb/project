#!/usr/bin/env python3

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
import time
from pathlib import Path
import numpy as np
import torch
from collections import deque

# Set CUDA device before importing other modules for AWS AMI compatibility
if torch.cuda.is_available():
    torch.cuda.set_device(0)  # Use first GPU on AWS
    print(f"Using GPU: {torch.cuda.get_device_name()}")
else:
    print("CUDA not available, using CPU")

# Import your PPO modules
from ppo_agent import PPOAgent
from ppo_buffer import PPOBufferSimple, PPODataLoader
from config import get_config
import utils
# Use the existing comprehensive logger
from logger import Logger

# Import the integrated environment
from integrated_pipette_environment import IntegratedPipetteEnv

# Enable optimizations for PyTorch 2.6.0 on AWS
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
if hasattr(torch.backends.cudnn, 'allow_tf32'):
    torch.backends.cudnn.allow_tf32 = True


class PipettingWorkspace:
    """
    Main training workspace - UPDATED for MuJoCo integration with AWS AMI compatibility
    """

    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg['seed'])
        
        # Improved device selection for AWS AMI
        self.device = self._setup_device(cfg['device'])
        print(f"Using device: {self.device}")
        
        self.setup()

        # Create agent with MuJoCo specs
        obs_spec = self.train_env.observation_space
        action_spec = self.train_env.action_space

        print(f"MuJoCo Observation Space: {obs_spec.shape}")
        print(f"MuJoCo Action Space: {action_spec.shape}")

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

    def _setup_device(self, device_cfg):
        """Setup device with AWS AMI optimizations"""
        if device_cfg == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
                # Warm up GPU
                dummy = torch.randn(1, device=device)
                del dummy
                return device
            else:
                return torch.device('cpu')
        else:
            return torch.device(device_cfg)

    def setup(self):
        """Setup environment, logger, and training components"""
        # 1) Create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg['use_tb'])

        # 2) Create MuJoCo environments
        xml_path = "particle_pipette_system.xml"
        
        try:
            self.train_env = IntegratedPipetteEnv(xml_path)
            self.eval_env = IntegratedPipetteEnv(xml_path)
        except Exception as e:
            print(f"Error creating environment: {e}")
            print("Make sure particle_pipette_system.xml exists and integrated_pipette_environment.py is available")
            raise

        # Verify environment setup
        print(f"Environment observation space: {self.train_env.observation_space}")
        print(f"Environment action space: {self.train_env.action_space}")

        # 3) Create PPO buffer with proper dimensions
        obs_dim = self.train_env.observation_space.shape[0]
        act_dim = self.train_env.action_space.shape[0]

        self.ppo_buffer = PPOBufferSimple(
            obs_dim=obs_dim,
            act_dim=act_dim,
            max_size=self.cfg['buffer_size'],
            gamma=self.cfg.get('gamma', 0.99),
            lam=self.cfg.get('lam', 0.95)
        )

        # Training metrics
        self.episode_rewards = deque(maxlen=50)
        self.success_rates = deque(maxlen=50)
        self.episode_lengths = deque(maxlen=50)
        self.physics_rewards = deque(maxlen=50)

        # Additional metrics for pipetting task
        self.particles_transferred = deque(maxlen=50)
        self.aspiration_success = deque(maxlen=50)
        self.dispensing_accuracy = deque(maxlen=50)

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
        """Collect one episode of experience - UPDATED for MuJoCo with error handling"""
        try:
            obs = self.train_env.reset()
            if obs is None:
                raise ValueError("Environment reset returned None")
            
            # Ensure observation is proper numpy array
            obs = np.asarray(obs, dtype=np.float32)
            
            episode_reward = 0.0
            episode_length = 0
            physics_reward_total = 0.0
            task_completed = False

            while True:
                if self.ppo_buffer.is_full():
                    break

                # 1) Get action & value with error handling
                try:
                    with torch.no_grad():
                        action = self.agent.act(obs, eval_mode=False)
                        value = self.agent.get_value(obs)

                        # Get log probability for PPO - fix device handling
                        obs_tensor = torch.as_tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
                        action_tensor = torch.as_tensor(action, device=self.device, dtype=torch.float32).unsqueeze(0)
                        dist = self.agent.actor(obs_tensor)
                        log_prob = dist.log_prob(action_tensor).sum(dim=-1).item()

                except Exception as e:
                    print(f"Error in action generation: {e}")
                    # Use random action as fallback
                    action = self.train_env.action_space.sample()
                    value = 0.0
                    log_prob = 0.0

                # 2) Step environment with error handling
                try:
                    step_result = self.train_env.step(action)
                    if len(step_result) != 4:
                        raise ValueError(f"Expected 4 values from env.step, got {len(step_result)}")
                    
                    next_obs, reward, done, info = step_result
                    
                    # Ensure types are correct
                    next_obs = np.asarray(next_obs, dtype=np.float32)
                    reward = float(reward)
                    done = bool(done)
                    info = info or {}
                    
                except Exception as e:
                    print(f"Error in environment step: {e}")
                    # Use safe defaults
                    next_obs = obs.copy()
                    reward = 0.0
                    done = True
                    info = {}

                # 3) Store in buffer
                try:
                    stored = self.ppo_buffer.store(obs, action, reward, value, log_prob)
                    if not stored:
                        break
                except Exception as e:
                    print(f"Error storing in buffer: {e}")
                    break

                # Update metrics
                episode_reward += reward
                episode_length += 1
                self._global_step += 1
                
                # Track physics-specific metrics
                physics_reward = info.get('physics_reward', 0.0)
                physics_reward_total += physics_reward
                task_completed = info.get('task_completed', False)

                obs = next_obs

                # Check termination conditions
                if done or episode_length >= self.cfg.get('max_episode_length', 1000):
                    try:
                        last_val = 0.0 if done else self.agent.get_value(obs)
                        self.ppo_buffer.finish_path(last_val)
                        self._global_episode += 1
                    except Exception as e:
                        print(f"Error finishing path: {e}")
                        self.ppo_buffer.finish_path(0.0)
                    break

        except Exception as e:
            print(f"Critical error in collect_episode: {e}")
            # Return safe defaults
            episode_reward = 0.0
            episode_length = 1
            physics_reward_total = 0.0
            task_completed = False
            info = {}

        return episode_reward, episode_length, task_completed, physics_reward_total, info

    def update_agent(self):
        """Update PPO agent with collected data - Enhanced error handling"""
        try:
            data = self.ppo_buffer.get()
            if data is None:
                return {'actor_loss': 0, 'critic_loss': 0, 'policy_ratio': 0}

            data_loader = PPODataLoader(data, batch_size=self.cfg['batch_size'])
            total_metrics = {'actor_loss': 0, 'critic_loss': 0, 'policy_ratio': 0}
            update_count = 0

            for epoch in range(self.cfg['ppo_epochs']):
                for batch in data_loader:
                    try:
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
                        
                    except Exception as e:
                        print(f"Error in agent update (epoch {epoch}): {e}")
                        continue

            # Average metrics
            if update_count > 0:
                for key in total_metrics:
                    total_metrics[key] /= update_count

            self.ppo_buffer.clear()
            return total_metrics
            
        except Exception as e:
            print(f"Critical error in update_agent: {e}")
            return {'actor_loss': 0, 'critic_loss': 0, 'policy_ratio': 0}

    def eval(self, num_eval_episodes=None):
        """Evaluate current policy - UPDATED for MuJoCo with enhanced metrics"""
        num_episodes = num_eval_episodes or self.cfg['num_eval_episodes']
        total_reward = 0.0
        total_success = 0.0
        total_steps = 0
        total_physics_reward = 0.0
        particles_transferred_total = 0

        for episode in range(num_episodes):
            try:
                obs = self.eval_env.reset()
                if obs is None:
                    continue
                    
                obs = np.asarray(obs, dtype=np.float32)
                ep_reward = 0.0
                ep_physics_reward = 0.0
                ep_steps = 0

                while True:
                    try:
                        with torch.no_grad(), utils.eval_mode(self.agent):
                            action = self.agent.act(obs, eval_mode=True)
                        
                        step_result = self.eval_env.step(action)
                        next_obs, reward, done, info = step_result
                        
                        next_obs = np.asarray(next_obs, dtype=np.float32)
                        reward = float(reward)
                        done = bool(done)
                        info = info or {}
                        
                    except Exception as e:
                        print(f"Error in eval step: {e}")
                        break

                    ep_reward += reward
                    ep_physics_reward += info.get('physics_reward', 0.0)
                    ep_steps += 1
                    total_steps += 1
                    obs = next_obs

                    if done or ep_steps >= self.cfg.get('max_episode_length', 1000):
                        total_success += float(info.get('task_completed', False))
                        particles_transferred_total += info.get('particles_transferred', 0)
                        break

                total_reward += ep_reward
                total_physics_reward += ep_physics_reward
                
            except Exception as e:
                print(f"Error in eval episode {episode}: {e}")
                continue

        # Log evaluation results
        try:
            with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
                log('episode_reward', total_reward / max(num_episodes, 1))
                log('episode_success', total_success / max(num_episodes, 1))
                log('episode_length', total_steps / max(num_episodes, 1))
                log('physics_reward', total_physics_reward / max(num_episodes, 1))
                log('particles_transferred', particles_transferred_total / max(num_episodes, 1))
                log('episode', self.global_episode)
                log('step', self.global_step)
                log('eval_total_time', self.timer.total_time())
        except Exception as e:
            print(f"Error logging eval results: {e}")

    def train(self):
        """Main training loop with AWS AMI optimizations"""
        print("Starting PPO Training with MuJoCo Physics!")
        print(f"Device: {self.device}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Observation dim: {self.train_env.observation_space.shape}")
        print(f"Action dim: {self.train_env.action_space.shape}")
        print(f"Buffer size: {self.cfg['buffer_size']}")
        print("=" * 50)

        train_until_step = utils.Until(self.cfg['num_train_frames'])
        eval_every_step = utils.Every(self.cfg['eval_every_frames'])
        save_every_step = utils.Every(self.cfg.get('save_every_frames', 50000))

        episodes_this_round = 0
        
        while train_until_step(self.global_step):
            try:
                # Collect episodes
                episodes_this_round = 0
                collection_start_time = time.time()  # Use time.time() directly
                
                while (not self.ppo_buffer.is_full()) and (episodes_this_round < self.cfg.get('max_episodes_per_round', 10)):
                    ep_r, ep_len, success, physics_r, info = self.collect_episode()
                    
                    # Store metrics
                    self.episode_rewards.append(ep_r)
                    self.episode_lengths.append(ep_len)
                    self.success_rates.append(float(success))
                    self.physics_rewards.append(physics_r)
                    
                    # Store task-specific metrics
                    self.particles_transferred.append(info.get('particles_transferred', 0))
                    
                    episodes_this_round += 1

                collection_time = time.time() - collection_start_time

                # Update agent
                update_start_time = time.time()
                update_metrics = self.update_agent()
                update_time = time.time() - update_start_time
                
                elapsed_time, total_time = self.timer.reset()

                # Log training metrics
                try:
                    with self.logger.log_and_dump_ctx(self.global_frame, ty='train') as log:
                        # Performance metrics
                        log('fps', episodes_this_round / max(elapsed_time, 0.001))
                        log('collection_time', collection_time)
                        log('update_time', update_time)
                        log('total_time', total_time)
                        
                        # Episode metrics
                        log('episode_reward', np.mean(self.episode_rewards) if self.episode_rewards else 0)
                        log('episode_length', np.mean(self.episode_lengths) if self.episode_lengths else 0)
                        log('success_rate', np.mean(self.success_rates) if self.success_rates else 0)
                        log('physics_reward', np.mean(self.physics_rewards) if self.physics_rewards else 0)
                        
                        # Task-specific metrics
                        log('particles_transferred', np.mean(self.particles_transferred) if self.particles_transferred else 0)
                        
                        # Training progress
                        log('episode', self.global_episode)
                        log('step', self.global_step)
                        log('episodes_this_round', episodes_this_round)
                        
                        # Agent metrics
                        for key, val in update_metrics.items():
                            log(key, val)
                        
                        # GPU memory usage if available
                        if torch.cuda.is_available():
                            log('gpu_memory_allocated', torch.cuda.memory_allocated() / 1024**3)  # GB
                            log('gpu_memory_reserved', torch.cuda.memory_reserved() / 1024**3)   # GB
                
                except Exception as e:
                    print(f"Error logging training metrics: {e}")

                # Evaluation
                if eval_every_step(self.global_step):
                    print(f"Running evaluation at step {self.global_step}")
                    self.eval()

                # Periodic saving
                if save_every_step(self.global_step) or self.cfg.get('save_snapshot', True):
                    self.save_snapshot()

                # Print progress
                if self.global_episode % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
                    avg_success = np.mean(self.success_rates) if self.success_rates else 0
                    avg_particles = np.mean(self.particles_transferred) if self.particles_transferred else 0
                    
                    print(f"Episode {self.global_episode:6d} | "
                          f"Step {self.global_step:8d} | "
                          f"Reward: {avg_reward:6.2f} | "
                          f"Success: {avg_success:4.2f} | "
                          f"Particles: {avg_particles:4.1f} | "
                          f"FPS: {episodes_this_round/max(elapsed_time, 0.001):5.1f}")

                # Memory cleanup for long training runs
                if self.global_step % 1000 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except KeyboardInterrupt:
                print("Training interrupted by user")
                self.save_snapshot()
                break
            except Exception as e:
                print(f"Error in training loop: {e}")
                print("Continuing training...")
                continue

        print("Training completed!")
        self.save_snapshot()

    def save_snapshot(self):
        """Save training snapshot with error handling"""
        try:
            snapshot = self.work_dir / 'snapshot.pt'
            keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
            payload = {k: self.__dict__[k] for k in keys_to_save if k in self.__dict__}
            
            # Add training metrics to snapshot
            payload['metrics'] = {
                'episode_rewards': list(self.episode_rewards),
                'success_rates': list(self.success_rates),
                'episode_lengths': list(self.episode_lengths),
                'physics_rewards': list(self.physics_rewards),
                'particles_transferred': list(self.particles_transferred)
            }
            
            with snapshot.open('wb') as f:
                torch.save(payload, f)
            print(f"Snapshot saved to {snapshot}")
            
        except Exception as e:
            print(f"Error saving snapshot: {e}")

    def load_snapshot(self):
        """Load training snapshot with compatibility for PyTorch 2.6.0"""
        try:
            snapshot = self.work_dir / 'snapshot.pt'
            with snapshot.open('rb') as f:
                # Use weights_only=False for compatibility with PyTorch 2.6.0
                payload = torch.load(f, map_location=self.device, weights_only=False)
            
            for k, v in payload.items():
                if k == 'metrics':
                    # Restore metrics
                    if 'episode_rewards' in v:
                        self.episode_rewards.extend(v['episode_rewards'])
                    if 'success_rates' in v:
                        self.success_rates.extend(v['success_rates'])
                    if 'episode_lengths' in v:
                        self.episode_lengths.extend(v['episode_lengths'])
                    if 'physics_rewards' in v:
                        self.physics_rewards.extend(v['physics_rewards'])
                    if 'particles_transferred' in v:
                        self.particles_transferred.extend(v['particles_transferred'])
                else:
                    self.__dict__[k] = v
            
            print(f"Snapshot loaded from {snapshot}")
            print(f"Resumed at episode {self.global_episode}, step {self.global_step}")
            
        except Exception as e:
            print(f"Error loading snapshot: {e}")
            print("Starting training from scratch")


def main():
    """Main training function with enhanced error handling for AWS AMI"""
    
    # Print system information
    print("PIPETTING PPO TRAINING WITH MUJOCO")
    print("=" * 50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    
    print(f"Python executable: {os.sys.executable}")
    print(f"Working directory: {os.getcwd()}")
    print("=" * 50)

    try:
        # Get configuration
        cfg = get_config()
        
        # Validate required files exist
        required_files = [
            'particle_pipette_system.xml',
            'ppo_agent.py',
            'ppo_buffer.py', 
            'config.py',
            'utils.py',
            'logger.py'
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            print(f"Warning: Missing required files: {missing_files}")
            print("Please ensure all required files are present")
        
        # Create workspace
        root_dir = Path.cwd()
        workspace = PipettingWorkspace(cfg)

        # Load snapshot if exists
        snapshot = root_dir / 'snapshot.pt'
        if snapshot.exists():
            print(f'Resuming training from: {snapshot}')
            workspace.load_snapshot()
        else:
            print('Starting new training session')

        # Start training
        workspace.train()
        
    except Exception as e:
        print(f"Critical error in main: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    if exit_code != 0:
        print(f"Training exited with code {exit_code}")
    exit(exit_code)