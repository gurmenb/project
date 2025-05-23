import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
import time

import utils
from logger import Logger
from replay_buffer import ReplayBuffer
from video import VideoRecorder
from reward_function import create_reward_function

# Import your custom environment here
# from ball_transfer_env import BallTransferEnv

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg: DictConfig):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        
        # Create reward function
        self.reward_fn = create_reward_function(cfg.reward)
        
        self.setup()

        # Create agent
        self.agent = make_agent(
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            self.cfg.agent
        )
        
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # Create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        
        # Create environments
        # NOTE: Replace with your actual environment
        # self.train_env = BallTransferEnv(**self.cfg.env)
        # self.eval_env = BallTransferEnv(**self.cfg.env)
        
        # For now, create dummy specs
        from types import SimpleNamespace
        self.train_env = SimpleNamespace(
            observation_spec=lambda: SimpleNamespace(shape=(21,)),
            action_spec=lambda: SimpleNamespace(shape=(8,)),
            reset=lambda: SimpleNamespace(observation=np.zeros(21), reward=0, discount=1.0, last=lambda: False),
            step=lambda a: SimpleNamespace(observation=np.zeros(21), reward=0, discount=0.99, last=lambda: False)
        )
        self.eval_env = self.train_env
        
        # Create replay buffer
        self.replay_buffer = ReplayBuffer(
            obs_shape=(21,),
            action_shape=(8,),
            capacity=self.cfg.replay_buffer_size,
            batch_size=self.cfg.batch_size,
            device=self.device
        )
        
        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None
        )

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step

    def eval(self):
        """Evaluate the agent."""
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(enabled=(episode == 0))
            
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation, eval_mode=True)
                    
                time_step = self.eval_env.step(action)
                self.video_recorder.record()
                total_reward += time_step.reward
                step += 1

            episode += 1

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def train(self):
        """Main training loop."""
        # Predicates
        train_until_step = utils.Until(self.cfg.num_train_frames)
        seed_until_step = utils.Until(self.cfg.num_seed_frames)
        eval_every_step = utils.Every(self.cfg.eval_every_frames)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                
                # Log stats
                elapsed_time, total_time = self.timer.reset()
                with self.logger.log_and_dump_ctx(self.global_frame, ty='train') as log:
                    log('fps', episode_step / elapsed_time)
                    log('total_time', total_time)
                    log('episode_reward', episode_reward)
                    log('episode_length', episode_step)
                    log('episode', self.global_episode)
                    log('buffer_size', len(self.replay_buffer))
                    log('step', self.global_step)

                # Reset env
                time_step = self.train_env.reset()
                self.reward_fn.reset()
                
                # Save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                    
                episode_step = 0
                episode_reward = 0

            # Evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(), self.global_frame)
                self.eval()

            # Sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(
                    time_step.observation,
                    eval_mode=seed_until_step(self.global_step)
                )

            # Take env step
            next_time_step = self.train_env.step(action)
            
            # Compute custom reward
            # NOTE: In real implementation, get info from environment
            info = {'task_phase': 'pick', 'balls_in_gripper': 0}  # Dummy info
            reward, reward_components = self.reward_fn.compute_reward(
                {'gripper_height': 0.2},  # Dummy obs dict
                action,
                {'gripper_height': 0.2},  # Dummy next_obs dict
                info
            )
            
            episode_reward += reward
            
            # Add to replay buffer
            self.replay_buffer.add(
                time_step.observation,
                action,
                reward,
                next_time_step.observation,
                next_time_step.discount,
                next_time_step.last()
            )
            
            # Update agent
            if self.global_step >= self.cfg.num_seed_frames:
                if self.global_step % self.cfg.agent.update_every_steps == 0:
                    for _ in range(self.cfg.agent.num_updates):
                        metrics = self.agent.update_critic(
                            self.replay_buffer.get_iterator()
                        )
                        
                    self.logger.log_metrics(metrics, self.global_frame, ty='train')
                    
                if self.global_step % self.cfg.agent.actor_update_every_steps == 0:
                    metrics = self.agent.update_actor(
                        self.replay_buffer.get_iterator()
                    )
                    self.logger.log_metrics(metrics, self.global_frame, ty='train')

            time_step = next_time_step
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path='cfgs', config_name='config', version_base="1.1")
def main(cfg: DictConfig):
    workspace = Workspace(cfg)
    snapshot = Path.cwd() / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()


# Placeholder classes - replace with actual implementations
class Logger:
    def __init__(self, log_dir, use_tb=True):
        self.log_dir = Path(log_dir)
        self.use_tb = use_tb
        
    def log(self, key, value, step):
        print(f"[{step}] {key}: {value}")
        
    def log_metrics(self, metrics, step, ty='train'):
        for key, value in metrics.items():
            self.log(f"{ty}/{key}", value, step)
            
    class log_and_dump_ctx:
        def __init__(self, step, ty):
            self.step = step
            self.ty = ty
            self.metrics = {}
            
        def __enter__(self):
            return lambda k, v: self.metrics.update({k: v})
            
        def __exit__(self, *args):
            for k, v in self.metrics.items():
                print(f"[{self.step}] {self.ty}/{k}: {v}")


class ReplayBuffer:
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.idx = 0
        self.size = 0
        
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.action = np.zeros((capacity, *action_shape), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.discount = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        
    def add(self, obs, action, reward, next_obs, discount, done):
        self.obs[self.idx] = obs
        self.action[self.idx] = action
        self.reward[self.idx] = reward
        self.next_obs[self.idx] = next_obs
        self.discount[self.idx] = discount
        self.done[self.idx] = done
        
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self):
        idxs = np.random.randint(0, self.size, size=self.batch_size)
        return (
            self.obs[idxs],
            self.action[idxs],
            self.reward[idxs],
            self.discount[idxs],
            self.next_obs[idxs]
        )
    
    def get_iterator(self):
        while True:
            yield self.sample()
            
    def __len__(self):
        return self.size


class VideoRecorder:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.enabled = False
        
    def init(self, enabled=True):
        self.enabled = enabled
        
    def record(self):
        if self.enabled:
            pass  # Implement actual recording
            
    def save(self, path):
        if self.enabled:
            pass  # Implement actual saving