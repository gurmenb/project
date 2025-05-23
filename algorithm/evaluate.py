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
import json
import matplotlib.pyplot as plt
from typing import List, Dict

import utils
from reward_function import create_reward_function
from train import make_agent


class Evaluator:
    """Evaluation class for pipetting RL agent."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        utils.set_seed_everywhere(cfg.seed)
        
        # Create environment
        # NOTE: Replace with your actual environment
        # self.env = BallTransferEnv(**cfg.env)
        
        # For now, create dummy env
        from types import SimpleNamespace
        self.env = SimpleNamespace(
            observation_spec=lambda: SimpleNamespace(shape=(21,)),
            action_spec=lambda: SimpleNamespace(shape=(8,)),
            reset=lambda: SimpleNamespace(observation=np.zeros(21), reward=0, discount=1.0, last=lambda: False),
            step=lambda a: SimpleNamespace(observation=np.zeros(21), reward=0, discount=0.99, last=lambda: np.random.rand() < 0.01)
        )
        
        # Create agent
        self.agent = make_agent(
            self.env.observation_spec(),
            self.env.action_spec(),
            cfg.agent
        )
        
        # Create reward function
        self.reward_fn = create_reward_function(cfg.reward)
        
        # Results storage
        self.results = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rate': 0.0,
            'avg_reward': 0.0,
            'avg_length': 0.0,
            'reward_components': {}
        }
        
    def load_agent(self, checkpoint_path: str):
        """Load trained agent from checkpoint."""
        print(f"Loading agent from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'agent' in checkpoint:
            self.agent = checkpoint['agent']
        else:
            # Load individual components
            self.agent.load(checkpoint_path)
            
    def evaluate_episode(self, render: bool = False) -> Dict:
        """Run single evaluation episode."""
        episode_reward = 0
        episode_length = 0
        reward_components_sum = {}
        
        time_step = self.env.reset()
        self.reward_fn.reset()
        
        while not time_step.last():
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation, eval_mode=True)
            
            next_time_step = self.env.step(action)
            
            # Compute reward
            # NOTE: In real implementation, get proper info from environment
            info = {
                'task_phase': 'pick',
                'balls_in_gripper': np.random.randint(0, 5),
                'target_balls': 5,
                'balls_transferred': 0,
                'task_completed': False
            }
            
            reward, components = self.reward_fn.compute_reward(
                {'gripper_height': 0.2},
                action,
                {'gripper_height': 0.2},
                info
            )
            
            episode_reward += reward
            episode_length += 1
            
            # Accumulate reward components
            for key, value in components.items():
                if key not in reward_components_sum:
                    reward_components_sum[key] = 0
                reward_components_sum[key] += value
            
            if render:
                # Render environment
                pass
                
            time_step = next_time_step
            
        # Average reward components
        reward_components_avg = {k: v/episode_length for k, v in reward_components_sum.items()}
        
        return {
            'reward': episode_reward,
            'length': episode_length,
            'success': info.get('task_completed', False),
            'reward_components': reward_components_avg
        }
    
    def evaluate(self, num_episodes: int = 100, render: bool = False):
        """Run full evaluation."""
        print(f"Evaluating for {num_episodes} episodes...")
        
        successes = 0
        
        for episode in range(num_episodes):
            results = self.evaluate_episode(render=(render and episode == 0))
            
            self.results['episode_rewards'].append(results['reward'])
            self.results['episode_lengths'].append(results['length'])
            
            if results['success']:
                successes += 1
                
            # Accumulate reward components
            for key, value in results['reward_components'].items():
                if key not in self.results['reward_components']:
                    self.results['reward_components'][key] = []
                self.results['reward_components'][key].append(value)
                
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{num_episodes} - "
                      f"Reward: {results['reward']:.2f}, "
                      f"Length: {results['length']}")
        
        # Compute statistics
        self.results['success_rate'] = successes / num_episodes
        self.results['avg_reward'] = np.mean(self.results['episode_rewards'])
        self.results['avg_length'] = np.mean(self.results['episode_lengths'])
        self.results['std_reward'] = np.std(self.results['episode_rewards'])
        self.results['std_length'] = np.std(self.results['episode_lengths'])
        
        # Average reward components
        for key in self.results['reward_components']:
            values = self.results['reward_components'][key]
            self.results['reward_components'][key] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
            
    def save_results(self, save_path: str):
        """Save evaluation results."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        print(f"Results saved to: {save_path}")
        
    def plot_results(self, save_dir: str):
        """Create evaluation plots."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Episode rewards plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.results['episode_rewards'])
        plt.axhline(y=self.results['avg_reward'], color='r', linestyle='--', 
                   label=f"Mean: {self.results['avg_reward']:.2f}")
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Episode Rewards')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_dir / 'episode_rewards.png')
        plt.close()
        
        # Reward components bar plot
        if self.results['reward_components']:
            components = list(self.results['reward_components'].keys())
            means = [self.results['reward_components'][c]['mean'] for c in components]
            stds = [self.results['reward_components'][c]['std'] for c in components]
            
            plt.figure(figsize=(12, 6))
            x = np.arange(len(components))
            plt.bar(x, means, yerr=stds, capsize=5)
            plt.xticks(x, components, rotation=45, ha='right')
            plt.xlabel('Reward Component')
            plt.ylabel('Average Value')
            plt.title('Average Reward Components')
            plt.grid(True, axis='y')
            plt.tight_layout()
            plt.savefig(save_dir / 'reward_components.png')
            plt.close()
            
        print(f"Plots saved to: {save_dir}")
        
    def print_summary(self):
        """Print evaluation summary."""
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Success Rate: {self.results['success_rate']*100:.1f}%")
        print(f"Average Reward: {self.results['avg_reward']:.2f} ± {self.results['std_reward']:.2f}")
        print(f"Average Episode Length: {self.results['avg_length']:.1f} ± {self.results['std_length']:.1f}")
        
        if self.results['reward_components']:
            print("\nReward Components (mean ± std):")
            for comp, stats in self.results['reward_components'].items():
                print(f"  {comp}: {stats['mean']:.3f} ± {stats['std']:.3f}")
        print("="*50 + "\n")


@hydra.main(config_path='cfgs', config_name='config', version_base="1.1")
def main(cfg: DictConfig):
    # Override some configs for evaluation
    cfg.num_eval_episodes = getattr(cfg, 'num_eval_episodes', 100)
    
    evaluator = Evaluator(cfg)
    
    # Load checkpoint
    checkpoint_path = Path(cfg.checkpoint_path) if hasattr(cfg, 'checkpoint_path') else Path.cwd() / 'snapshot.pt'
    if checkpoint_path.exists():
        evaluator.load_agent(str(checkpoint_path))
    else:
        print(f"Warning: No checkpoint found at {checkpoint_path}, using random agent")
    
    # Run evaluation
    evaluator.evaluate(num_episodes=cfg.num_eval_episodes, render=cfg.get('render', False))
    
    # Save and display results
    evaluator.save_results('eval_results.json')
    evaluator.plot_results('eval_plots')
    evaluator.print_summary()


if __name__ == '__main__':
    main()