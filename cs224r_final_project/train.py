import os
import time
import datetime
from pathlib import Path
import torch
import numpy as np
from omegaconf import OmegaConf

from pipette_env import PipetteEnv
from ppo_agent import PPOAgent
from logger import Logger
from utils import set_seed_everywhere, Timer

def evaluate_agent(agent, cfg, num_episodes=5):

    eval_env = PipetteEnv(cfg)
    
    eval_rewards = []
    eval_lengths = []
    eval_successes = []
    
    for episode in range(num_episodes):
        obs, _ = eval_env.reset()  
        episode_reward = 0
        episode_length = 0
        
        while True:
            
            action, _, _ = agent.act(obs)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                success = eval_env._check_success()
                eval_successes.append(float(success))
                break
        
        eval_rewards.append(episode_reward)
        eval_lengths.append(episode_length)
    
    return {
        'episode_reward': np.mean(eval_rewards),
        'episode_length': np.mean(eval_lengths), 
        'episode_success': np.mean(eval_successes),
        'reward_std': np.std(eval_rewards),
        'success_rate': np.mean(eval_successes)
    }


def train_ppo_pipette(config_path="config.yaml"):
    cfg = OmegaConf.load(config_path)

    #create filename
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
    original_experiment = cfg.experiment
    unique_experiment = f"{original_experiment}_seed{cfg.seed}_{timestamp}"
    cfg.experiment = unique_experiment
    
    print("Loaded configuration:")
    print(OmegaConf.to_yaml(cfg))
    print(f"\nrun identifier: {unique_experiment}")
    
    #device setup
    if cfg.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg.device
    print(f"device: {device}")
    
    set_seed_everywhere(cfg.seed)
    print(f"seed: {cfg.seed}")
    
    
    log_dir = Path(cfg.log_dir) / cfg.experiment
    log_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, log_dir / "config.yaml")
    
    # Initialize logger
    logger = Logger(log_dir, cfg.use_tb)
    
    #create env and pass config to environment
    env = PipetteEnv(cfg)
    print(f"env created: {env.__class__.__name__}")
    print(f"obs: {env.observation_space}")
    print(f"action space: {env.action_space}")
    
    #create agent
    agent = PPOAgent(cfg)
    
    #training variables
    timer = Timer()
    global_step = 0
    epoch = 0
    episode = 0
    
    # Training loop
    print("\nBEGIN TRAINING:")
    
    while global_step < cfg.num_train_frames:
        epoch += 1
        epoch_start_time = time.time()
        
        #one epoch
        epoch_episodes = 0
        while not agent.ready_to_update() and global_step < cfg.num_train_frames:
            episode += 1
            epoch_episodes += 1
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            
            #run episode
            while True:
                action, value, logp = agent.act(obs)
                
                #env step
                next_obs, reward, terminated, truncated, _ = env.step(action)
                
                #store experience
                if not agent.ready_to_update():
                    agent.store_experience(obs, action, reward, value, logp)
                
                #update counters
                global_step += 1
                episode_reward += reward
                episode_length += 1
                obs = next_obs
                
                if terminated or truncated:

                    #get last state estimate
                    if truncated:
                        _, last_val, _ = agent.act(next_obs)
                    else:
                        last_val = 0.0
                    
                    agent.finish_episode(last_val)
                    
                    #log things
                    success = env._check_success()
                    logger.log('train/episode', episode, global_step)
                    logger.log('train/step', global_step, global_step) 
                    logger.log('train/episode_reward', episode_reward, global_step)
                    logger.log('train/episode_length', episode_length, global_step)
                    logger.log('train/episode_success', float(success), global_step)
                    logger.log('train/buffer_size', agent.buffer.size(), global_step)
                    
                    #logger process
                    if episode % 10 == 0:
                        aspirated = sum(env.aspirated_droplets)
                        dispensed = sum(env.droplets_in_target) 
                        print(f"Episode {episode}: Reward={episode_reward:.1f}, Length={episode_length}, A={aspirated}/3, D={dispensed}/3, Success={success}")
                    
                    break
                
                if global_step >= cfg.num_train_frames or agent.ready_to_update():
                    break
            
            if global_step >= cfg.num_train_frames:
                break
        
        #update agent when buffer is full
        if agent.ready_to_update():
            print(f"\nEpoch {epoch}:updating agent...")
            agent.update()
            
            train_stats = agent.get_training_stats()
            for key, value in train_stats.items():
                logger.log(f'train/{key}', value, global_step)
            
            logger.log('train/episode', episode, global_step)
            logger.log('train/step', global_step, global_step)
            logger.log('train/frame', global_step, global_step)
            logger.log('train/buffer_size', 0, global_step) 
            
            # Log timing and throughput
            epoch_time = time.time() - epoch_start_time
            fps = cfg.steps_per_epoch / epoch_time
            logger.log('train/fps', fps, global_step)
            logger.log('train/epoch_time', epoch_time, global_step)
            
            # Calculate total elapsed time
            elapsed_time, total_time = timer.reset()
            logger.log('train/total_time', total_time, global_step)
            
            print(f"Epoch {epoch} completed:")
            print(f"  Steps: {global_step}/{cfg.num_train_frames}")
            print(f"  Episodes: {episode}")
            print(f"  FPS: {fps:.1f}")
            print(f"  Actor Loss: {train_stats['actor_loss']:.4f}")
            print(f"  Critic Loss: {train_stats['critic_loss']:.4f}")
            print(f"  KL Divergence: {train_stats['kl_divergence']:.4f}")
            print(f"  Clip Fraction: {train_stats['clip_fraction']:.3f}")
            print(f"  Training Iterations: {train_stats['training_iterations']}")
        
        #evaluation
        if epoch % cfg.eval_every_frames == 0:
            print(f"\nEvaluating agent at epoch {epoch}...")
            eval_metrics = evaluate_agent(agent, cfg, cfg.num_eval_episodes)
            
            # Log evaluation metrics
            for key, value in eval_metrics.items():
                logger.log(f'eval/{key}', value, global_step)
            
            # Add required fields for eval CSV
            logger.log('eval/episode', episode, global_step)
            logger.log('eval/step', global_step, global_step)
            logger.log('eval/frame', global_step, global_step)
            logger.log('eval/total_time', timer.total_time(), global_step)
            
            print(f"Evaluation Results:")
            print(f"  Average Reward: {eval_metrics['episode_reward']:.2f} ± {eval_metrics['reward_std']:.2f}")
            print(f"  Average Length: {eval_metrics['episode_length']:.1f}")
            print(f"  Success Rate: {eval_metrics['success_rate']:.2%}")
        
        # Save model
        if epoch % cfg.save_frequency == 0:
            save_path = log_dir / f"agent_epoch_{epoch}.pt"
            agent.save(save_path)
            print(f"Model saved to {save_path}")
        
        # Dump logs
        if epoch % cfg.log_frequency == 0:
            logger.dump(global_step, 'train')
            logger.dump(global_step, 'eval')

    print("\nDONE")
    final_save_path = log_dir / "agent_final.pt"
    agent.save(final_save_path)
    
    final_eval = evaluate_agent(agent, cfg, cfg.num_eval_episodes * 2)  
    for key, value in final_eval.items():
        logger.log(f'eval/{key}', value, global_step)
    
    logger.log('eval/episode', episode, global_step)
    logger.log('eval/step', global_step, global_step)
    logger.log('eval/frame', global_step, global_step)
    logger.log('eval/total_time', timer.total_time(), global_step)
    
    print(f"Final Results:")
    print(f"  Average Reward: {final_eval['episode_reward']:.2f}")
    print(f"  Success Rate: {final_eval['success_rate']:.2%}")
    
    total_time = timer.total_time()
    print(f"Total training time: {total_time:.1f} seconds")
    print(f"Logs saved to: {log_dir}")
    
    print(f"\nTB instructions:")
    print(f"  tensorboard --logdir {cfg.log_dir}")
    print(f"  Then go to: http://localhost:6006")
    print(f"  Your run will be named: {unique_experiment}")
    
    # Final log dump
    logger.dump(global_step, 'eval')
    
    return agent, final_eval


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--log-dir", type=str, default=None,
                       help="Override log directory")
    parser.add_argument("--seed", type=int, default=None,
                       help="Override random seed")
    parser.add_argument("--name", type=str, default=None,
                       help="Custom experiment name suffix")
    
    args = parser.parse_args()
    
    # Load and potentially override config
    cfg = OmegaConf.load(args.config)
    if args.log_dir:
        cfg.log_dir = args.log_dir
    if args.seed:
        cfg.seed = args.seed
    
    # Add custom name suffix if provided
    if args.name:
        cfg.experiment = f"{cfg.experiment}_{args.name}"
    
    # Save modified config temporarily
    temp_config = "temp_config.yaml"
    OmegaConf.save(cfg, temp_config)
    
    try:
        # Run training
        agent, results = train_ppo_pipette(temp_config)
        print("\nTraining completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        
    except Exception as e:
        print(f"\Training failed with error: {e}")
        raise
        
    finally:
        if os.path.exists(temp_config):
            os.remove(temp_config)