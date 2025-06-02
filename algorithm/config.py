#!/usr/bin/env python3

def get_config():
    """
    Simple configuration for PPO pipetting training
    """
    cfg = {
        # Environment settings
        'seed': 42,
        'device': 'auto',  # Will auto-detect GPU/CPU
        
        # Training settings  
        'num_train_frames': 1000000,
        'eval_every_frames': 50000,
        'num_eval_episodes': 10,
        'max_episode_length': 500,
        
        # PPO settings
        'lr': 3e-4,
        'hidden_dim': 256,
        'clip_ratio': 0.2,
        'stddev_clip': 0.3,
        
        # Buffer settings
        'buffer_size': 1024,
        'batch_size': 64,
        'ppo_epochs': 10,
        'gamma': 0.99,
        'lam': 0.95,
        
        # Logging
        'use_tb': False,  # Disable tensorboard for simplicity
        'save_snapshot': False,  # Disable frequent saving during training
        'save_every_frames': 1000000,  # Only save occasionally
        
        # Training control
        'max_episodes_per_round': 10,
    }
    
    return cfg