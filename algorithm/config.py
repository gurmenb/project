# config.py - Fixed to match train_pipette.py expectations

config = {
    # System Settings
    'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu',  
    'seed': 42,
    'use_tb': True,  
    'save_snapshot': True,
    
    # Environment Settings
    'max_episode_steps': 200,
    'target_volume': 20,
    'source_volume': 50,
    
    # PPO Algorithm Settings
    'lr': 3e-4,
    'hidden_dim': 256,
    'clip_ratio': 0.2,
    'stddev_clip': 0.3,
    'buffer_size': 2048,
    'batch_size': 64,
    'ppo_epochs': 10,
    'gamma': 0.99,
    'lam': 0.95,
    
    # Training Settings
    'num_train_frames': 100000,  
    'eval_every_frames': 5000,   
    'num_eval_episodes': 10,
    
    # Reward Function Weights 
    'w_volume': 10.0,
    'w_completion': 50.0,
    'w_time': 0.01,
    'w_collision': 2.0,
    'w_drop': 5.0,
    'w_contamination': 3.0,
    'w_miss': 2.0,
    'w_jerk': 0.1,
}

def get_config():
    """Get a copy of the configuration"""
    return config.copy()

LEGACY_CONFIG = {
    'num_episodes': 100,
    'eval_frequency': 10,
    'save_frequency': 25,
    'use_tensorboard': True,
}