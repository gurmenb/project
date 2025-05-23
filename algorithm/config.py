# Configuration file for Pipetting RL Project

# Environment Configuration
ENV_CONFIG = {
    'max_episode_steps': 200,
    'target_volume': 20,           # Target number of balls to transfer
    'source_volume': 50,           # Initial balls in source well
    'action_bounds': [-1.0, 1.0], # Action space bounds
}

# PPO Agent Configuration
AGENT_CONFIG = {
    'learning_rate': 3e-4,
    'hidden_dim': 256,
    'clip_ratio': 0.2,             # PPO clipping parameter
    'value_coef': 0.5,             # Value function loss coefficient
    'entropy_coef': 0.01,          # Entropy bonus coefficient
    'max_grad_norm': 0.5,          # Gradient clipping
    'ppo_epochs': 10,              # Number of PPO epochs per update
    'batch_size': 64,              # Mini-batch size for PPO updates
}

# Reward Function Configuration
REWARD_CONFIG = {
    'w_volume': 10.0,              # Volume accuracy weight
    'w_time': 0.1,                 # Time efficiency penalty
    'w_completion': 5.0,           # Task completion bonus
    'w_collision': 2.0,            # Collision penalty
    'w_drop': 10.0,                # Ball dropping penalty
    'w_contamination': 3.0,        # Contamination penalty
    'w_miss': 5.0,                 # Missing target penalty
    'w_jerk': 1.0,                 # Jerky movement penalty
}

# Training Configuration
TRAINING_CONFIG = {
    'num_episodes': 1000,          # Total training episodes
    'buffer_size': 2048,           # PPO buffer size
    'eval_frequency': 50,          # Evaluate every N episodes
    'save_frequency': 100,         # Save checkpoint every N episodes
    'num_eval_episodes': 10,       # Episodes per evaluation
    'device': 'auto',              # 'cpu', 'cuda', or 'auto'
    'seed': 42,                    # Random seed
    'use_tensorboard': True,       # Enable tensorboard logging
}

# Observation Space Configuration (14D)
OBSERVATION_CONFIG = {
    'liquid_in_plunger': {'index': 0, 'type': 'continuous', 'range': [0, 1]},
    'balls_in_plunger': {'index': 1, 'type': 'continuous', 'range': [0, 50]},
    'source_well_amount': {'index': 2, 'type': 'continuous', 'range': [0, 50]},
    'target_well_amount': {'index': 3, 'type': 'continuous', 'range': [0, 50]},
    'source_well_position_x': {'index': 4, 'type': 'continuous', 'range': [-1, 1]},
    'source_well_position_y': {'index': 5, 'type': 'continuous', 'range': [-1, 1]},
    'source_well_position_z': {'index': 6, 'type': 'continuous', 'range': [-1, 1]},
    'target_well_position_x': {'index': 7, 'type': 'continuous', 'range': [-1, 1]},
    'target_well_position_y': {'index': 8, 'type': 'continuous', 'range': [-1, 1]},
    'target_well_position_z': {'index': 9, 'type': 'continuous', 'range': [-1, 1]},
    'aspiration_pressure': {'index': 10, 'type': 'continuous', 'range': [0, 2]},
    'dispersion_pressure': {'index': 11, 'type': 'continuous', 'range': [0, 2]},
    'task_phase': {'index': 12, 'type': 'discrete', 'range': [0, 3]},  # 0=approach, 1=aspirate, 2=transfer, 3=dispense
    'submerged': {'index': 13, 'type': 'continuous', 'range': [0, 1]},
}

# Action Space Configuration (6D)
ACTION_CONFIG = {
    'x_cube_pipette': {'index': 0, 'type': 'continuous', 'range': [-1, 1]},
    'y_cube_pipette': {'index': 1, 'type': 'continuous', 'range': [-1, 1]},
    'z_cube_pipette': {'index': 2, 'type': 'continuous', 'range': [-1, 1]},
    'z_position_plunger': {'index': 3, 'type': 'continuous', 'range': [-1, 1]},
    'z_plunger_force': {'index': 4, 'type': 'continuous', 'range': [-1, 1]},
    'z_plunger_speed': {'index': 5, 'type': 'continuous', 'range': [-1, 1]},
}

# Complete configuration combining all components
COMPLETE_CONFIG = {
    **ENV_CONFIG,
    **AGENT_CONFIG,
    **REWARD_CONFIG,
    **TRAINING_CONFIG,
    'observation_config': OBSERVATION_CONFIG,
    'action_config': ACTION_CONFIG,
}

# Easy access functions
def get_obs_dim():
    """Get observation space dimensionality"""
    return len(OBSERVATION_CONFIG)

def get_action_dim():
    """Get action space dimensionality"""
    return len(ACTION_CONFIG)

def get_config():
    """Get complete configuration"""
    return COMPLETE_CONFIG.copy()

def get_hyperparameter_variants():
    """Different hyperparameter configurations for experimentation"""
    
    base_config = get_config()
    
    variants = {
        'baseline': base_config,
        
        'high_lr': {
            **base_config,
            'learning_rate': 1e-3,
            'clip_ratio': 0.3,
        },
        
        'low_lr': {
            **base_config,
            'learning_rate': 1e-4,
            'ppo_epochs': 20,
        },
        
        'volume_focused': {
            **base_config,
            'w_volume': 20.0,
            'w_completion': 10.0,
            'w_time': 0.05,
        },
        
        'efficiency_focused': {
            **base_config,
            'w_time': 0.5,
            'w_jerk': 2.0,
            'entropy_coef': 0.005,
        },
        
        'large_batch': {
            **base_config,
            'buffer_size': 4096,
            'batch_size': 128,
            'ppo_epochs': 15,
        },
    }
    
    return variants

# Print configuration summary
def print_config_summary():
    """Print a summary of the current configuration"""
    config = get_config()
    
    print("="*50)
    print("PIPETTING RL CONFIGURATION SUMMARY")
    print("="*50)
    
    print(f"Observation Dimension: {get_obs_dim()}")
    print(f"Action Dimension: {get_action_dim()}")
    print(f"Target Volume: {config['target_volume']} balls")
    print(f"Max Episode Steps: {config['max_episode_steps']}")
    
    print("\nAgent Configuration:")
    print(f"  Learning Rate: {config['learning_rate']}")
    print(f"  Hidden Dimension: {config['hidden_dim']}")
    print(f"  PPO Clip Ratio: {config['clip_ratio']}")
    print(f"  PPO Epochs: {config['ppo_epochs']}")
    print(f"  Batch Size: {config['batch_size']}")
    
    print("\nReward Weights:")
    print(f"  Volume Accuracy: {config['w_volume']}")
    print(f"  Time Efficiency: {config['w_time']}")
    print(f"  Completion Bonus: {config['w_completion']}")
    print(f"  Collision Penalty: {config['w_collision']}")
    print(f"  Drop Penalty: {config['w_drop']}")
    
    print("\nTraining Configuration:")
    print(f"  Total Episodes: {config['num_episodes']}")
    print(f"  Buffer Size: {config['buffer_size']}")
    print(f"  Evaluation Frequency: {config['eval_frequency']}")
    print(f"  Save Frequency: {config['save_frequency']}")
    
    print("="*50)


if __name__ == '__main__':
    print_config_summary()