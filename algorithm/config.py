# config.py - Updated for MuJoCo Integration

config = {
    # System Settings
    'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu',  
    'seed': 42,
    'use_tb': True,  
    'save_snapshot': True,
    
    # MuJoCo Environment Settings (UPDATED)
    'mujoco_xml_path': 'particle_pipette_system.xml',
    'max_episode_steps': 500,  # Increased for complex physics simulation
    'render_mode': 'human',    # 'human' for visualization, 'rgb_array' for recording
    
    # Task Settings
    'target_volume': 20,
    'source_volume': 50,
    'success_threshold': 2,    # Number of particles to transfer for success
    
    # PPO Algorithm Settings
    'lr': 3e-4,
    'hidden_dim': 256,
    'clip_ratio': 0.2,
    'stddev_clip': 0.3,
    'buffer_size': 4096,       # Increased for more complex environment
    'batch_size': 128,         # Increased batch size
    'ppo_epochs': 10,
    'gamma': 0.99,
    'lam': 0.95,
    
    # Training Settings
    'num_train_frames': 200000,  # Increased for physics simulation
    'eval_every_frames': 10000,  # Less frequent evaluation
    'num_eval_episodes': 5,      # Fewer eval episodes (physics is slower)
    
    # Reward Function Weights (for optional custom rewards)
    'w_volume': 5.0,        # Reduced since MuJoCo has its own rewards
    'w_completion': 25.0,   # Reduced since MuJoCo has its own rewards
    'w_time': 0.005,        # Reduced time penalty
    'w_collision': 1.0,     # Reduced collision penalty  
    'w_drop': 2.5,          # Reduced drop penalty
    'w_contamination': 1.5, # Reduced contamination penalty
    'w_miss': 1.0,          # Reduced miss penalty
    'w_jerk': 0.05,         # Reduced jerk penalty
    
    # MuJoCo-specific Settings
    'physics_timestep': 0.001,   # MuJoCo simulation timestep
    'control_timestep': 0.01,    # Control frequency (10x slower than physics)
    'enable_viewer': True,       # Show MuJoCo viewer during training
    'record_video': False,       # Set True to record training videos
    'video_frequency': 1000,     # Record every N episodes
}

def get_config():
    """Get a copy of the configuration"""
    return config.copy()

# MuJoCo Integration Notes
INTEGRATION_NOTES = {
    'observation_space': {
        'shape': (25,),  # Changed from (14,) 
        'components': [
            'joint_positions (4D): [x, y, z, plunger]',
            'joint_velocities (4D): [vx, vy, vz, v_plunger]', 
            'pipette_tip_position (3D): [tip_x, tip_y, tip_z]',
            'plunger_position (1D): plunger depth',
            'held_particle_count (1D): particles in pipette',
            'nearby_particle_count (1D): particles near tip',
            'suction_pressure (1D): current suction force',
            'pipette_state_flags (3D): [aspirating, holding, dispensing]',
            'particle_distances (8D): nearest particle distances'
        ]
    },
    'action_space': {
        'shape': (4,),  # Changed from (6,)
        'components': [
            'x_position: pipette x coordinate [-0.4, 0.4]',
            'y_position: pipette y coordinate [-0.4, 0.4]', 
            'z_position: pipette z coordinate [-0.3, 0.1]',
            'plunger_depth: plunger extension [0.0, 1.0]'
        ]
    },
    'reward_components': [
        'physics_reward: from particle aspiration/dispensing',
        'task_completion: bonus for successful transfer',
        'efficiency_reward: reward for holding particles',
        'movement_penalty: penalize excessive movement',
        'custom_rewards: optional additional reward shaping'
    ]
}

# Legacy configuration for backward compatibility
LEGACY_CONFIG = {
    'num_episodes': 100,
    'eval_frequency': 10,
    'save_frequency': 25,
    'use_tensorboard': True,
}