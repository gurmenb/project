# config.yaml - Configuration for Pipetting RL with Ball Transfer

defaults:
  - agent: ppo

# experiment
experiment: pipetting_ball_transfer
seed: 1
device: cuda
use_tb: true
save_video: true
save_snapshot: true

# environment - simplified for ball transfer
env:
  name: BallTransferEnv
  num_balls_range: [3, 10]  # Random number of balls to transfer
  container_size: 0.1  # meters
  workspace_bounds: 
    min: [-0.5, -0.5, 0.0]
    max: [0.5, 0.5, 0.5]
  max_episode_steps: 200

# training
num_train_frames: 1000000
num_seed_frames: 1000
eval_every_frames: 10000
num_eval_episodes: 10
warmup: 1000

# replay buffer
replay_buffer_size: 1000000
replay_buffer_num_workers: 2
batch_size: 256
nstep: 1
discount: 0.99

# agent configuration
agent:
  name: ppo_agent.PPOAgent
  obs_shape: ???  # to be specified by env
  action_shape: ???  # to be specified by env
  device: ${device}
  lr: 3e-4
  critic_lr: 3e-4
  hidden_dim: 256
  num_layers: 3
  num_critics: 2
  critic_target_tau: 0.005
  stddev_clip: 0.3
  use_tb: ${use_tb}
  
  # PPO specific
  ppo_epoch: 10
  ppo_clip: 0.2
  value_loss_coef: 0.5
  entropy_coef: 0.01
  max_grad_norm: 0.5
  gae_lambda: 0.95
  
  # Actor std
  std_min: 0.1
  std_max: 1.0
  
  # Update frequencies
  update_every_steps: 2
  actor_update_every_steps: 2
  num_updates: 1
  
# reward function weights
reward:
  volume_accuracy: 10.0
  time_efficiency: 0.01
  completion_bonus: 50.0
  collision_penalty: 5.0
  drop_penalty: 10.0
  contamination_penalty: 3.0
  miss_penalty: 5.0
  jerk_penalty: 0.1
  height_penalty: 0.5

# logging
log_every_step: 1000
log_every_episode: 1

hydra:
  run:
    dir: ./exp/${now:%Y.%m.%d}/${now:%H%M%S}_${experiment}
  sweep:
    dir: ./exp/${now:%Y.%m.%d}/${now:%H%M}_${experiment}
    subdir: ${hydra.job.num}