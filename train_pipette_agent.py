import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from pipette_env import PipetteEnv

# Create output directories
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

def main():
    # Create the vectorized environment
    # We'll use 4 parallel environments to speed up training
    env = make_vec_env(
        PipetteEnv,
        n_envs=4,
        monitor_dir="./logs"
    )
    
    # Create the agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="./logs"
    )
    
    # Set up the checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,  # Save every 10,000 steps
        save_path="./models/",
        name_prefix="pipette_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    # Train the agent
    print("Starting training...")
    model.learn(
        total_timesteps=100000,  # Adjust based on your time constraints
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save the final model
    model.save("models/pipette_final_model")
    print("Training complete. Final model saved.")
    
    # Evaluate the model
    print("Evaluating model...")
    # Create a separate environment for evaluation
    eval_env = Monitor(PipetteEnv())
    
    # Run evaluation
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Record a video of the trained agent
    print("Recording video of trained agent...")
    record_video(model)

def record_video(model, video_length=500):
    """Record a video of the trained agent."""
    try:
        # Create environment with video recording
        os.makedirs("videos", exist_ok=True)
        env = gym.wrappers.RecordVideo(
            PipetteEnv(render_mode="rgb_array"),
            video_folder="videos",
            episode_trigger=lambda x: True  # Record all episodes
        )
        
        # Run the model
        obs, info = env.reset()
        for _ in range(video_length):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            env.render()
            if terminated or truncated:
                obs, info = env.reset()
        
        env.close()
        print("Video recorded and saved to 'videos' directory")
    except Exception as e:
        print(f"Error recording video: {e}")
        print("Skipping video recording.")

if __name__ == "__main__":
    main()