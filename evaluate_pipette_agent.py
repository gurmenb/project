import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from pipette_env import PipetteEnv
import time

def evaluate_agent(model_path="models/pipette_final_model.zip", episodes=10, render=True):
    """Evaluate a trained agent and collect performance metrics."""
    # Load the model
    model = PPO.load(model_path)
    
    # Create the environment
    render_mode = "human" if render else None
    env = PipetteEnv(render_mode=render_mode)
    
    # Initialize statistics
    rewards = []
    volumes = []
    steps_to_complete = []
    success_rate = 0
    
    # Run evaluation episodes
    for episode in range(episodes):
        print(f"Episode {episode+1}/{episodes}")
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        done = False
        
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            if render:
                env.render()
                time.sleep(0.01)  # Slow down rendering
            
            # Store current volume for analysis
            volumes.append(info['volume'])
            
            # Check if episode is done
            done = terminated or truncated
            
            # Check if max steps reached
            if step_count >= env.max_steps:
                break
                
        # Record episode statistics
        rewards.append(episode_reward)
        steps_to_complete.append(step_count)
        
        # Check if task was successful
        if env._dispensed_correct_amount():
            success_rate += 1
    
    # Close environment
    env.close()
    
    # Calculate overall metrics
    success_rate = (success_rate / episodes) * 100
    avg_reward = np.mean(rewards)
    avg_steps = np.mean(steps_to_complete)
    
    # Print results
    print("\n--- Evaluation Results ---")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps to Complete: {avg_steps:.1f}")
    
    # Plot results
    plot_results(rewards, steps_to_complete, volumes)
    
    return {
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "avg_steps": avg_steps,
        "rewards": rewards,
        "steps": steps_to_complete,
        "volumes": volumes
    }

def plot_results(rewards, steps, volumes):
    """Plot evaluation metrics."""
    # Create directory for plots
    os.makedirs("results", exist_ok=True)
    
    # Plot rewards
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # Plot steps to complete
    plt.subplot(1, 2, 2)
    plt.plot(steps)
    plt.title('Steps to Complete')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    plt.tight_layout()
    plt.savefig('results/performance_metrics.png')
    
    # Plot volume changes over time
    plt.figure(figsize=(10, 5))
    plt.plot(volumes)
    plt.title('Pipette Volume Over Time')
    plt.xlabel('Step')
    plt.ylabel('Volume (μL)')
    plt.savefig('results/volume_profile.png')
    
    plt.close('all')

def compare_with_baseline():
    """Compare trained agent with a simple rule-based baseline."""
    # Create environment
    env = PipetteEnv(render_mode="human")
    
    # Run baseline policy (simplified sequence of actions)
    baseline_rewards = []
    baseline_steps = []
    baseline_success = 0
    
    episodes = 5
    for episode in range(episodes):
        print(f"Running baseline episode {episode+1}/{episodes}")
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        
        # Simple hard-coded policy
        # 1. Move to source
        for _ in range(3):
            action = np.array([0.3, 0, 0.3, 0])  # Move above source
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            env.render()
            time.sleep(0.1)
        
        # 2. Lower into source
        for _ in range(2):
            action = np.array([0.3, 0, 0.15, 0])  # Lower into source
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            env.render()
            time.sleep(0.1)
        
        # 3. Aspirate
        for _ in range(12):  # Need more attempts to reach target volume
            action = np.array([0.3, 0, 0.15, 1.0])  # Aspirate
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            env.render()
            time.sleep(0.1)
        
        # 4. Move up
        for _ in range(2):
            action = np.array([0.3, 0, 0.3, 0])  # Move up
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            env.render()
            time.sleep(0.1)
        
        # 5. Move to destination
        for _ in range(3):
            action = np.array([-0.3, 0, 0.3, 0])  # Move above destination
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            env.render()
            time.sleep(0.1)
        
        # 6. Lower into destination
        for _ in range(2):
            action = np.array([-0.3, 0, 0.15, 0])  # Lower into destination
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            env.render()
            time.sleep(0.1)
        
        # 7. Dispense
        for _ in range(12):  # Need more attempts to dispense full volume
            action = np.array([-0.3, 0, 0.15, -1.0])  # Dispense
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            env.render()
            time.sleep(0.1)
        
        # Record results
        baseline_rewards.append(episode_reward)
        baseline_steps.append(step_count)
        if env._dispensed_correct_amount():
            baseline_success += 1
    
    # Close environment
    env.close()
    
    # Calculate baseline metrics
    baseline_success_rate = (baseline_success / episodes) * 100
    baseline_avg_reward = np.mean(baseline_rewards)
    baseline_avg_steps = np.mean(baseline_steps)
    
    print("\n--- Baseline Results ---")
    print(f"Success Rate: {baseline_success_rate:.1f}%")
    print(f"Average Reward: {baseline_avg_reward:.2f}")
    print(f"Average Steps: {baseline_avg_steps:.1f}")
    