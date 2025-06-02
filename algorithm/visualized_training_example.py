#!/usr/bin/env python3
"""
Example of training with comprehensive visualization.
Shows how to integrate visualization into your existing training loops.
"""

import numpy as np
import time
from integrated_pipette_environment import IntegratedPipetteEnv
from pipette_visualization_tools import VisualizationWrapper, TrainingVisualizer

def train_with_mujoco_visualization():
    """Training example with MuJoCo + real-time plots"""
    print("Starting training with full visualization...")
    print("You'll see:")
    print("1. MuJoCo 3D simulation window")
    print("2. Real-time plots showing rewards, position, particles")
    print("3. Console output with training progress")
    
    # Create environment
    base_env = IntegratedPipetteEnv("particle_pipette_system.xml")
    
    # Wrap with visualization
    env = VisualizationWrapper(base_env)
    
    try:
        total_episodes = 50
        
        for episode in range(total_episodes):
            print(f"\n--- Episode {episode + 1}/{total_episodes} ---")
            
            obs = env.reset()
            episode_reward = 0
            episode_steps = 0
            
            # Episode loop
            for step in range(500):  # Max steps per episode
                # Simple policy for demonstration
                # You can replace this with your RL agent's action selection
                action = get_demo_action(step, env.physics_sim.task_phase.value)
                
                # Step environment with visualization
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                episode_steps += 1
                
                # Print detailed info every 20 steps
                if step % 20 == 0:
                    print(f"  Step {step}: Reward={reward:.3f}, "
                          f"Phase={info.get('task_phase', 'unknown')}, "
                          f"Particles={info.get('particles_held', 0)}, "
                          f"Well={info.get('current_well', 'none')}")
                
                # Small delay for visualization (remove for faster training)
                time.sleep(0.03)
                
                if done:
                    break
            
            print(f"Episode {episode + 1} completed:")
            print(f"  Total Reward: {episode_reward:.3f}")
            print(f"  Steps: {episode_steps}")
            print(f"  Task Completed: {info.get('task_completed', False)}")
            print(f"  Final Particles Held: {info.get('particles_held', 0)}")
            
            # Show reward breakdown
            if 'reward_breakdown' in info:
                breakdown = info['reward_breakdown']
                print(f"  Reward Breakdown:")
                for component, value in breakdown.items():
                    print(f"    {component}: {value:.3f}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Closing visualization...")
        env.close()

def get_demo_action(step: int, task_phase: str) -> np.ndarray:
    """
    Demo policy that shows a realistic pipetting sequence
    Replace this with your actual RL agent
    """
    
    # Normalize step for periodic behavior
    cycle_step = step % 200
    
    if task_phase == "approach_source" or cycle_step < 50:
        # Move to source well (well 1 at -0.12, 0)
        return np.array([-0.8, 0.0, -0.3, 0.0])  # Move left, slightly down
    
    elif task_phase == "aspirate" or (50 <= cycle_step < 80):
        # Lower and aspirate
        return np.array([-0.8, 0.0, -0.8, 0.8])  # Stay at well 1, lower, extend plunger
    
    elif task_phase == "transport" or (80 <= cycle_step < 130):
        # Move to target well (well 3 at 0.12, 0)
        progress = (cycle_step - 80) / 50.0
        x_action = -0.8 + 1.6 * progress  # Interpolate from -0.8 to 0.8
        return np.array([x_action, 0.0, 0.2, 0.8])  # Move right, raise pipette
    
    elif task_phase == "approach_target" or (130 <= cycle_step < 150):
        # Lower at target well
        return np.array([0.8, 0.0, -0.8, 0.8])  # Stay at well 3, lower
    
    elif task_phase == "dispense" or (150 <= cycle_step < 180):
        # Dispense
        return np.array([0.8, 0.0, -0.8, -0.8])  # Stay at well 3, retract plunger
    
    else:
        # Return to neutral
        return np.array([0.0, 0.0, 0.2, 0.0])

def train_with_plots_only():
    """Training example with plots only (no MuJoCo viewer)"""
    print("Starting training with plot-only visualization...")
    
    # Create environment
    env = IntegratedPipetteEnv("particle_pipette_system.xml")
    
    # Create custom visualizer (plots only)
    visualizer = TrainingVisualizer(
        model=None,  # No MuJoCo viewer
        data=None,
        use_mujoco=False,
        use_plots=True
    )
    
    try:
        for episode in range(20):
            obs = env.reset()
            episode_reward = 0
            
            for step in range(300):
                action = get_demo_action(step, env.physics_sim.task_phase.value)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                
                # Update visualization every few steps
                if step % 3 == 0:
                    visualizer.update(env, action, reward, info)
                
                if done:
                    break
            
            print(f"Episode {episode}: Reward = {episode_reward:.2f}")
    
    except KeyboardInterrupt:
        print("Training interrupted")
    finally:
        visualizer.close()
        env.close()

def debug_single_episode():
    """Debug a single episode with detailed state printing"""
    print("Debug mode: Single episode with detailed state information")
    
    env = IntegratedPipetteEnv("particle_pipette_system.xml")
    
    # Create visualizer
    visualizer = TrainingVisualizer(
        model=env.model,
        data=env.data,
        use_mujoco=True,
        use_plots=True
    )
    
    obs = env.reset()
    
    print("Initial State:")
    print_detailed_state(env, np.zeros(4), 0, {})
    
    try:
        for step in range(100):
            action = get_demo_action(step, env.physics_sim.task_phase.value)
            obs, reward, done, info = env.step(action)
            
            # Update visualization
            visualizer.update(env, action, reward, info)
            
            # Print detailed state every 10 steps
            if step % 10 == 0:
                print(f"\n--- Step {step} ---")
                print_detailed_state(env, action, reward, info)
            
            # Pause for inspection
            time.sleep(0.1)
            
            if done:
                print(f"\nEpisode completed at step {step}")
                break
    
    except KeyboardInterrupt:
        print("Debug interrupted")
    finally:
        visualizer.close()
        env.close()

def print_detailed_state(env, action, reward, info):
    """Print detailed state information for debugging"""
    physics_state = env.physics_sim.get_detailed_state_dict()
    
    print(f"Action: [{', '.join(f'{a:6.3f}' for a in action)}]")
    print(f"Reward: {reward:8.3f}")
    print(f"Pipette Position: ({physics_state['pipette_tip_position'][0]:6.3f}, "
          f"{physics_state['pipette_tip_position'][1]:6.3f}, "
          f"{physics_state['pipette_tip_position'][2]:6.3f})")
    print(f"Pipette State: {physics_state['pipette_state']:12s} | Task Phase: {physics_state['task_phase']}")
    print(f"Particles Held: {physics_state['held_particle_count']}/3 | Current Well: {physics_state.get('current_well', 'None')}")
    
    if 'reward_breakdown' in info:
        breakdown = info['reward_breakdown']
        print("Reward Breakdown:")
        for component, value in breakdown.items():
            if value != 0:
                print(f"  {component:25s}: {value:6.3f}")
    
    if 'recent_events' in info:
        events = info['recent_events']
        active_events = [k for k, v in events.items() if v > 0]
        if active_events:
            print(f"Recent Events: {', '.join(active_events)}")

def quick_test():
    """Quick test to verify visualization works"""
    print("Quick visualization test...")
    
    try:
        # Test with minimal setup
        from pipette_visualization_tools import quick_visualization_test
        quick_visualization_test()
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "full":
            train_with_mujoco_visualization()
        elif mode == "plots":
            train_with_plots_only()
        elif mode == "debug":
            debug_single_episode()
        elif mode == "test":
            quick_test()
        else:
            print("Usage: python visualized_training_example.py [full|plots|debug|test]")
    else:
        # Default: full visualization
        train_with_mujoco_visualization()