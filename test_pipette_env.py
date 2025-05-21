import numpy as np
from pipette_env import PipetteEnv
import time

def test_environment():
    # Create the environment
    env = PipetteEnv()
    
    # Reset the environment
    obs = env.reset()
    
    print("Observation space shape:", env.observation_space.shape)
    print("Action space shape:", env.action_space.shape)
    print("Initial observation:", obs)
    
    # Run a simple test sequence
    # 1. Move to source container
    # 2. Lower into source
    # 3. Aspirate
    # 4. Move to destination container
    # 5. Lower into destination
    # 6. Dispense
    
    # Define sequence of actions
    actions = [
        # Move above source container
        [0.3, 0, 0.3, 0],      # [x, y, z, plunger]
        [0.3, 0, 0.3, 0],
        [0.3, 0, 0.3, 0],
        
        # Lower into source
        [0.3, 0, 0.15, 0],
        [0.3, 0, 0.15, 0],
        
        # Aspirate
        [0.3, 0, 0.15, 1.0],
        [0.3, 0, 0.15, 1.0],
        [0.3, 0, 0.15, 1.0],
        
        # Raise from source
        [0.3, 0, 0.3, 0],
        [0.3, 0, 0.3, 0],
        
        # Move above destination
        [-0.3, 0, 0.3, 0],
        [-0.3, 0, 0.3, 0],
        [-0.3, 0, 0.3, 0],
        
        # Lower into destination
        [-0.3, 0, 0.15, 0],
        [-0.3, 0, 0.15, 0],
        
        # Dispense
        [-0.3, 0, 0.15, -1.0],
        [-0.3, 0, 0.15, -1.0],
        [-0.3, 0, 0.15, -1.0],
        
        # Raise from destination
        [-0.3, 0, 0.3, 0],
        [-0.3, 0, 0.3, 0],
    ]
    
    # Run sequence
    total_reward = 0
    for i, action in enumerate(actions):
        obs, reward, done, info = env.step(np.array(action))
        total_reward += reward
        
        print(f"Step {i+1}:")
        print(f"  Action: {action}")
        print(f"  Current volume: {info['volume']:.1f}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Total Reward: {total_reward:.2f}")
        
        # Render the environment
        env.render()
        time.sleep(0.1)  # Slow down visualization
        
        if done:
            print("Episode finished early!")
            break
    
    print(f"Final Total Reward: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    test_environment()