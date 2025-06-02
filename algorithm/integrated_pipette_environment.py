#!/usr/bin/env python3

import numpy as np
import mujoco
import mujoco.viewer
from gymnasium import spaces
import os

# Import the physics simulation
from pipette_physics_simulation import PipettePhysicsWrapper, PipetteConfig


class IntegratedPipetteEnv:
    """
    Simple MuJoCo + Physics integrated environment for pipetting
    Keeps it basic but functional
    """
    
    def __init__(self, xml_path):
        # Load MuJoCo model
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"MuJoCo XML file not found: {xml_path}")
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Set up action and observation spaces
        # Actions: [x, y, z, plunger] positions - match your XML ranges
        self.action_space = spaces.Box(
            low=np.array([-0.15, -0.12, -0.15, 0.0]),  # Match XML joint ranges
            high=np.array([0.15, 0.12, 0.05, 0.08]),
            dtype=np.float32
        )
        
        # Observations: joint positions + velocities + some sensor data = 26D
        # 4 joint pos + 4 joint vel + 18 other sensors (3x6 for sites/particles)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(26,),
            dtype=np.float32
        )
        
        # Physics simulation
        self.physics_sim = PipettePhysicsWrapper(PipetteConfig())
        
        # Episode tracking
        self.step_count = 0
        self.max_episode_steps = 1000
        
        # Reset to initial state
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Reset physics simulation
        self.physics_sim.reset()
        
        # Reset step counter
        self.step_count = 0
        
        # Get initial observation
        obs = self._get_observation()
        return obs
    
    def step(self, action):
        """Step the environment with the given action"""
        # Ensure action is in correct format
        action = np.array(action, dtype=np.float32)
        
        # Apply action to MuJoCo simulation
        self.data.ctrl[:] = action
        
        # Step MuJoCo physics
        mujoco.mj_step(self.model, self.data)
        
        # Get sensor data from MuJoCo
        sensor_data = self._get_sensor_data()
        
        # Update physics simulation with MuJoCo data
        obs_dict, prev_state = self.physics_sim.update_from_mujoco(self.data, sensor_data)
        
        # Calculate reward from physics simulation
        reward, reward_components = self.physics_sim.calculate_reward(action)
        
        # Check if episode is done
        done = self.physics_sim.is_done() or self.step_count >= self.max_episode_steps
        
        # Get observation for RL agent
        obs = self._get_observation()
        
        # Create info dictionary - FIXED: held_particles is an int, not a list
        info = {
            'physics_reward': reward,
            'reward_components': reward_components,
            'task_completed': self.physics_sim.simulator.task_completed,
            'held_particle_count': self.physics_sim.simulator.held_particles,  # This is an int
            'particles_transferred': self.physics_sim.simulator.particles_transferred,
            'step_count': self.step_count
        }
        
        self.step_count += 1
        
        return obs, reward, done, info
    
    def _get_observation(self):
        """Get 26D observation vector"""
        obs = []
        
        # Joint positions (4D)
        joint_pos = self.data.qpos[:4] if len(self.data.qpos) >= 4 else np.zeros(4)
        obs.extend(joint_pos)
        
        # Joint velocities (4D)  
        joint_vel = self.data.qvel[:4] if len(self.data.qvel) >= 4 else np.zeros(4)
        obs.extend(joint_vel)
        
        # Sensor data (18D) - get what we can, pad with zeros if needed
        sensor_data = []
        for i in range(min(len(self.data.sensordata), 18)):
            sensor_data.append(self.data.sensordata[i])
        
        # Pad to 18D if needed
        while len(sensor_data) < 18:
            sensor_data.append(0.0)
        
        obs.extend(sensor_data)
        
        # Ensure we have exactly 26 dimensions
        obs = np.array(obs, dtype=np.float32)
        if len(obs) > 26:
            obs = obs[:26]
        elif len(obs) < 26:
            # Pad with zeros if too short
            padding = np.zeros(26 - len(obs))
            obs = np.concatenate([obs, padding])
        
        return obs
    
    def _get_sensor_data(self):
        """Get sensor data from MuJoCo in dictionary format"""
        sensor_data = {}
        
        # Map sensor indices to names (based on our XML)
        sensor_names = [
            'joint_pos_x', 'joint_pos_y', 'joint_pos_z', 'joint_pos_plunger',
            'joint_vel_x', 'joint_vel_y', 'joint_vel_z', 'joint_vel_plunger', 
            'site_suction_zone', 'site_well_1', 'site_well_2', 'site_well_3',
            'particle_pos_1_1', 'particle_pos_1_2', 'particle_pos_2_1', 'particle_pos_lp_1'
        ]
        
        # Extract sensor data
        sensor_idx = 0
        for name in sensor_names:
            if sensor_idx < len(self.data.sensordata):
                if 'pos' in name or 'site' in name or 'particle' in name:
                    # Position sensors are 3D
                    if sensor_idx + 2 < len(self.data.sensordata):
                        sensor_data[name] = self.data.sensordata[sensor_idx:sensor_idx+3].copy()
                        sensor_idx += 3
                    else:
                        sensor_data[name] = np.array([0.0, 0.0, 0.0])
                        break
                else:
                    # Scalar sensors
                    sensor_data[name] = np.array([self.data.sensordata[sensor_idx]])
                    sensor_idx += 1
            else:
                # Default values if sensor not available
                if 'pos' in name or 'site' in name or 'particle' in name:
                    sensor_data[name] = np.array([0.0, 0.0, 0.0])
                else:
                    sensor_data[name] = np.array([0.0])
        
        return sensor_data
    
    def render(self, mode='human'):
        """Render the environment (optional)"""
        if mode == 'human':
            # Simple text output - FIXED: held_particles is an int
            print(f"Step: {self.step_count}, Held particles: {self.physics_sim.simulator.held_particles}")
        return None
    
    def close(self):
        """Clean up"""
        pass


# Simple test function
def test_environment():
    """Test the integrated environment"""
    print("Testing IntegratedPipetteEnv...")
    
    try:
        env = IntegratedPipetteEnv("particle_pipette_system.xml")
        print(f"✓ Environment created successfully")
        print(f"✓ Action space: {env.action_space}")
        print(f"✓ Observation space: {env.observation_space}")
        
        # Test reset
        obs = env.reset()
        print(f"✓ Reset successful, obs shape: {obs.shape}")
        
        # Test a few steps
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print(f"✓ Step {i}: reward={reward:.3f}, done={done}, obs_shape={obs.shape}")
            print(f"  Info: held={info['held_particle_count']}, transferred={info['particles_transferred']}")
            
            if done:
                break
        
        print("✓ Environment test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_environment()