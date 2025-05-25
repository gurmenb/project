import numpy as np
import gym
from gym import spaces
from typing import Dict, List, Tuple
import mujoco
import mujoco.viewer
from pipette_physics_simulation import PipettePhysicsSimulator, PipetteConfig, Particle, PipetteEnvironmentWrapper

class IntegratedPipetteEnv(gym.Env):
    """
    Integrated environment that combines MuJoCo visualization with physics-based 
    particle aspiration simulation for actor-critic training.
    """
    
    def __init__(self, mujoco_model_path: str = "particle_pipette_system.xml"):
        super().__init__()
        
        # Initialize MuJoCo (NEW API)
        self.model = mujoco.MjModel.from_xml_path(mujoco_model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        
        # Initialize physics simulator
        config = PipetteConfig(
            max_capacity=3,
            suction_range=0.03,
            min_suction_depth=0.02,
            max_suction_depth=0.08,
            suction_force_max=8.0
        )
        self.physics_sim = PipettePhysicsSimulator(config)
        self.env_wrapper = PipetteEnvironmentWrapper(self.physics_sim)
        
        # Define action and observation spaces
        # Actions: [x_pos, y_pos, z_pos, plunger_depth]
        self.action_space = spaces.Box(
            low=np.array([-0.4, -0.4, -0.3, 0.0]),
            high=np.array([0.4, 0.4, 0.1, 1.0]),
            dtype=np.float32
        )
        
        # Observation space includes both MuJoCo and physics states
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(26,),  # Adjust based on actual observation size
            dtype=np.float32
        )
        
        # Get joint and actuator indices
        self._setup_mujoco_indices()
        
        # Initialize particles based on XML configuration
        self._initialize_particles()
        
        # Episode tracking
        self.episode_step = 0
        self.max_episode_steps = 500
            
    def _setup_mujoco_indices(self):
        """Setup indices for MuJoCo joints and actuators"""
        import mujoco
        
        self.joint_indices = {
            'x': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'machine_x'),
            'y': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'machine_y'),
            'z': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'machine_z'),
            'plunger': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'plunger_joint')
        }
        
        self.actuator_indices = {
            'x': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'x_control'),
            'y': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'y_control'),
            'z': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'z_control'),
            'plunger': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'plunger_control')
        }
        
        # Get particle body indices
        self.particle_body_indices = {}
        for i in range(1, 7):  # well 1 particles
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f'particle_1_{i}')
                self.particle_body_indices[f'particle_1_{i}'] = body_id
            except:
                pass
        
        for i in range(1, 5):  # well 2 particles
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f'particle_2_{i}')
                self.particle_body_indices[f'particle_2_{i}'] = body_id
            except:
                pass
        
        for i in range(1, 6):  # loose particles
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f'loose_particle_{i}')
                self.particle_body_indices[f'loose_particle_{i}'] = body_id
            except:
                pass
    
    def _initialize_particles(self):
        """Initialize physics simulation particles based on MuJoCo model"""
        particle_id = 0
        
        # Well 1 particles
        well_1_center = np.array([-0.3, -0.3, 0.017])
        for i, offset in enumerate([
            np.array([-0.015, -0.015, 0]), np.array([0.015, -0.015, 0]),
            np.array([-0.015, 0.015, 0]), np.array([0.015, 0.015, 0]),
            np.array([0, 0, 0]), np.array([0, 0, 0.018])
        ]):
            pos = well_1_center + offset
            self.physics_sim.add_particle(pos, particle_id, well_id=1)
            particle_id += 1
        
        # Well 2 particles
        well_2_center = np.array([0, -0.3, 0.017])
        for i, offset in enumerate([
            np.array([-0.015, -0.015, 0]), np.array([0.015, -0.015, 0]),
            np.array([-0.015, 0.015, 0]), np.array([0.015, 0.015, 0])
        ]):
            pos = well_2_center + offset
            self.physics_sim.add_particle(pos, particle_id, well_id=2)
            particle_id += 1
        
        # Container particles
        container_center = np.array([0.4, 0.2, 0.065])
        for i, offset in enumerate([
            np.array([-0.02, -0.02, 0]), np.array([0.02, -0.02, 0]),
            np.array([-0.02, 0.02, 0]), np.array([0.02, 0.02, 0]),
            np.array([0, 0, 0.01])
        ]):
            pos = container_center + offset
            self.physics_sim.add_particle(pos, particle_id, well_id=None)
            particle_id += 1
    
    def _sync_particles_to_mujoco(self):
        """Synchronize physics simulation particle positions with MuJoCo"""
        for i, particle in enumerate(self.physics_sim.particles):
            body_name = self._get_mujoco_body_name(i)
            if body_name in self.particle_body_indices:
                body_id = self.particle_body_indices[body_name]
                
                if not particle.is_held:
                    # Update MuJoCo body position (NEW API - different indexing)
                    joint_adr = self.model.body_jntadr[body_id]
                    if joint_adr >= 0:
                        self.data.qpos[joint_adr:joint_adr+3] = particle.position
                        self.data.qvel[joint_adr:joint_adr+3] = particle.velocity

    def _get_mujoco_body_name(self, particle_index: int) -> str:
        """Map physics particle index to MuJoCo body name"""
        if particle_index < 6:
            return f'particle_1_{particle_index + 1}'
        elif particle_index < 10:
            return f'particle_2_{particle_index - 5}'
        else:
            return f'loose_particle_{particle_index - 9}'
    
    def _get_observation(self) -> np.ndarray:
        """Get combined observation from MuJoCo and physics simulation"""
        # MuJoCo state - FIXED: Changed all self.sim.data to self.data
        joint_positions = np.array([
            self.data.qpos[self.joint_indices['x']],
            self.data.qpos[self.joint_indices['y']],
            self.data.qpos[self.joint_indices['z']],
            self.data.qpos[self.joint_indices['plunger']]
        ])
        
        joint_velocities = np.array([
            self.data.qvel[self.joint_indices['x']],
            self.data.qvel[self.joint_indices['y']],
            self.data.qvel[self.joint_indices['z']],
            self.data.qvel[self.joint_indices['plunger']]
        ])
        
        # Physics simulation state
        physics_state = self.physics_sim.get_state_dict()
        
        # Compile observation vector
        obs = np.concatenate([
            joint_positions,                                    # 4 values
            joint_velocities,                                   # 4 values
            physics_state['pipette_tip_position'],              # 3 values
            [physics_state['plunger_position']],                # 1 value
            [physics_state['held_particle_count']],             # 1 value
            [physics_state['nearby_particle_count']],           # 1 value
            [physics_state['suction_pressure']],                # 1 value
            [float(physics_state['pipette_state'] == 'aspirating')],  # 1 value
            [float(physics_state['pipette_state'] == 'dispensing')],  # 1 value
            [float(physics_state['pipette_state'] == 'holding')],     # 1 value
            
            # Particle distance information (top 8 nearest)
            self._get_particle_distance_features(physics_state, 8)  # 8 values
        ])
        
        return obs.astype(np.float32)
    
    def _get_particle_distance_features(self, physics_state: Dict, max_particles: int) -> np.ndarray:
        """Extract distance features for nearest particles"""
        particles_in_range = physics_state.get('particles_in_range', [])
        features = np.zeros(max_particles)
        
        for i, particle_info in enumerate(particles_in_range[:max_particles]):
            features[i] = particle_info['distance']
        
        return features
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        # Apply action to MuJoCo
        self.data.ctrl[self.actuator_indices['x']] = action[0]
        self.data.ctrl[self.actuator_indices['y']] = action[1]
        self.data.ctrl[self.actuator_indices['z']] = action[2]
        
        # Step MuJoCo simulation (NEW API)
        mujoco.mj_step(self.model, self.data)
        
        # Get current MuJoCo state
        mujoco_state = {
            'x_pos': self.data.qpos[self.joint_indices['x']],
            'y_pos': self.data.qpos[self.joint_indices['y']],
            'z_pos': self.data.qpos[self.joint_indices['z']]
        }

        
        # Step physics simulation with RL-controlled plunger
        physics_obs, physics_reward, physics_done, physics_info = self.env_wrapper.step(mujoco_state, action)
        
        # Synchronize particle positions
        self._sync_particles_to_mujoco()
        
        # Get combined observation
        observation = self._get_observation()
        
        # Calculate combined reward
        reward = self._calculate_reward(physics_reward, physics_info)
        
        # Check if episode is done
        self.episode_step += 1
        done = (self.episode_step >= self.max_episode_steps or 
                physics_done or 
                self._check_task_completion())
        
        # Compile info
        info = {
            'physics_info': physics_info,
            'episode_step': self.episode_step,
            'particles_held': len(self.physics_sim.held_particles),
            'pipette_state': self.physics_sim.pipette_state.value,
            'task_completed': self._check_task_completion()
        }
        
        return observation, reward, done, info
    
    def _calculate_reward(self, physics_reward: float, physics_info: Dict) -> float:
        """Calculate combined reward including task-specific objectives"""
        total_reward = physics_reward
        
        # Task completion bonus
        if self._check_task_completion():
            total_reward += 10.0
        
        # Efficiency rewards
        particles_held = len(self.physics_sim.held_particles)
        if particles_held > 0:
            total_reward += 0.1 * particles_held
        
        # Penalize excessive movement - FIXED: Changed self.sim.data to self.data
        joint_vels = np.array([
            self.data.qvel[self.joint_indices['x']],
            self.data.qvel[self.joint_indices['y']],
            self.data.qvel[self.joint_indices['z']]
        ])
        movement_penalty = np.sum(np.square(joint_vels)) * 0.01
        total_reward -= movement_penalty
        
        return total_reward
    
    def _check_task_completion(self) -> bool:
        """Check if the pipetting task has been completed successfully"""
        # Example: Transfer particles from well 1 to well 3
        well_3_pos = np.array([0.3, -0.3, 0.01])
        particles_in_well_3 = 0
        
        for particle in self.physics_sim.particles:
            if (particle.well_id == 1 and  # Originally from well 1
                np.linalg.norm(particle.position[:2] - well_3_pos[:2]) < 0.05):
                particles_in_well_3 += 1
        
        return particles_in_well_3 >= 2  # Success if 2+ particles transferred
    
    def reset(self):
        """Reset the environment - ADDED this method"""
        # Reset MuJoCo simulation (NEW API)
        mujoco.mj_resetData(self.model, self.data)
        
        # Reset physics simulation
        self.physics_sim.reset()
        
        # Re-initialize particles
        self._initialize_particles()
        
        # Reset episode step counter
        self.episode_step = 0
        
        return self._get_observation()
    
    def reset_model(self):
        """Reset the environment"""
        # Reset MuJoCo simulation (NEW API)
        mujoco.mj_resetData(self.model, self.data)
        
        # Reset physics simulation
        self.physics_sim.reset()
        
        # Re-initialize particles
        self._initialize_particles()
        
        return self._get_observation()

    def render(self, mode='human'):
        """Render the environment"""
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        # Update viewer
        self.viewer.sync()
        
        if mode == 'rgb_array':
            # For recording (more complex with new API)
            return self._render_rgb_array()

    def _render_rgb_array(self):
        """Render RGB array for recording"""
        import mujoco.viewer
        
        # Create offscreen renderer
        renderer = mujoco.Renderer(self.model, height=480, width=640)
        renderer.update_scene(self.data)
        return renderer.render()
    
    def close(self):
        """Close the environment"""
        if self.viewer:
            self.viewer.close()
            self.viewer = None

# Example training script integration
def create_actor_critic_compatible_env():
    """Create environment compatible with actor-critic algorithms like PPO, A2C, etc."""
    return IntegratedPipetteEnv()

if __name__ == "__main__":
    # Example usage
    env = create_actor_critic_compatible_env()
    
    # Test random actions
    obs = env.reset()
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        
        print(f"Step {i}: Reward={reward:.3f}, Held={info['particles_held']}, State={info['pipette_state']}")
        
        if done:
            print("Episode finished!")
            obs = env.reset()
    
    env.close()