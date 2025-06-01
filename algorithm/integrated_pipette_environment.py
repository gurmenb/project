import numpy as np
import gym
from gym import spaces
from typing import Dict, List, Tuple
import mujoco_py
from pipette_physics_simulation import PipettePhysicsSimulator, PipetteConfig, Particle, PipetteEnvironmentWrapper

class IntegratedPipetteEnv(gym.Env):
    """
    Integrated environment that combines MuJoCo visualization with physics-based 
    particle aspiration simulation for actor-critic training.
    """
    
    def __init__(self, mujoco_model_path: str = "particle_pipette_system.xml"):
        super().__init__()
        
        # Initialize MuJoCo
        self.model = mujoco_py.load_model_from_path(mujoco_model_path)
        self.sim = mujoco_py.MjSim(self.model)
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
        # UPDATED: Action space to match XML actuator ranges
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0]),   # Normalized -1 to 1
            high=np.array([1.0, 1.0, 1.0, 1.0]),      # Will be scaled to XML ranges
            dtype=np.float32
        )
        
        # Observation space includes both MuJoCo and physics states
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(25,),  # Adjust based on actual observation size
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
        self.joint_indices = {
            'x': self.model.joint_name2id('machine_x'),
            'y': self.model.joint_name2id('machine_y'),
            'z': self.model.joint_name2id('machine_z'),
            'plunger': self.model.joint_name2id('plunger_joint')
        }
        
        self.actuator_indices = {
            'x': self.model.actuator_name2id('x_control'),
            'y': self.model.actuator_name2id('y_control'),
            'z': self.model.actuator_name2id('z_control'),
            'plunger': self.model.actuator_name2id('plunger_control')
        }
        
        # Get particle body indices
        self.particle_body_indices = {}
        for i in range(1, 7):  # well 1 particles
            try:
                body_id = self.model.body_name2id(f'particle_1_{i}')
                self.particle_body_indices[f'particle_1_{i}'] = body_id
            except:
                pass
        
        for i in range(1, 5):  # well 2 particles
            try:
                body_id = self.model.body_name2id(f'particle_2_{i}')
                self.particle_body_indices[f'particle_2_{i}'] = body_id
            except:
                pass
        
        for i in range(1, 6):  # loose particles
            try:
                body_id = self.model.body_name2id(f'loose_particle_{i}')
                self.particle_body_indices[f'loose_particle_{i}'] = body_id
            except:
                pass
    def _initialize_particles(self):
        """Initialize physics simulation particles based on updated XML positions"""
        particle_id = 0
        
        # UPDATED: Well 1 particles (moved to -0.12 center)
        well_1_center = np.array([-0.12, 0.0, 0.017])  # Updated from -0.3, -0.3
        for i, offset in enumerate([
            np.array([-0.02, -0.015, 0]), np.array([0.02, -0.015, 0]),
            np.array([-0.02, 0.015, 0]), np.array([0.02, 0.015, 0]),
            np.array([0, 0, 0]), np.array([0, 0, 0.008])
        ]):
            pos = well_1_center + offset
            self.physics_sim.add_particle(pos, particle_id, well_id=1)
            particle_id += 1
        
        # UPDATED: Well 2 particles (at center 0, 0)
        well_2_center = np.array([0.0, 0.0, 0.017])  # Updated from 0, -0.3
        for i, offset in enumerate([
            np.array([-0.015, -0.015, 0]), np.array([0.015, -0.015, 0]),
            np.array([-0.015, 0.015, 0]), np.array([0.015, 0.015, 0])
        ]):
            pos = well_2_center + offset
            self.physics_sim.add_particle(pos, particle_id, well_id=2)
            particle_id += 1
        
        # UPDATED: Container particles (moved to 0, 0.1)
        container_center = np.array([0.0, 0.1, 0.025])  # Updated from 0.4, 0.2
        for i, offset in enumerate([
            np.array([-0.015, -0.015, 0]), np.array([0.015, -0.015, 0]),
            np.array([-0.015, 0.015, 0]), np.array([0.015, 0.015, 0]),
            np.array([0, 0, 0.01])
        ]):
            pos = container_center + offset
            self.physics_sim.add_particle(pos, particle_id, well_id=None)
            particle_id += 1

    
    def _sync_particles_to_mujoco(self):
        """Synchronize physics simulation particle positions with MuJoCo"""
        for i, particle in enumerate(self.physics_sim.particles):
            # Map physics particle to MuJoCo body
            body_name = self._get_mujoco_body_name(i)
            if body_name in self.particle_body_indices:
                body_id = self.particle_body_indices[body_name]
                
                # Update MuJoCo body position
                if not particle.is_held:
                    # Only update free particles, held particles follow pipette
                    self.sim.data.qpos[body_id*7:body_id*7+3] = particle.position
                    self.sim.data.qvel[body_id*6:body_id*6+3] = particle.velocity
    
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
        # MuJoCo state
        joint_positions = np.array([
            self.sim.data.qpos[self.joint_indices['x']],
            self.sim.data.qpos[self.joint_indices['y']],
            self.sim.data.qpos[self.joint_indices['z']],
            self.sim.data.qpos[self.joint_indices['plunger']]
        ])
        
        joint_velocities = np.array([
            self.sim.data.qvel[self.joint_indices['x']],
            self.sim.data.qvel[self.joint_indices['y']],
            self.sim.data.qvel[self.joint_indices['z']],
            self.sim.data.qvel[self.joint_indices['plunger']]
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
        """Step function with proper action scaling for compact world"""
        self.episode_step += 1
        
        # UPDATED: Scale actions to match your XML actuator ranges
        scaled_action = np.array([
            action[0] * 0.5,      # X: -0.5 to 0.5 (XML ctrlrange)
            action[1] * 0.5,      # Y: -0.5 to 0.5 (XML ctrlrange)  
            action[2] * 0.5 + 0.5,  # Z: 0 to 1.0 (XML ctrlrange "0 1")
            action[3] * 0.25 + 0.25  # Plunger: 0 to 0.5 (XML ctrlrange "0 0.5")
        ])
        
        # Apply scaled action to MuJoCo
        self.data.ctrl[self.actuator_indices['x']] = scaled_action[0]
        self.data.ctrl[self.actuator_indices['y']] = scaled_action[1]
        self.data.ctrl[self.actuator_indices['z']] = scaled_action[2]
        self.data.ctrl[self.actuator_indices['plunger']] = scaled_action[3]
        
        # Step MuJoCo simulation
        mujoco.mj_step(self.model, self.data)
        
        # Get current MuJoCo state
        mujoco_state = {
            'x_pos': self.data.qpos[self.joint_indices['x']],
            'y_pos': self.data.qpos[self.joint_indices['y']],
            'z_pos': self.data.qpos[self.joint_indices['z']]
        }
        
        # Step physics simulation with properly scaled plunger
        physics_obs, physics_reward, physics_done, physics_info = self.env_wrapper.step(
            mujoco_state, 
            np.array([scaled_action[0], scaled_action[1], scaled_action[2], action[3]])  # Keep plunger in -1,1 for physics
        )

        # Synchronize particle positions
        self._sync_particles_to_mujoco()
        
        # Get combined observation
        observation = self._get_observation()
        
        # Calculate combined reward
        reward = self._calculate_reward(physics_reward, physics_info)
        
        # Check if episode is done
        done = (self.episode_step >= self.max_episode_steps or 
                physics_done or 
                self._check_task_completion())
        
        # Compile enhanced info
        info = {
            'physics_info': physics_info,
            'episode_step': self.episode_step,
            'particles_held': len(self.physics_sim.held_particles),
            'pipette_state': self.physics_sim.pipette_state.value,
            'task_phase': self.physics_sim.task_phase.value,
            'current_well': self.physics_sim._get_current_well(),
            'task_completed': self._check_task_completion(),
            'reward_breakdown': self.last_reward_breakdown,
            'recent_events': {
                'aspirations': len([e for e in self.physics_sim.aspiration_events if self.physics_sim.time - e.timestamp < 1.0]),
                'dispensing': len([e for e in self.physics_sim.dispensing_events if self.physics_sim.time - e.timestamp < 1.0]),
                'ball_losses': len([e for e in self.physics_sim.ball_loss_events if self.physics_sim.time - e.timestamp < 1.0]),
                'phase_violations': len([e for e in self.physics_sim.phase_violation_events if self.physics_sim.time - e.timestamp < 1.0])
            }
        }
        
        return observation, reward, done, info
    
    def _calculate_reward(self, physics_reward: float, physics_info: Dict) -> float:
        """Calculate reward using only the four specified components"""
        # Get the specific reward components from physics simulation
        reward_components = self.physics_sim.calculate_reward_components()
        
        # Sum only the four specified components
        total_reward = (
            reward_components['aspiration_component'] +
            reward_components['dispensing_component'] +
            reward_components['ball_loss_penalty'] +
            reward_components['phase_violation_penalty']
        )
        
        # Store detailed breakdown for analysis
        self.last_reward_breakdown = reward_components
        
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
    
    def reset(self) -> np.ndarray:
        """Reset the environment"""
        self.episode_step = 0
        
        # Reset MuJoCo simulation
        self.sim.reset()
        
        # Reset physics simulation
        self.physics_sim.reset()
        
        # Re-initialize particles
        self._initialize_particles()
        
        return self._get_observation()
    
    def render(self, mode='human'):
        """Render the environment"""
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
        
        self.viewer.render()
        
        if mode == 'rgb_array':
            # Return RGB array for recording
            width, height = 640, 480
            return self.sim.render(width, height, camera_name='top_view')
    
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