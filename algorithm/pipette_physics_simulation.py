import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass
from enum import Enum


class PipetteState(Enum):
    IDLE = "idle"
    ASPIRATING = "aspirating"
    HOLDING = "holding"
    DISPENSING = "dispensing"


@dataclass
class PipetteConfig:
    """Simple configuration for pipette physics"""
    # Reward parameters
    aspiration_reward: float = 10.0
    dispensing_reward: float = 8.0
    holding_reward: float = 1.0
    task_completion_bonus: float = 50.0
    
    # Physics parameters
    max_capacity: int = 3
    suction_range: float = 0.05


class SimplePipettePhysics:
    """
    Simplified physics simulator that just tracks basic pipette state
    and provides rewards based on sensor data
    """
    
    def __init__(self, config: PipetteConfig = None):
        self.config = config or PipetteConfig()
        
        # Simple state tracking
        self.held_particles = 0
        self.particles_transferred = 0
        self.task_completed = False
        self.time_step = 0
        
        # Pipette state
        self.pipette_state = PipetteState.IDLE
        self.pipette_tip_pos = np.array([0.0, 0.0, 0.0])
        self.plunger_position = 0.0
        
        # Simple particle tracking
        self.particles_near_tip = 0
        self.particles_in_wells = 0
    
    def update_from_mujoco(self, mujoco_data, sensor_data: Dict[str, np.ndarray]):
        """Update state from MuJoCo data"""
        self.time_step += 1
        
        # Update pipette tip position
        if 'site_suction_zone' in sensor_data:
            self.pipette_tip_pos = sensor_data['site_suction_zone']
        
        # Update plunger position
        if 'joint_pos_plunger' in sensor_data:
            self.plunger_position = sensor_data['joint_pos_plunger'][0]
        
        # Simple particle detection based on tip position
        self._update_particle_detection()
        
        # Update pipette state
        self._update_pipette_state()
        
        # Return observation dict
        obs_dict = {
            'pipette_tip_position': self.pipette_tip_pos,
            'plunger_position': self.plunger_position,
            'held_particle_count': self.held_particles,
            'particles_in_range': self.particles_near_tip,
            'task_completed': self.task_completed
        }
        
        return obs_dict, {}
    
    def _update_particle_detection(self):
        """Simple particle detection based on position - updated for new XML layout"""
        # Count particles near tip (simplified)
        tip_x, tip_y, tip_z = self.pipette_tip_pos
        
        # Check if tip is near particle container (updated position: 0, 0.1, 0)
        container_pos = np.array([0.0, 0.1, 0.0])
        dist_to_container = np.linalg.norm(self.pipette_tip_pos[:2] - container_pos[:2])  # Only X,Y distance
        
        if dist_to_container < 0.06:  # Smaller range for new compact layout
            self.particles_near_tip = 3  # Assume 3 particles available
        else:
            self.particles_near_tip = 0
        
        # Check if tip is near wells (updated positions for new layout)
        well_positions = [
            np.array([-0.12, 0.0, 0.0]),  # Well 1
            np.array([0.0, 0.0, 0.0]),    # Well 2  
            np.array([0.12, 0.0, 0.0])    # Well 3
        ]
        
        near_well = any(np.linalg.norm(self.pipette_tip_pos[:2] - well_pos[:2]) < 0.08 
                       for well_pos in well_positions)
        
        # Only count if we're actually dispensing near a well
        if near_well and self.pipette_state == PipetteState.DISPENSING:
            pass  # Don't increment here, let dispensing logic handle it
    
    def _update_pipette_state(self):
        """Update pipette operational state"""
        # Simple state machine based on plunger position and location
        if self.plunger_position > 0.05:  # Plunger extended
            if self.held_particles > 0:
                self.pipette_state = PipetteState.DISPENSING
            else:
                self.pipette_state = PipetteState.ASPIRATING
        elif self.held_particles > 0:
            self.pipette_state = PipetteState.HOLDING
        else:
            self.pipette_state = PipetteState.IDLE
        
        # Simple aspiration logic
        if (self.pipette_state == PipetteState.ASPIRATING and 
            self.particles_near_tip > 0 and 
            self.held_particles < self.config.max_capacity):
            # "Aspirate" a particle
            self.held_particles += 1
            self.particles_near_tip -= 1
        
        # Simple dispensing logic - only near wells
        if (self.pipette_state == PipetteState.DISPENSING and 
            self.held_particles > 0):
            # Check if we're actually near a well
            well_positions = [
                np.array([-0.12, 0.0, 0.0]),  # Well 1
                np.array([0.0, 0.0, 0.0]),    # Well 2  
                np.array([0.12, 0.0, 0.0])    # Well 3
            ]
            
            near_well = any(np.linalg.norm(self.pipette_tip_pos[:2] - well_pos[:2]) < 0.08 
                           for well_pos in well_positions)
            
            if near_well:
                # "Dispense" a particle
                self.held_particles -= 1
                self.particles_transferred += 1
        
        # Check task completion
        if self.particles_transferred >= 2:
            self.task_completed = True
    
    def calculate_reward(self, action: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Calculate reward based on current state"""
        reward_components = {
            'aspiration_reward': 0.0,
            'dispensing_reward': 0.0,
            'holding_reward': 0.0,
            'task_completion_bonus': 0.0,
            'movement_penalty': 0.0
        }
        
        # Reward for picking up particles
        if self.pipette_state == PipetteState.ASPIRATING and self.particles_near_tip > 0:
            reward_components['aspiration_reward'] = self.config.aspiration_reward
        
        # Reward for dispensing particles
        if self.pipette_state == PipetteState.DISPENSING and self.held_particles > 0:
            reward_components['dispensing_reward'] = self.config.dispensing_reward
        
        # Small reward for holding particles
        if self.pipette_state == PipetteState.HOLDING:
            reward_components['holding_reward'] = self.config.holding_reward
        
        # Big bonus for task completion
        if self.task_completed:
            reward_components['task_completion_bonus'] = self.config.task_completion_bonus
        
        # Small penalty for large movements (encourage efficiency)
        action_magnitude = np.linalg.norm(action)
        if action_magnitude > 0.5:
            reward_components['movement_penalty'] = -0.1 * action_magnitude
        
        total_reward = sum(reward_components.values())
        
        return total_reward, reward_components
    
    def reset(self):
        """Reset the physics simulation"""
        self.held_particles = 0
        self.particles_transferred = 0
        self.task_completed = False
        self.time_step = 0
        self.pipette_state = PipetteState.IDLE
        self.particles_near_tip = 0
        self.particles_in_wells = 0


class PipettePhysicsWrapper:
    """Simple wrapper to maintain compatibility"""
    
    def __init__(self, config: PipetteConfig = None):
        self.simulator = SimplePipettePhysics(config)
    
    def update_from_mujoco(self, mujoco_data, sensor_data):
        """Update from MuJoCo and return observation"""
        return self.simulator.update_from_mujoco(mujoco_data, sensor_data)
    
    def calculate_reward(self, action):
        """Calculate reward"""
        return self.simulator.calculate_reward(action)
    
    def is_done(self):
        """Check if episode should terminate"""
        return (self.simulator.task_completed or 
                self.simulator.time_step > 1000)
    
    def reset(self):
        """Reset simulation"""
        self.simulator.reset()