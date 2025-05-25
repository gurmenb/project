import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class PipetteState(Enum):
    IDLE = "idle"
    ASPIRATING = "aspirating"
    HOLDING = "holding"
    DISPENSING = "dispensing"

@dataclass
class Particle:
    """Represents a single particle/ball in the simulation"""
    id: int
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    radius: float = 0.008
    mass: float = 0.01
    is_held: bool = False
    well_id: Optional[int] = None  # Which well it belongs to (if any)

@dataclass
class PipetteConfig:
    """Configuration parameters for the pipette system"""
    # Physical parameters
    tip_radius: float = 0.008  # Inner radius of pipette tip
    max_capacity: int = 3      # Maximum number of particles that can be held
    suction_range: float = 0.025  # Maximum distance for suction effect
    
    # Plunger parameters
    plunger_travel: float = 0.08  # Maximum plunger travel distance
    min_suction_depth: float = 0.02  # Minimum plunger depth for suction
    max_suction_depth: float = 0.08  # Maximum effective suction depth
    
    # Physics parameters
    suction_force_max: float = 5.0    # Maximum suction force (N)
    dispense_force: float = 3.0       # Force for dispensing particles
    particle_cohesion: float = 0.1    # Attraction between particles
    air_resistance: float = 0.5       # Air resistance coefficient
    
    # Timing parameters
    aspiration_time: float = 0.5      # Time needed for full aspiration
    dispense_time: float = 0.3        # Time needed for dispensing

class PipettePhysicsSimulator:
    """
    Simulates the physics of particle aspiration and dispensing in a pipette system.
    
    Key Features:
    - Realistic suction force based on plunger depth and tip proximity
    - Limited capacity with priority-based particle selection
    - Fluid dynamics simulation for particle movement
    - State-based pipette operation (idle, aspirating, holding, dispensing)
    - Multi-particle interaction and cohesion effects
    """
    
    def __init__(self, config: PipetteConfig = None):
        self.config = config or PipetteConfig()
        self.particles: List[Particle] = []
        self.held_particles: List[Particle] = []
        self.pipette_state = PipetteState.IDLE
        
        # Pipette position and state
        self.pipette_tip_pos = np.array([0.0, 0.0, 0.0])
        self.plunger_position = 0.0  # 0 = fully retracted, max = fully extended
        self.plunger_velocity = 0.0
        
        # Internal simulation state
        self.suction_pressure = 0.0
        self.time_in_state = 0.0
        self.dt = 0.01  # Simulation timestep
        
        # Statistics and logging
        self.aspiration_history = []
        self.dispense_history = []
        
    def add_particle(self, position: np.ndarray, particle_id: int = None, well_id: int = None) -> Particle:
        """Add a new particle to the simulation"""
        if particle_id is None:
            particle_id = len(self.particles)
        
        particle = Particle(
            id=particle_id,
            position=position.copy(),
            velocity=np.zeros(3),
            well_id=well_id
        )
        self.particles.append(particle)
        return particle
    
    def update_pipette_state(self, tip_position: np.ndarray, plunger_depth: float, dt: float):
        """
        Update pipette state based on position and plunger depth.
        
        Args:
            tip_position: 3D position of pipette tip
            plunger_depth: Depth of plunger (0-1, where 1 is fully extended)
            dt: Time step for simulation
        """
        self.pipette_tip_pos = tip_position.copy()
        self.plunger_position = np.clip(plunger_depth, 0.0, 1.0)
        self.dt = dt
        self.time_in_state += dt
        
        # Calculate plunger velocity for state detection
        prev_plunger = getattr(self, '_prev_plunger_pos', self.plunger_position)
        self.plunger_velocity = (self.plunger_position - prev_plunger) / dt
        self._prev_plunger_pos = self.plunger_position
        
        # Update pipette state based on plunger movement and position
        self._update_pipette_state()
        
        # Apply physics simulation
        self._simulate_particle_physics()
        
        # Handle aspiration and dispensing
        self._handle_particle_interactions()
    
    def _update_pipette_state(self):
        """Update the pipette operational state"""
        plunger_depth = self.plunger_position * self.config.plunger_travel
        
        if self.plunger_velocity < -0.1 and plunger_depth > self.config.min_suction_depth:
            # Plunger moving up (creating suction)
            if len(self.held_particles) < self.config.max_capacity:
                self.pipette_state = PipetteState.ASPIRATING
            else:
                self.pipette_state = PipetteState.HOLDING
        elif self.plunger_velocity > 0.1 and len(self.held_particles) > 0:
            # Plunger moving down (dispensing)
            self.pipette_state = PipetteState.DISPENSING
        elif len(self.held_particles) > 0:
            # Holding particles
            self.pipette_state = PipetteState.HOLDING
        else:
            # No active operation
            self.pipette_state = PipetteState.IDLE
    
    def _calculate_suction_force(self, particle_pos: np.ndarray) -> float:
        """
        Calculate suction force on a particle based on pipette state and position.
        
        Returns:
            Suction force magnitude (positive = toward pipette)
        """
        if self.pipette_state != PipetteState.ASPIRATING:
            return 0.0
        
        # Distance from particle to pipette tip
        distance = np.linalg.norm(particle_pos - self.pipette_tip_pos)
        
        if distance > self.config.suction_range:
            return 0.0
        
        # Suction force based on plunger depth and distance
        plunger_depth = self.plunger_position * self.config.plunger_travel
        depth_factor = np.clip((plunger_depth - self.config.min_suction_depth) / 
                              (self.config.max_suction_depth - self.config.min_suction_depth), 0.0, 1.0)
        
        # Distance factor (inverse square law with minimum)
        distance_factor = max(0.1, 1.0 / (1.0 + distance / self.config.tip_radius))
        
        # Velocity factor (more suction for faster plunger movement)
        velocity_factor = max(1.0, abs(self.plunger_velocity) * 5)
        
        return self.config.suction_force_max * depth_factor * distance_factor * velocity_factor
    
    def _get_particles_in_range(self) -> List[Tuple[Particle, float]]:
        """Get particles within suction range, sorted by distance"""
        candidates = []
        
        for particle in self.particles:
            if particle.is_held:
                continue
                
            distance = np.linalg.norm(particle.position - self.pipette_tip_pos)
            if distance <= self.config.suction_range:
                candidates.append((particle, distance))
        
        # Sort by distance (closest first)
        candidates.sort(key=lambda x: x[1])
        return candidates
    
    def _select_particles_for_aspiration(self, candidates: List[Tuple[Particle, float]]) -> List[Particle]:
        """
        Select which particles to aspirate based on priority and capacity.
        
        Priority factors:
        1. Distance to pipette tip (closer = higher priority)
        2. Particle size (smaller particles easier to aspirate)
        3. Current capacity limit
        4. Existing particle cohesion
        """
        available_capacity = self.config.max_capacity - len(self.held_particles)
        if available_capacity <= 0:
            return []
        
        selected = []
        
        # Simple selection: take closest particles up to capacity
        for particle, distance in candidates[:available_capacity]:
            # Additional checks for aspiration feasibility
            suction_force = self._calculate_suction_force(particle.position)
            
            # Minimum force threshold for aspiration
            if suction_force > 1.0:  # Arbitrary threshold
                selected.append(particle)
        
        return selected
    
    def _simulate_particle_physics(self):
        """Simulate physics for all particles"""
        for particle in self.particles:
            if particle.is_held:
                # Held particles move with pipette tip
                target_pos = self.pipette_tip_pos + np.array([0, 0, 0.01])  # Slightly inside tip
                particle.position = 0.9 * particle.position + 0.1 * target_pos
                particle.velocity *= 0.8  # Damping
            else:
                # Free particles subject to forces
                forces = np.zeros(3)
                
                # Suction force
                suction_force = self._calculate_suction_force(particle.position)
                if suction_force > 0:
                    direction = self.pipette_tip_pos - particle.position
                    direction_norm = np.linalg.norm(direction)
                    if direction_norm > 0:
                        forces += (direction / direction_norm) * suction_force
                
                # Dispense force (if pipette is dispensing and particle was held)
                if (self.pipette_state == PipetteState.DISPENSING and 
                    np.linalg.norm(particle.position - self.pipette_tip_pos) < 0.02):
                    direction = particle.position - self.pipette_tip_pos
                    direction_norm = np.linalg.norm(direction)
                    if direction_norm > 0:
                        forces += (direction / direction_norm) * self.config.dispense_force
                
                # Gravity
                forces += np.array([0, 0, -9.81]) * particle.mass
                
                # Air resistance
                forces -= particle.velocity * self.config.air_resistance
                
                # Particle cohesion (attraction to nearby particles)
                for other in self.particles:
                    if other.id != particle.id and not other.is_held:
                        diff = other.position - particle.position
                        distance = np.linalg.norm(diff)
                        if 0 < distance < 0.05:  # Cohesion range
                            forces += (diff / distance) * self.config.particle_cohesion / distance
                
                # Update velocity and position
                acceleration = forces / particle.mass
                particle.velocity += acceleration * self.dt
                particle.position += particle.velocity * self.dt
                
                # Simple collision with ground
                if particle.position[2] < particle.radius:
                    particle.position[2] = particle.radius
                    particle.velocity[2] = max(0, particle.velocity[2])
    
    def _handle_particle_interactions(self):
        """Handle aspiration and dispensing of particles"""
        if self.pipette_state == PipetteState.ASPIRATING:
            self._process_aspiration()
        elif self.pipette_state == PipetteState.DISPENSING:
            self._process_dispensing()
    
    def _process_aspiration(self):
        """Process particle aspiration"""
        candidates = self._get_particles_in_range()
        selected_particles = self._select_particles_for_aspiration(candidates)
        
        for particle in selected_particles:
            # Check if particle is close enough and conditions are met
            distance = np.linalg.norm(particle.position - self.pipette_tip_pos)
            
            if distance < self.config.tip_radius * 2:  # Very close to tip
                particle.is_held = True
                self.held_particles.append(particle)
                
                # Log aspiration event
                self.aspiration_history.append({
                    'time': self.time_in_state,
                    'particle_id': particle.id,
                    'position': particle.position.copy(),
                    'plunger_depth': self.plunger_position
                })
    
    def _process_dispensing(self):
        """Process particle dispensing"""
        particles_to_release = []
        
        for particle in self.held_particles[:]:  # Copy list to allow modification
            # Release particle with some probability based on plunger movement
            release_probability = abs(self.plunger_velocity) * self.dt * 10
            
            if np.random.random() < release_probability:
                particle.is_held = False
                particles_to_release.append(particle)
                
                # Give particle some initial velocity away from pipette
                direction = np.random.normal(0, 1, 3)
                direction[2] = abs(direction[2])  # Downward bias
                direction = direction / np.linalg.norm(direction)
                particle.velocity = direction * 0.5  # Initial dispense velocity
                
                # Log dispense event
                self.dispense_history.append({
                    'time': self.time_in_state,
                    'particle_id': particle.id,
                    'position': particle.position.copy(),
                    'plunger_depth': self.plunger_position
                })
        
        # Remove released particles from held list
        for particle in particles_to_release:
            if particle in self.held_particles:
                self.held_particles.remove(particle)
    
    def get_state_dict(self) -> Dict:
        """Get current simulation state for RL algorithm"""
        return {
            'pipette_tip_position': self.pipette_tip_pos.copy(),
            'plunger_position': self.plunger_position,
            'plunger_velocity': self.plunger_velocity,
            'pipette_state': self.pipette_state.value,
            'held_particle_count': len(self.held_particles),
            'held_particle_ids': [p.id for p in self.held_particles],
            'nearby_particle_count': len(self._get_particles_in_range()),
            'suction_pressure': self._calculate_suction_force(self.pipette_tip_pos),
            'particles_in_range': [
                {
                    'id': p.id,
                    'position': p.position.copy(),
                    'distance': dist,
                    'suction_force': self._calculate_suction_force(p.position)
                }
                for p, dist in self._get_particles_in_range()
            ]
        }
    
    def calculate_reward_components(self) -> Dict[str, float]:
        """Calculate reward components for RL training"""
        rewards = {
            'aspiration_reward': 0.0,
            'dispensing_reward': 0.0,
            'efficiency_penalty': 0.0,
            'capacity_bonus': 0.0,
            'state_bonus': 0.0
        }
        
        # Reward recent aspirations
        recent_aspirations = [a for a in self.aspiration_history if self.time_in_state - a['time'] < 0.1]
        rewards['aspiration_reward'] = len(recent_aspirations) * 2.0
        
        # Reward recent dispensing
        recent_dispensing = [d for d in self.dispense_history if self.time_in_state - d['time'] < 0.1]
        rewards['dispensing_reward'] = len(recent_dispensing) * 1.5
        
        # Penalty for inefficient operation
        if self.pipette_state == PipetteState.ASPIRATING and len(self._get_particles_in_range()) == 0:
            rewards['efficiency_penalty'] = -0.1
        
        # Bonus for optimal capacity usage
        capacity_ratio = len(self.held_particles) / self.config.max_capacity
        if 0.5 <= capacity_ratio <= 1.0:
            rewards['capacity_bonus'] = capacity_ratio * 0.5
        
        # State-based rewards
        if self.pipette_state == PipetteState.HOLDING and len(self.held_particles) > 0:
            rewards['state_bonus'] = 0.1
        
        return rewards
    
    def reset(self):
        """Reset simulation state"""
        for particle in self.particles:
            particle.is_held = False
            particle.velocity = np.zeros(3)
        
        self.held_particles.clear()
        self.pipette_state = PipetteState.IDLE
        self.time_in_state = 0.0
        self.aspiration_history.clear()
        self.dispense_history.clear()
    
    def visualize_state(self, ax=None):
        """Create a visualization of current simulation state"""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Plot particles
        for particle in self.particles:
            color = 'red' if particle.is_held else 'blue'
            ax.scatter(particle.position[0], particle.position[2], 
                      c=color, s=100, alpha=0.7, label=f'Particle {particle.id}')
        
        # Plot pipette tip
        ax.scatter(self.pipette_tip_pos[0], self.pipette_tip_pos[2], 
                  c='green', s=200, marker='^', label='Pipette Tip')
        
        # Plot suction range
        circle = plt.Circle((self.pipette_tip_pos[0], self.pipette_tip_pos[2]), 
                           self.config.suction_range, fill=False, color='green', alpha=0.3)
        ax.add_patch(circle)
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Z Position')
        ax.set_title(f'Pipette State: {self.pipette_state.value} | '
                    f'Held: {len(self.held_particles)} | '
                    f'Plunger: {self.plunger_position:.2f}')
        ax.grid(True)
        ax.axis('equal')
        
        return ax

# Example usage and integration with RL
class PipetteEnvironmentWrapper:
    """Wrapper to integrate the physics simulator with RL algorithms"""
    
    def __init__(self, physics_sim: PipettePhysicsSimulator):
        self.physics_sim = physics_sim
        self.last_reward_time = 0.0
    
    def step(self, mujoco_state: Dict, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """
        Step function compatible with RL algorithms
        
        Args:
            mujoco_state: State from MuJoCo simulation
            action: Action from RL agent [x, y, z, plunger]
        
        Returns:
            observation, reward, done, info
        """
        # Extract pipette position from MuJoCo state
        tip_position = np.array([action[0], action[1], mujoco_state.get('z_pos', 0)])
        plunger_depth = action[3]  # Use RL action for plunger instead of MuJoCo
        
        # Update physics simulation
        self.physics_sim.update_pipette_state(tip_position, plunger_depth, 0.01)
        
        # Get observation
        observation = self.physics_sim.get_state_dict()
        
        # Calculate reward
        reward_components = self.physics_sim.calculate_reward_components()
        total_reward = sum(reward_components.values())
        
        # Check if episode is done (example conditions)
        done = (len(self.physics_sim.held_particles) >= self.physics_sim.config.max_capacity and 
                self.physics_sim.pipette_state == PipetteState.HOLDING)
        
        # Additional info
        info = {
            'reward_components': reward_components,
            'physics_state': observation,
            'particles_aspirated': len(self.physics_sim.aspiration_history),
            'particles_dispensed': len(self.physics_sim.dispense_history)
        }
        
        return observation, total_reward, done, info