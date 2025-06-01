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

class TaskPhase(Enum):
    APPROACH_SOURCE = "approach_source"
    ASPIRATE = "aspirate"
    TRANSPORT = "transport"
    APPROACH_TARGET = "approach_target"
    DISPENSE = "dispense"
    COMPLETE = "complete"

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
    original_well: Optional[int] = None  # Original well for tracking transfers

@dataclass
class PipetteConfig:
    """Configuration parameters updated for compact world"""
    # Physical parameters
    tip_radius: float = 0.008
    max_capacity: int = 3
    suction_range: float = 0.03  # Slightly larger for bigger wells
    
    # Volume parameters (in terms of particle count)
    target_aspiration_volume: int = 2
    min_aspiration_volume: int = 1
    max_aspiration_volume: int = 3
    
    # Plunger parameters - UPDATED
    plunger_travel: float = 0.08  # Matches XML joint range
    min_suction_depth: float = 0.02
    max_suction_depth: float = 0.08
    
    # Physics parameters
    suction_force_max: float = 5.0
    dispense_force: float = 3.0
    particle_cohesion: float = 0.1
    air_resistance: float = 0.5
    
    # NEW: Well positions matching your XML
    well_radius: float = 0.055  # Bigger wells (0.05 + margin)

@dataclass
class AspirationEvent:
    """Records an aspiration event for reward calculation"""
    timestamp: float
    particles_aspirated: List[int]  # Particle IDs
    volume_aspirated: int  # Number of particles
    plunger_depth: float
    pipette_position: np.ndarray
    source_well: Optional[int]
    was_correct_timing: bool
    was_correct_volume: bool

@dataclass
class DispensingEvent:
    """Records a dispensing event for reward calculation"""
    timestamp: float
    particles_dispensed: List[int]  # Particle IDs
    volume_dispensed: int  # Number of particles
    plunger_depth: float
    pipette_position: np.ndarray
    target_well: Optional[int]
    was_correct_position: bool
    was_correct_volume: bool

@dataclass
class BallLossEvent:
    """Records when balls are lost unexpectedly"""
    timestamp: float
    particle_id: int
    loss_position: np.ndarray
    expected_state: str  # What state the ball should have been in

@dataclass
class PhaseViolationEvent:
    """Records task sequence violations"""
    timestamp: float
    expected_phase: TaskPhase
    actual_action: str
    violation_type: str

class PipettePhysicsSimulator:
    """
    Enhanced physics simulator with detailed reward component tracking
    """
    
    def __init__(self, config: PipetteConfig = None):
        self.config = config or PipetteConfig()
        self.particles: List[Particle] = []
        self.held_particles: List[Particle] = []
        self.pipette_state = PipetteState.IDLE
        self.task_phase = TaskPhase.APPROACH_SOURCE
        
        # Pipette position and state
        self.pipette_tip_pos = np.array([0.0, 0.0, 0.0])
        self.plunger_position = 0.0
        self.plunger_velocity = 0.0
        self.time = 0.0
        self.dt = 0.01
        
        # Event tracking for rewards
        self.aspiration_events: List[AspirationEvent] = []
        self.dispensing_events: List[DispensingEvent] = []
        self.ball_loss_events: List[BallLossEvent] = []
        self.phase_violation_events: List[PhaseViolationEvent] = []
        
        # State tracking
        self.last_held_particles = []
        self.expected_particles_held = 0
        self.source_well_id = None
        self.target_well_id = None
        
    
        # Well definitions - UPDATED to match XML
        self.wells = {
            1: np.array([-0.12, 0.0, 0.01]),  # Source well (moved from -0.08)
            2: np.array([0.0, 0.0, 0.01]),    # Middle well  
            3: np.array([0.12, 0.0, 0.01])    # Target well (moved from 0.08)
        }
    
        
        # Task configuration
        self.source_well_id = 1
        self.target_well_id = 3
        
    def add_particle(self, position: np.ndarray, particle_id: int = None, well_id: int = None) -> Particle:
        """Add a new particle to the simulation"""
        if particle_id is None:
            particle_id = len(self.particles)
        
        particle = Particle(
            id=particle_id,
            position=position.copy(),
            velocity=np.zeros(3),
            well_id=well_id,
            original_well=well_id
        )
        self.particles.append(particle)
        return particle
    
    def update_pipette_state(self, tip_position: np.ndarray, plunger_depth: float, dt: float):
        """Update pipette state and detect events"""
        self.pipette_tip_pos = tip_position.copy()
        self.plunger_position = np.clip(plunger_depth, 0.0, 1.0)
        self.dt = dt
        self.time += dt
        
        # Calculate plunger velocity
        prev_plunger = getattr(self, '_prev_plunger_pos', self.plunger_position)
        self.plunger_velocity = (self.plunger_position - prev_plunger) / dt
        self._prev_plunger_pos = self.plunger_position
        
        # Store previous state for loss detection
        self.last_held_particles = [p.id for p in self.held_particles]
        
        # Update states
        self._update_pipette_state()
        self._update_task_phase()
        self._simulate_particle_physics()
        self._handle_particle_interactions()
        self._detect_ball_losses()
        
    def _update_pipette_state(self):
        """Update the pipette operational state"""
        plunger_depth = self.plunger_position * self.config.plunger_travel
        
        if self.plunger_velocity < -0.1 and plunger_depth > self.config.min_suction_depth:
            if len(self.held_particles) < self.config.max_capacity:
                self.pipette_state = PipetteState.ASPIRATING
            else:
                self.pipette_state = PipetteState.HOLDING
        elif self.plunger_velocity > 0.1 and len(self.held_particles) > 0:
            self.pipette_state = PipetteState.DISPENSING
        elif len(self.held_particles) > 0:
            self.pipette_state = PipetteState.HOLDING
        else:
            self.pipette_state = PipetteState.IDLE
    
    def _update_task_phase(self):
        """Update task phase and detect violations"""
        current_well = self._get_current_well()
        
        old_phase = self.task_phase
        
        # Phase transition logic
        if self.task_phase == TaskPhase.APPROACH_SOURCE:
            if current_well == self.source_well_id and len(self.held_particles) == 0:
                if self.pipette_state == PipetteState.ASPIRATING:
                    self.task_phase = TaskPhase.ASPIRATE
        
        elif self.task_phase == TaskPhase.ASPIRATE:
            if len(self.held_particles) > 0:
                self.task_phase = TaskPhase.TRANSPORT
        
        elif self.task_phase == TaskPhase.TRANSPORT:
            if current_well == self.target_well_id:
                self.task_phase = TaskPhase.APPROACH_TARGET
        
        elif self.task_phase == TaskPhase.APPROACH_TARGET:
            if self.pipette_state == PipetteState.DISPENSING:
                self.task_phase = TaskPhase.DISPENSE
        
        elif self.task_phase == TaskPhase.DISPENSE:
            if len(self.held_particles) == 0:
                self.task_phase = TaskPhase.COMPLETE
        
        # Detect phase violations
        self._detect_phase_violations(old_phase)
    
    def _detect_phase_violations(self, old_phase: TaskPhase):
        """Detect and record phase violations"""
        current_well = self._get_current_well()
        
        # Violation: Aspirating at wrong time/place
        if (self.pipette_state == PipetteState.ASPIRATING and 
            self.task_phase != TaskPhase.ASPIRATE and 
            old_phase != TaskPhase.APPROACH_SOURCE):
            
            self.phase_violation_events.append(PhaseViolationEvent(
                timestamp=self.time,
                expected_phase=self.task_phase,
                actual_action="aspirating",
                violation_type="wrong_phase_aspiration"
            ))
        
        # Violation: Dispensing at wrong location
        if (self.pipette_state == PipetteState.DISPENSING and 
            current_well != self.target_well_id and 
            len(self.held_particles) > 0):
            
            self.phase_violation_events.append(PhaseViolationEvent(
                timestamp=self.time,
                expected_phase=TaskPhase.APPROACH_TARGET,
                actual_action="dispensing_wrong_location",
                violation_type="wrong_location_dispensing"
            ))
        
        # Violation: Skipping transport phase
        if (old_phase == TaskPhase.ASPIRATE and 
            self.task_phase == TaskPhase.APPROACH_TARGET and 
            current_well == self.source_well_id):
            
            self.phase_violation_events.append(PhaseViolationEvent(
                timestamp=self.time,
                expected_phase=TaskPhase.TRANSPORT,
                actual_action="skipped_transport",
                violation_type="phase_skip"
            ))
    
    def _get_current_well(self) -> Optional[int]:
        """Determine which well the pipette is currently over"""
        for well_id, well_pos in self.wells.items():
            distance = np.linalg.norm(self.pipette_tip_pos[:2] - well_pos[:2])
            if distance < self.config.well_radius:
                return well_id
        return None
    
    def _simulate_particle_physics(self):
        """Simulate physics for all particles"""
        for particle in self.particles:
            if particle.is_held:
                # Held particles move with pipette tip
                target_pos = self.pipette_tip_pos + np.array([0, 0, 0.01])
                particle.position = 0.9 * particle.position + 0.1 * target_pos
                particle.velocity *= 0.8
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
                
                # Dispense force
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
                
                # Update particle
                acceleration = forces / particle.mass
                particle.velocity += acceleration * self.dt
                particle.position += particle.velocity * self.dt
                
                # Ground collision
                if particle.position[2] < particle.radius:
                    particle.position[2] = particle.radius
                    particle.velocity[2] = max(0, particle.velocity[2])
    
    def _calculate_suction_force(self, particle_pos: np.ndarray) -> float:
        """Calculate suction force on a particle"""
        if self.pipette_state != PipetteState.ASPIRATING:
            return 0.0
        
        distance = np.linalg.norm(particle_pos - self.pipette_tip_pos)
        if distance > self.config.suction_range:
            return 0.0
        
        plunger_depth = self.plunger_position * self.config.plunger_travel
        depth_factor = np.clip((plunger_depth - self.config.min_suction_depth) / 
                              (self.config.max_suction_depth - self.config.min_suction_depth), 0.0, 1.0)
        
        distance_factor = max(0.1, 1.0 / (1.0 + distance / self.config.tip_radius))
        velocity_factor = max(1.0, abs(self.plunger_velocity) * 5)
        
        return self.config.suction_force_max * depth_factor * distance_factor * velocity_factor
    
    def _handle_particle_interactions(self):
        """Handle aspiration and dispensing with event recording"""
        if self.pipette_state == PipetteState.ASPIRATING:
            self._process_aspiration()
        elif self.pipette_state == PipetteState.DISPENSING:
            self._process_dispensing()
    
    def _process_aspiration(self):
        """Process particle aspiration with detailed event recording"""
        candidates = self._get_particles_in_range()
        selected_particles = self._select_particles_for_aspiration(candidates)
        
        aspirated_particles = []
        current_well = self._get_current_well()
        
        for particle in selected_particles:
            distance = np.linalg.norm(particle.position - self.pipette_tip_pos)
            
            if distance < self.config.tip_radius * 2:
                particle.is_held = True
                self.held_particles.append(particle)
                aspirated_particles.append(particle.id)
        
        # Record aspiration event if particles were aspirated
        if aspirated_particles:
            volume_aspirated = len(aspirated_particles)
            
            # Determine if aspiration was correct
            was_correct_timing = (self.task_phase == TaskPhase.ASPIRATE and 
                                current_well == self.source_well_id)
            was_correct_volume = (self.config.min_aspiration_volume <= 
                                volume_aspirated <= self.config.max_aspiration_volume)
            
            event = AspirationEvent(
                timestamp=self.time,
                particles_aspirated=aspirated_particles,
                volume_aspirated=volume_aspirated,
                plunger_depth=self.plunger_position,
                pipette_position=self.pipette_tip_pos.copy(),
                source_well=current_well,
                was_correct_timing=was_correct_timing,
                was_correct_volume=was_correct_volume
            )
            
            self.aspiration_events.append(event)
    
    def _process_dispensing(self):
        """Process particle dispensing with detailed event recording"""
        particles_to_release = []
        current_well = self._get_current_well()
        
        for particle in self.held_particles[:]:
            release_probability = abs(self.plunger_velocity) * self.dt * 10
            
            if np.random.random() < release_probability:
                particle.is_held = False
                particles_to_release.append(particle)
                
                # Give particle initial velocity
                direction = np.random.normal(0, 1, 3)
                direction[2] = abs(direction[2])
                direction = direction / np.linalg.norm(direction)
                particle.velocity = direction * 0.5
        
        # Record dispensing event if particles were dispensed
        if particles_to_release:
            dispensed_particle_ids = [p.id for p in particles_to_release]
            volume_dispensed = len(particles_to_release)
            
            # Determine if dispensing was correct
            was_correct_position = (current_well == self.target_well_id)
            was_correct_volume = (volume_dispensed <= self.config.target_aspiration_volume)
            
            event = DispensingEvent(
                timestamp=self.time,
                particles_dispensed=dispensed_particle_ids,
                volume_dispensed=volume_dispensed,
                plunger_depth=self.plunger_position,
                pipette_position=self.pipette_tip_pos.copy(),
                target_well=current_well,
                was_correct_position=was_correct_position,
                was_correct_volume=was_correct_volume
            )
            
            self.dispensing_events.append(event)
            
            # Remove from held particles
            for particle in particles_to_release:
                if particle in self.held_particles:
                    self.held_particles.remove(particle)
    
    def _detect_ball_losses(self):
        """Detect unexpected ball losses"""
        current_held_ids = [p.id for p in self.held_particles]
        
        for particle_id in self.last_held_particles:
            if particle_id not in current_held_ids:
                # Find the lost particle
                lost_particle = next((p for p in self.particles if p.id == particle_id), None)
                if lost_particle and not self.pipette_state == PipetteState.DISPENSING:
                    # This is an unexpected loss
                    self.ball_loss_events.append(BallLossEvent(
                        timestamp=self.time,
                        particle_id=particle_id,
                        loss_position=lost_particle.position.copy(),
                        expected_state="held"
                    ))
    
    def _get_particles_in_range(self) -> List[Tuple[Particle, float]]:
        """Get particles within suction range, sorted by distance"""
        candidates = []
        
        for particle in self.particles:
            if particle.is_held:
                continue
                
            distance = np.linalg.norm(particle.position - self.pipette_tip_pos)
            if distance <= self.config.suction_range:
                candidates.append((particle, distance))
        
        candidates.sort(key=lambda x: x[1])
        return candidates
    
    def _select_particles_for_aspiration(self, candidates: List[Tuple[Particle, float]]) -> List[Particle]:
        """Select particles for aspiration"""
        available_capacity = self.config.max_capacity - len(self.held_particles)
        if available_capacity <= 0:
            return []
        
        selected = []
        for particle, distance in candidates[:available_capacity]:
            suction_force = self._calculate_suction_force(particle.position)
            if suction_force > 1.0:
                selected.append(particle)
        
        return selected
    
    def calculate_reward_components(self) -> Dict[str, float]:
        """Calculate the four specific reward components"""
        rewards = {
            'aspiration_component': 0.0,
            'dispensing_component': 0.0,
            'ball_loss_penalty': 0.0,
            'phase_violation_penalty': 0.0
        }
        
        # 1. Aspiration Component (recent events only)
        recent_aspirations = [e for e in self.aspiration_events if self.time - e.timestamp < 0.1]
        for event in recent_aspirations:
            if event.was_correct_timing and event.was_correct_volume:
                # Good aspiration reward
                rewards['aspiration_component'] += 2.0 * event.volume_aspirated
            else:
                # Bad aspiration penalty
                penalty = 0.0
                if not event.was_correct_timing:
                    penalty -= 1.0  # Wrong timing
                if not event.was_correct_volume:
                    penalty -= 1.0  # Wrong volume
                if event.volume_aspirated > self.config.max_aspiration_volume:
                    penalty -= 2.0  # Severe over-aspiration
                rewards['aspiration_component'] += penalty
        
        # 2. Dispensing Component (recent events only)
        recent_dispensing = [e for e in self.dispensing_events if self.time - e.timestamp < 0.1]
        for event in recent_dispensing:
            if event.was_correct_position and event.was_correct_volume:
                # Good dispensing reward
                rewards['dispensing_component'] += 3.0 * event.volume_dispensed
            else:
                # Bad dispensing penalty
                penalty = 0.0
                if not event.was_correct_position:
                    penalty -= 2.0  # Wrong position is serious
                if not event.was_correct_volume:
                    penalty -= 1.0  # Wrong volume
                rewards['dispensing_component'] += penalty
        
        # 3. Ball Loss Penalty (recent events only)
        recent_losses = [e for e in self.ball_loss_events if self.time - e.timestamp < 0.1]
        rewards['ball_loss_penalty'] = -3.0 * len(recent_losses)  # Severe penalty
        
        # 4. Phase Violation Penalty (recent events only)
        recent_violations = [e for e in self.phase_violation_events if self.time - e.timestamp < 0.1]
        violation_penalties = {
            'wrong_phase_aspiration': -2.0,
            'wrong_location_dispensing': -3.0,
            'phase_skip': -1.5
        }
        
        for event in recent_violations:
            penalty = violation_penalties.get(event.violation_type, -1.0)
            rewards['phase_violation_penalty'] += penalty
        
        return rewards
    
    def get_detailed_state_dict(self) -> Dict:
        """Get detailed state information for debugging and analysis"""
        return {
            'pipette_tip_position': self.pipette_tip_pos.copy(),
            'plunger_position': self.plunger_position,
            'plunger_velocity': self.plunger_velocity,
            'pipette_state': self.pipette_state.value,
            'task_phase': self.task_phase.value,
            'held_particle_count': len(self.held_particles),
            'held_particle_ids': [p.id for p in self.held_particles],
            'current_well': self._get_current_well(),
            'source_well_id': self.source_well_id,
            'target_well_id': self.target_well_id,
            'recent_events': {
                'aspirations': len([e for e in self.aspiration_events if self.time - e.timestamp < 1.0]),
                'dispensing': len([e for e in self.dispensing_events if self.time - e.timestamp < 1.0]),
                'ball_losses': len([e for e in self.ball_loss_events if self.time - e.timestamp < 1.0]),
                'phase_violations': len([e for e in self.phase_violation_events if self.time - e.timestamp < 1.0])
            }
        }
    
    def reset(self):
        """Reset simulation state"""
        for particle in self.particles:
            particle.is_held = False
            particle.velocity = np.zeros(3)
        
        self.held_particles.clear()
        self.pipette_state = PipetteState.IDLE
        self.task_phase = TaskPhase.APPROACH_SOURCE
        self.time = 0.0
        
        # Clear event history
        self.aspiration_events.clear()
        self.dispensing_events.clear()
        self.ball_loss_events.clear()
        self.phase_violation_events.clear()
        
        self.last_held_particles = []