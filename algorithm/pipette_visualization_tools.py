#!/usr/bin/env python3
"""
Comprehensive visualization tools for pipette training monitoring.
Includes MuJoCo rendering, real-time plots, and training dashboards.
"""
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque
import threading
import queue
import tkinter
matplotlib.use('TkAgg')  # or 'Qt5Agg' depending on your system

# Try to import MuJoCo viewer
try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False

@dataclass
class VisualizationState:
    """State information for visualization"""
    timestep: int
    pipette_position: np.ndarray
    pipette_state: str
    task_phase: str
    particles_held: int
    particles_positions: List[np.ndarray]
    particles_held_ids: List[int]
    well_positions: Dict[int, np.ndarray]
    current_well: Optional[int]
    reward_components: Dict[str, float]
    total_reward: float
    action: np.ndarray

class MuJoCoVisualizer:
    """Enhanced MuJoCo visualizer with state overlay"""
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.viewer = None
        self.state_text = ""
        
    def setup_viewer(self):
        """Setup MuJoCo viewer with custom settings"""
        if not MUJOCO_AVAILABLE:
            print("MuJoCo not available for visualization")
            return
        
        try:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance = 0.8
            self.viewer.cam.azimuth = 45
            self.viewer.cam.elevation = -30
            print("✓ MuJoCo viewer initialized")
        except Exception as e:
            print(f"Failed to initialize MuJoCo viewer: {e}")
    
    def update_visualization(self, vis_state: VisualizationState):
        """Update MuJoCo visualization with state information"""
        if not self.viewer:
            return
        
        try:
            # Update state text overlay
            self.state_text = self._format_state_text(vis_state)
            
            # Sync viewer
            self.viewer.sync()
            
            # Add custom overlays (if MuJoCo supports it)
            self._draw_state_overlays(vis_state)
        except Exception as e:
            print(f"Visualization update error: {e}")
    
    def _format_state_text(self, state: VisualizationState) -> str:
        """Format state information as text"""
        return f"""
Timestep: {state.timestep}
Pipette State: {state.pipette_state}
Task Phase: {state.task_phase}
Position: ({state.pipette_position[0]:.3f}, {state.pipette_position[1]:.3f}, {state.pipette_position[2]:.3f})
Particles Held: {state.particles_held}
Current Well: {state.current_well}
Total Reward: {state.total_reward:.3f}
Action: [{', '.join(f'{a:.2f}' for a in state.action)}]

Reward Components:
  Aspiration: {state.reward_components.get('aspiration_component', 0):.2f}
  Dispensing: {state.reward_components.get('dispensing_component', 0):.2f}
  Ball Loss: {state.reward_components.get('ball_loss_penalty', 0):.2f}
  Phase Violation: {state.reward_components.get('phase_violation_penalty', 0):.2f}
        """.strip()
    
    def _draw_state_overlays(self, state: VisualizationState):
        """Draw custom overlays on MuJoCo viewer"""
        # Note: MuJoCo's new API has limited overlay support
        # This is a placeholder for potential future implementations
        pass
    
    def close(self):
        """Close the viewer"""
        if self.viewer:
            self.viewer.close()

class RealtimePlotter:
    """Real-time plotting for training metrics"""
    
    def __init__(self, max_points: int = 1000):
        self.max_points = max_points
        self.data_queue = queue.Queue()
        self.fig = None
        self.axes = None
        self.lines = {}
        self.history = {
            'timesteps': deque(maxlen=max_points),
            'total_reward': deque(maxlen=max_points),
            'aspiration_reward': deque(maxlen=max_points),
            'dispensing_reward': deque(maxlen=max_points),
            'ball_loss_penalty': deque(maxlen=max_points),
            'phase_violation_penalty': deque(maxlen=max_points),
            'particles_held': deque(maxlen=max_points),
            'pipette_x': deque(maxlen=max_points),
            'pipette_y': deque(maxlen=max_points),
            'pipette_z': deque(maxlen=max_points)
        }
    
    def setup_plots(self):
        """Setup matplotlib figures and axes"""
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 10))
        self.fig.suptitle('Pipette Training Visualization', fontsize=16)
        
        # Reward components plot
        ax = self.axes[0, 0]
        ax.set_title('Reward Components')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Reward')
        self.lines['total_reward'] = ax.plot([], [], 'k-', label='Total', linewidth=2)[0]
        self.lines['aspiration_reward'] = ax.plot([], [], 'g-', label='Aspiration')[0]
        self.lines['dispensing_reward'] = ax.plot([], [], 'b-', label='Dispensing')[0]
        self.lines['ball_loss_penalty'] = ax.plot([], [], 'r-', label='Ball Loss')[0]
        self.lines['phase_violation_penalty'] = ax.plot([], [], 'm-', label='Phase Violation')[0]
        ax.legend()
        ax.grid(True)
        
        # Pipette position plot
        ax = self.axes[0, 1]
        ax.set_title('Pipette Position')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Position')
        self.lines['pipette_x'] = ax.plot([], [], 'r-', label='X')[0]
        self.lines['pipette_y'] = ax.plot([], [], 'g-', label='Y')[0]
        self.lines['pipette_z'] = ax.plot([], [], 'b-', label='Z')[0]
        ax.legend()
        ax.grid(True)
        
        # Particles held plot
        ax = self.axes[0, 2]
        ax.set_title('Particles Held')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Count')
        self.lines['particles_held'] = ax.plot([], [], 'o-', label='Particles')[0]
        ax.legend()
        ax.grid(True)
        
        # 2D top-down view
        ax = self.axes[1, 0]
        ax.set_title('Top-Down View')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(-0.15, 0.15)
        
        # Draw wells
        wells = {1: (-0.12, 0), 2: (0, 0), 3: (0.12, 0)}
        for well_id, (x, y) in wells.items():
            circle = plt.Circle((x, y), 0.05, fill=False, color='blue', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y-0.08, f'Well {well_id}', ha='center', va='top')
        
        self.pipette_marker = ax.plot([], [], 'ro', markersize=10, label='Pipette')[0]
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal')
        
        # Task phase timeline
        ax = self.axes[1, 1]
        ax.set_title('Task Phase Timeline')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Phase')
        
        # State text display
        ax = self.axes[1, 2]
        ax.set_title('Current State')
        ax.axis('off')
        self.state_text = ax.text(0.1, 0.9, '', transform=ax.transAxes, 
                                 fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.ion()  # Interactive mode
        plt.show()
    
    def update_data(self, vis_state: VisualizationState):
        """Add new data point"""
        self.history['timesteps'].append(vis_state.timestep)
        self.history['total_reward'].append(vis_state.total_reward)
        self.history['aspiration_reward'].append(vis_state.reward_components.get('aspiration_component', 0))
        self.history['dispensing_reward'].append(vis_state.reward_components.get('dispensing_component', 0))
        self.history['ball_loss_penalty'].append(vis_state.reward_components.get('ball_loss_penalty', 0))
        self.history['phase_violation_penalty'].append(vis_state.reward_components.get('phase_violation_penalty', 0))
        self.history['particles_held'].append(vis_state.particles_held)
        self.history['pipette_x'].append(vis_state.pipette_position[0])
        self.history['pipette_y'].append(vis_state.pipette_position[1])
        self.history['pipette_z'].append(vis_state.pipette_position[2])
    
    def update_plots(self, vis_state: VisualizationState):
        """Update all plots with latest data"""
        if not self.fig:
            return
        
        self.update_data(vis_state)
        
        timesteps = list(self.history['timesteps'])
        
        # Update line plots
        for key, line in self.lines.items():
            if key in self.history:
                data = list(self.history[key])
                line.set_data(timesteps, data)
        
        # Update pipette position marker
        if len(timesteps) > 0:
            self.pipette_marker.set_data([vis_state.pipette_position[0]], [vis_state.pipette_position[1]])
        
        # Update state text
        state_info = f"""
Timestep: {vis_state.timestep}
Pipette State: {vis_state.pipette_state}
Task Phase: {vis_state.task_phase}
Particles Held: {vis_state.particles_held}
Current Well: {vis_state.current_well}
Total Reward: {vis_state.total_reward:.3f}

Position:
  X: {vis_state.pipette_position[0]:.3f}
  Y: {vis_state.pipette_position[1]:.3f}
  Z: {vis_state.pipette_position[2]:.3f}

Last Action:
  [{', '.join(f'{a:.2f}' for a in vis_state.action)}]
        """.strip()
        
        self.state_text.set_text(state_info)
        
        # Auto-scale axes
        for ax in self.axes.flat:
            if ax.get_title() in ['Reward Components', 'Pipette Position', 'Particles Held']:
                ax.relim()
                ax.autoscale_view()
        
        # Refresh display
        plt.pause(0.01)

class TrainingVisualizer:
    """Main visualization coordinator"""
    
    def __init__(self, model=None, data=None, use_mujoco=True, use_plots=True):
        self.use_mujoco = use_mujoco and MUJOCO_AVAILABLE and model is not None
        self.use_plots = use_plots
        
        # Initialize visualizers
        self.mujoco_viz = None
        self.plotter = None
        
        if self.use_mujoco:
            self.mujoco_viz = MuJoCoVisualizer(model, data)
            self.mujoco_viz.setup_viewer()
        
        if self.use_plots:
            self.plotter = RealtimePlotter()
            self.plotter.setup_plots()
        
        # Statistics
        self.update_count = 0
        self.last_update_time = time.time()
    
    def update(self, env, action: np.ndarray, reward: float, info: Dict):
        """Update all visualizations with current environment state"""
        try:
            # Extract state information
            vis_state = self._extract_visualization_state(env, action, reward, info)
            
            # Update MuJoCo viewer
            if self.mujoco_viz:
                self.mujoco_viz.update_visualization(vis_state)
            
            # Update plots
            if self.plotter:
                self.plotter.update_plots(vis_state)
            
            self.update_count += 1
            
            # Performance monitoring
            if self.update_count % 100 == 0:
                current_time = time.time()
                fps = 100 / (current_time - self.last_update_time)
                print(f"Visualization FPS: {fps:.1f}")
                self.last_update_time = current_time
                
        except Exception as e:
            print(f"Visualization update error: {e}")
    
    def _extract_visualization_state(self, env, action: np.ndarray, reward: float, info: Dict) -> VisualizationState:
        """Extract visualization state from environment"""
        
        # Get physics state
        physics_state = env.physics_sim.get_detailed_state_dict()
        
        # Get particle positions
        particle_positions = [p.position.copy() for p in env.physics_sim.particles]
        
        # Well positions
        well_positions = {
            1: np.array([-0.12, 0.0, 0.01]),
            2: np.array([0.0, 0.0, 0.01]),
            3: np.array([0.12, 0.0, 0.01])
        }
        
        return VisualizationState(
            timestep=env.episode_step,
            pipette_position=physics_state['pipette_tip_position'],
            pipette_state=physics_state['pipette_state'],
            task_phase=physics_state['task_phase'],
            particles_held=physics_state['held_particle_count'],
            particles_positions=particle_positions,
            particles_held_ids=physics_state['held_particle_ids'],
            well_positions=well_positions,
            current_well=physics_state.get('current_well'),
            reward_components=info.get('reward_breakdown', {}),
            total_reward=reward,
            action=action
        )
    
    def close(self):
        """Close all visualizations"""
        if self.mujoco_viz:
            self.mujoco_viz.close()
        
        if self.plotter and self.plotter.fig:
            plt.close(self.plotter.fig)

# Integration with training loop
class VisualizationWrapper:
    """Wrapper to easily add visualization to any training loop"""
    
    def __init__(self, env):
        self.env = env
        self.visualizer = TrainingVisualizer(
            model=getattr(env, 'model', None),
            data=getattr(env, 'data', None),
            use_mujoco=True,
            use_plots=True
        )
        
    def step(self, action):
        """Wrapped step function with visualization"""
        obs, reward, done, info = self.env.step(action)
        
        # Update visualization
        self.visualizer.update(self.env, action, reward, info)
        
        return obs, reward, done, info
    
    def reset(self):
        """Wrapped reset function"""
        return self.env.reset()
    
    def close(self):
        """Close environment and visualization"""
        self.visualizer.close()
        self.env.close()
    
    def __getattr__(self, name):
        """Delegate other attributes to the wrapped environment"""
        return getattr(self.env, name)

# Simple usage examples
def example_training_with_visualization():
    """Example showing how to add visualization to training"""
    from integrated_pipette_environment import IntegratedPipetteEnv
    
    # Create environment
    base_env = IntegratedPipetteEnv("particle_pipette_system.xml")
    
    # Wrap with visualization
    env = VisualizationWrapper(base_env)
    
    print("Starting training with visualization...")
    print("Close the plot window to stop training")
    
    try:
        for episode in range(100):
            obs = env.reset()
            episode_reward = 0
            
            for step in range(200):
                # Random action for demo
                action = env.action_space.sample()
                
                # Step with visualization
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                
                # Small delay for visualization
                time.sleep(0.02)
                
                if done:
                    break
            
            print(f"Episode {episode}: Reward = {episode_reward:.2f}")
            
    except KeyboardInterrupt:
        print("Training interrupted")
    except Exception as e:
        print(f"Training error: {e}")
    finally:
        env.close()

def quick_visualization_test():
    """Quick test of visualization components"""
    print("Testing visualization components...")
    
    # Test plotter only
    plotter = RealtimePlotter()
    plotter.setup_plots()
    
    # Generate some test data
    for i in range(100):
        test_state = VisualizationState(
            timestep=i,
            pipette_position=np.array([0.1*np.sin(i*0.1), 0.1*np.cos(i*0.1), 0.05*np.sin(i*0.05)]),
            pipette_state="holding" if i % 20 < 10 else "aspirating",
            task_phase="transport" if i % 30 < 15 else "aspirate",
            particles_held=max(0, int(3*np.sin(i*0.2))),
            particles_positions=[],
            particles_held_ids=[],
            well_positions={1: np.array([-0.12, 0, 0.01]), 2: np.array([0, 0, 0.01]), 3: np.array([0.12, 0, 0.01])},
            current_well=1 if i % 40 < 20 else 3,
            reward_components={
                'aspiration_component': 2*np.sin(i*0.1),
                'dispensing_component': 1.5*np.cos(i*0.1),
                'ball_loss_penalty': -0.5*abs(np.sin(i*0.15)),
                'phase_violation_penalty': -1*abs(np.cos(i*0.12))
            },
            total_reward=np.sin(i*0.1) + 0.5*np.cos(i*0.2),
            action=np.array([0.5*np.sin(i*0.1), 0.3*np.cos(i*0.1), 0.2*np.sin(i*0.05), 0.4*np.cos(i*0.08)])
        )
        
        plotter.update_plots(test_state)
        time.sleep(0.05)
    
    input("Press Enter to close...")
    plt.close('all')

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        quick_visualization_test()
    else:
        example_training_with_visualization()