# Pipette Physics Simulation - Setup and Usage Guide

## File Structure

Your project should be organized as follows:

```
pipette_project/
├── particle_pipette_system.xml          # Your MuJoCo XML file
├── pipette_physics_simulation.py        # Core physics simulation
├── integrated_pipette_environment.py    # MuJoCo + Physics integration
├── test_integration.py                  # Test suite and examples
├── train_actor_critic.py               # RL training script (see below)
└── SETUP_AND_USAGE.md                  # This guide
```

## Installation Requirements

### Required Packages

```bash
pip install numpy matplotlib gym mujoco-py
```

### Optional (for advanced RL)

```bash
pip install stable-baselines3 torch tensorboard
```

## Quick Start

### 1. Test Your Setup

```bash
# Run all tests
python test_integration.py

# Run interactive demo
python test_integration.py demo
```

### 2. Test Physics Only (No MuJoCo Required)

```python
from pipette_physics_simulation import PipettePhysicsSimulator, PipetteConfig

# Create simulator
sim = PipettePhysicsSimulator()

# Add some particles
sim.add_particle(np.array([0.0, 0.0, 0.02]), particle_id=0)

# Update pipette state
tip_position = np.array([0.0, 0.0, 0.025])
plunger_depth = 0.8  # 0-1 range
sim.update_pipette_state(tip_position, plunger_depth, dt=0.01)

# Check results
state = sim.get_state_dict()
print(f"Particles held: {state['held_particle_count']}")
```

### 3. Test Full Environment

```python
from integrated_pipette_environment import IntegratedPipetteEnv

# Create environment
env = IntegratedPipetteEnv("particle_pipette_system.xml")

# Standard OpenAI Gym interface
obs = env.reset()
action = env.action_space.sample()  # [x, y, z, plunger]
obs, reward, done, info = env.step(action)
env.render()
env.close()
```

## Integration with Actor-Critic Algorithms

### Basic Integration Example

```python
import gym
from stable_baselines3 import PPO
from integrated_pipette_environment import IntegratedPipetteEnv

# Create environment
env = IntegratedPipetteEnv("particle_pipette_system.xml")

# Train with PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Test trained model
obs = env.reset()
for i in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()
```

## Configuration Options

### Physics Configuration

```python
from pipette_physics_simulation import PipetteConfig

config = PipetteConfig(
    max_capacity=5,              # Max particles in pipette
    suction_range=0.03,          # Suction range in meters
    min_suction_depth=0.02,      # Minimum plunger depth for suction
    max_suction_depth=0.08,      # Maximum effective suction depth
    suction_force_max=10.0,      # Maximum suction force
    particle_cohesion=0.2        # Particle attraction strength
)

sim = PipettePhysicsSimulator(config)
```

### Environment Configuration

```python
# Custom reward function
class CustomPipetteEnv(IntegratedPipetteEnv):
    def _calculate_reward(self, physics_reward, physics_info):
        # Add your custom reward logic
        custom_reward = physics_reward

        # Example: Bonus for transferring specific particles
        if self._particles_in_target_well() >= 3:
            custom_reward += 20.0

        return custom_reward

env = CustomPipetteEnv("particle_pipette_system.xml")
```

## Troubleshooting

### Common Issues

1. **"No module named 'mujoco_py'"**

   - Install MuJoCo and mujoco-py following the official guide
   - Alternative: Run physics-only tests with `test_integration.py`

2. **"XML file not found"**

   - Ensure your XML file is in the same directory
   - Update the path in `IntegratedPipetteEnv` constructor

3. **"Mass and inertia error"**

   - Your XML file needs proper `<inertial>` tags for moving bodies
   - Use the provided XML structure as reference

4. **Particles not being picked up**
   - Check plunger depth range (should be 0.6-1.0 for strong suction)
   - Verify pipette is close enough to particles (< 0.03m)
   - Ensure plunger is moving upward (negative velocity)

### Debug Mode

```python
# Enable detailed logging
sim = PipettePhysicsSimulator()
sim.debug_mode = True  # Add this to see detailed state information

# Check particle states
for particle in sim.particles:
    print(f"Particle {particle.id}: pos={particle.position}, held={particle.is_held}")

# Check pipette state
state = sim.get_state_dict()
print(f"Pipette state: {state}")
```

## Performance Tips

### For Training

- Use `render_mode='rgb_array'` for headless training
- Reduce `max_episode_steps` for faster iterations
- Use multiple parallel environments for faster training

### For Real-time Visualization

- Increase `timestep` in XML for smoother visualization
- Reduce particle count for better performance
- Use simpler reward functions

## Example Training Scripts

### PPO Training

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Create vectorized environment
env = make_vec_env(lambda: IntegratedPipetteEnv("particle_pipette_system.xml"), n_envs=4)

# Train
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./pipette_tensorboard/")
model.learn(total_timesteps=100000)
model.save("pipette_ppo_model")

# Evaluate
model = PPO.load("pipette_ppo_model")
# ... evaluation code
```

### Custom Actor-Critic

```python
import torch
import torch.nn as nn

class PipetteActorCritic(nn.Module):
    def __init__(self, obs_dim=25, action_dim=4):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.actor = nn.Linear(256, action_dim)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        shared = self.shared(x)
        return self.actor(shared), self.critic(shared)
```

## Next Steps

1. **Test the basic setup** with `test_integration.py`
2. **Customize the physics** parameters for your specific use case
3. **Define your task** by modifying the reward function
4. **Train your agent** using your preferred RL library
5. **Evaluate and iterate** on the results

For questions or issues, check the test output and debug information provided by the test suite.
