import numpy as np
import os
from gym import utils
from gym import spaces
from gym.envs.mujoco import mujoco_env
import mujoco_py

class PipetteEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        # Create temp directory for model files if it doesn't exist
        os.makedirs("tmp", exist_ok=True)
        
        # Create a simplified pipette model XML
        self.model_xml = self._create_model_xml()
        with open("tmp/pipette_model.xml", "w") as f:
            f.write(self.model_xml)
        
        # Initialize MuJoCo environment
        mujoco_env.MujocoEnv.__init__(self, "tmp/pipette_model.xml", 5)
        utils.EzPickle.__init__(self)
        
        # Define observation and action spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
        
        # Action space: [pipette_x, pipette_y, pipette_z, plunger]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # Simulation parameters
        self.target_volume = 100.0  # Target volume in μL
        self.current_volume = 0.0   # Volume in pipette
        self.max_steps = 100
        self.steps = 0

    def _create_model_xml(self):
        """Generate a simplified MuJoCo XML model"""
        return f"""
        <mujoco model="pipette_env">
            <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
            <option integrator="RK4" timestep="0.01"/>
            <default>
                <joint armature="1" damping="1" limited="true"/>
                <geom conaffinity="1" condim="3" density="100" friction="1 0.5 0.5" margin="0.002" rgba="0.8 0.6 0.4 1"/>
            </default>
            <asset>
                <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
                <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
                <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
                <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
                <material name="geom" texture="texgeom" texuniform="true"/>
            </asset>
            <worldbody>
                <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
                <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="40 40 40" type="plane"/>
                
                <!-- Source container -->
                <body name="source" pos="0.3 0 0.05">
                    <geom name="source_geom" type="cylinder" size="0.05 0.05" rgba="0.2 0.2 0.8 0.5"/>
                    <geom name="liquid_source" type="cylinder" size="0.045 0.03" pos="0 0 0.03" rgba="0 0 1 0.5"/>
                </body>
                
                <!-- Destination container -->
                <body name="destination" pos="-0.3 0 0.05">
                    <geom name="dest_geom" type="cylinder" size="0.05 0.05" rgba="0.2 0.8 0.2 0.5"/>
                    <geom name="liquid_dest" type="cylinder" size="0.045 0.001" pos="0 0 0.001" rgba="0 0 1 0.5"/>
                </body>
                
                <!-- Pipette -->
                <body name="pipette" pos="0 0 0.3">
                    <joint name="pipette_x" type="slide" axis="1 0 0" limited="true" range="-0.5 0.5"/>
                    <joint name="pipette_y" type="slide" axis="0 1 0" limited="true" range="-0.5 0.5"/>
                    <joint name="pipette_z" type="slide" axis="0 0 1" limited="true" range="0.1 0.4"/>
                    
                    <!-- Pipette body -->
                    <geom name="pipette_body" type="cylinder" size="0.01 0.05" rgba="0.7 0.7 0.7 1"/>
                    
                    <!-- Pipette tip -->
                    <body name="tip" pos="0 0 -0.06">
                        <geom name="pipette_tip" type="cone" size="0.007 0.02" rgba="0.9 0.9 0.9 1"/>
                        
                        <!-- Plunger - represented as a site -->
                        <site name="plunger" pos="0 0 0.01" size="0.005" rgba="1 0 0 1"/>
                    </body>
                </body>
                
                <!-- Sites for visualization -->
                <site name="source_top" pos="0.3 0 0.1" size="0.01" rgba="1 0 0 1"/>
                <site name="dest_top" pos="-0.3 0 0.1" size="0.01" rgba="1 0 0 1"/>
            </worldbody>
            
            <actuator>
                <position name="pip_x" joint="pipette_x" kp="100" ctrlrange="-1 1"/>
                <position name="pip_y" joint="pipette_y" kp="100" ctrlrange="-1 1"/>
                <position name="pip_z" joint="pipette_z" kp="100" ctrlrange="-1 1"/>
                <position name="plunger" joint="pipette_x" kp="0" ctrlrange="-1 1"/> <!-- Dummy joint for plunger control -->
            </actuator>
            
            <sensor>
                <framepos name="pipette_pos" objtype="body" objname="pipette"/>
                <framepos name="tip_pos" objtype="body" objname="tip"/>
                <framepos name="source_pos" objtype="body" objname="source"/>
                <framepos name="dest_pos" objtype="body" objname="destination"/>
            </sensor>
        </mujoco>
        """

    def step(self, action):
        # Extract actions
        x_action, y_action, z_action, plunger_action = action
        
        # Apply position control actions to pipette
        self.data.ctrl[0] = x_action  # x position
        self.data.ctrl[1] = y_action  # y position
        self.data.ctrl[2] = z_action  # z position
        
        # Simulate plunger action (aspiration/dispensing)
        # In a real implementation, we'd modify the fluid level visually
        # Here we'll track it internally for simplicity
        tip_pos = self.data.get_site_xpos('plunger')
        source_pos = self.data.get_site_xpos('source_top')
        dest_pos = self.data.get_site_xpos('dest_top')
        
        # Check if tip is in source container
        in_source = np.linalg.norm(tip_pos[:2] - source_pos[:2]) < 0.045 and tip_pos[2] < source_pos[2]
        
        # Check if tip is in destination container
        in_dest = np.linalg.norm(tip_pos[:2] - dest_pos[:2]) < 0.045 and tip_pos[2] < dest_pos[2]
        
        # Simulate aspiration/dispensing based on plunger action
        if plunger_action > 0.5:  # Aspirate
            if in_source and self.current_volume < self.target_volume:
                # Aspirate fluid
                volume_change = min(10.0, self.target_volume - self.current_volume)
                self.current_volume += volume_change
                
        elif plunger_action < -0.5:  # Dispense
            if in_dest and self.current_volume > 0:
                # Dispense fluid
                volume_change = min(10.0, self.current_volume)
                self.current_volume -= volume_change
        
        # Do the simulation
        self.do_simulation(self.data.ctrl, self.frame_skip)
        self.steps += 1
        
        # Calculate reward
        reward = self._get_reward(in_source, in_dest)
        
        # Check if done
        done = self.steps >= self.max_steps or (self.current_volume == 0 and self._dispensed_correct_amount())
        
        # Get observation
        ob = self._get_obs()
        
        return ob, reward, done, {'volume': self.current_volume}

    def _get_obs(self):
        # Get positions
        pipette_pos = self.data.get_body_xpos('pipette')
        tip_pos = self.data.get_body_xpos('tip')
        source_pos = self.data.get_body_xpos('source')
        dest_pos = self.data.get_body_xpos('destination')
        
        # Create observation
        obs = np.concatenate([
            pipette_pos,          # 3 values
            tip_pos,              # 3 values
            source_pos,           # 3 values
            dest_pos,             # 3 values
            [self.current_volume] # 1 value
        ])
        
        return obs

    def _get_reward(self, in_source, in_dest):
        reward = 0
        
        # Small negative reward for each step (encourages efficiency)
        reward -= 0.1
        
        # Check for collisions (hitting container walls)
        tip_pos = self.data.get_site_xpos('plunger')
        source_pos = self.data.get_body_xpos('source')
        dest_pos = self.data.get_body_xpos('destination')
        
        # Distance to source and dest centers (xy plane)
        dist_to_source = np.linalg.norm(tip_pos[:2] - source_pos[:2])
        dist_to_dest = np.linalg.norm(tip_pos[:2] - dest_pos[:2])
        
        # Collision penalties
        if 0.045 < dist_to_source < 0.05 and tip_pos[2] < source_pos[2] + 0.1:
            reward -= 5.0  # Penalty for hitting source wall
            
        if 0.045 < dist_to_dest < 0.05 and tip_pos[2] < dest_pos[2] + 0.1:
            reward -= 5.0  # Penalty for hitting destination wall
        
        # Reward for successful operations
        if in_source and self.current_volume > 0:
            reward += 0.5  # Small reward for successful aspiration
            
        if in_dest and self._dispensed_correct_amount():
            reward += 20.0  # Large reward for completing the task
            
        return reward

    def _dispensed_correct_amount(self):
        # Check if we've dispensed the target volume
        # In a more complex implementation, this would track how much
        # was dispensed into the destination container
        return abs(self.target_volume - self.current_volume) < 1.0

    def reset(self):
        self.steps = 0
        self.current_volume = 0.0
        mujoco_env.MujocoEnv.reset(self)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = 1.0
        self.viewer.cam.elevation = -20