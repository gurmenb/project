import numpy as np
import gymnasium as gym
from gymnasium import spaces
from reward_function import PipetteRewardFunction

class PipetteEnv(gym.Env):

    def __init__(self, cfg):
        super().__init__()
        
        # Environment parameters
        self.workspace_size = cfg.env.environment_size
        self.well_radius = cfg.env.well_radius  
        self.well_depth = cfg.env.well_depth
        self.max_steps = cfg.env.max_episode_steps
        
        # Well positions X, Y, Z 
        self.source_well_pos = np.array([2.0, 2.0, 0.0]) # 2, 2
        self.target_well_pos = np.array([8.0, 8.0, 0.0]) # 8,8
        
        # Action space: [x_pos, y_pos, z_pos, plunger_depth] normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        
        # Observation space: [pipette_xyz(3), plunger_pos(1), droplet1_xyz(3), 
        # droplet2_xyz(3), droplet3_xyz(3), task_flag(1)] Total: 14 dimensions
        obs_low = np.array([
            0.0, 0.0, 0.0, 0.0,      
            0.0, 0.0, 0.0,         
            0.0, 0.0, 0.0,          
            0.0, 0.0, 0.0,         
            0.0                      
        ])

        obs_high = np.array([
            self.workspace_size, self.workspace_size, 3.0, 1.0, 
            self.workspace_size, self.workspace_size, 3.0,      
            self.workspace_size, self.workspace_size, 3.0,      
            self.workspace_size, self.workspace_size, 3.0,      
            1.0                                                
        ])
        
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, shape=(14,), dtype=np.float32
        )
        
        # Initialize reward function
        self.reward_function = PipetteRewardFunction()
        
        # Initialize state variables
        self.reset()
    
    def reset(self):
        # Reset pipette to starting position above source well
        self.pipette_pos = np.array([2.0, 2.0, 3.0])  
        self.plunger_depth = 0.0 
        
        # Reset droplets to random positions in source well
        self.droplets = []

        for i in range(3):
            angle_rad = np.random.uniform(0, 2 * np.pi)
            dist_from_center = np.random.uniform(0, self.well_radius * 0.8) 
            
            droplet_x = self.source_well_pos[0] + dist_from_center * np.cos(angle_rad)
            droplet_y = self.source_well_pos[1] + dist_from_center * np.sin(angle_rad)
            droplet_z = self.source_well_pos[2] + 0.1
            
            self.droplets.append(np.array([droplet_x, droplet_y, droplet_z]))
        
        # State tracking
        self.aspirated_droplets = [False, False, False]
        self.droplets_in_target = [False, False, False]
        self.step_count = 0
        
        # Action tracking for reward function
        self.prev_action = None
        self.current_action = None
        
        # Reset reward function
        self.reward_function.reset()

        return self._get_observation(), {}  
    
    def step(self, action):
        self.step_count += 1
        
        # Store action for reward calculation
        self.current_action = action.copy()
        
        # Ensure action values are within bounds
        action = np.clip(action, -1, 1)
        
        # Apply movement to pipette
        move_scale = 0.2  

        self.pipette_pos[0] += action[0] * move_scale
        self.pipette_pos[1] += action[1] * move_scale  
        self.pipette_pos[2] += action[2] * move_scale
        
        # Constrain pipette to workspace
        workspace_min = [0.2, 0.2, 0]
        workspace_max = [self.workspace_size-0.2, self.workspace_size-0.2, 3.0]
        self.pipette_pos = np.clip(self.pipette_pos, workspace_min, workspace_max)
        
        # Set plunger depth from action scaling to [0, 1]
        plunger_action = action[3]
        self.plunger_depth = (plunger_action + 1) / 2  
        
        # Run physics simulation
        self._droplet_physics()
        
        # Get reward using reward function
        reward = self.reward_function.calculate_reward(self)
        
        # Check ending conditions
        task_complete = self._check_success()
        time_limit_reached = self.step_count >= self.max_steps
        
        # Store action for next step
        self.prev_action = action.copy()
        
        obs = self._get_observation()

        return obs, reward, task_complete, time_limit_reached, {}
    
    def _droplet_physics(self):
        
        for i in range(len(self.droplets)):
            droplet_pos = self.droplets[i]
            
            # Mimic aspirating - only aspirate droplets that haven't been picked up yet 
            if not self.aspirated_droplets[i]: 
                
                # Calculate XY distance (ignore Z for aspiration check)
                # √[(x2-x1)^2 + (y2-y1)^2]
                xy_distance = np.sqrt((droplet_pos[0] - self.pipette_pos[0])**2 + 
                                     (droplet_pos[1] - self.pipette_pos[1])**2)
                
                # Check all conditions: pipette needs to be 0.5 units horizontally and below 1.0 height
                close_enough = xy_distance < 0.5
                z_low_enough = self.pipette_pos[2] < 1.0 
                # Check if the suction is on greater than 30% retraction 
                suction_on = self.plunger_depth > 0.3
                
                # All three conditions for aspiration is met aspirate it 
                if close_enough and z_low_enough and suction_on:
                    self.aspirated_droplets[i] = True
            
            # Grabbing the droplet  
            if self.aspirated_droplets[i] and not self.droplets_in_target[i]:
                # Aspirated droplets follow the pipette place it 0.2 units below tip
                self.droplets[i] = self.pipette_pos + np.array([0, 0, -0.2])
            
            # Dispensing the droplet - make sure it has been aspirated first 
            if self.aspirated_droplets[i] and not self.droplets_in_target[i]:
                
                # Check if at target well by calculating distance between current location and target well
                target_distance = np.sqrt((self.pipette_pos[0] - self.target_well_pos[0])**2 + 
                                         (self.pipette_pos[1] - self.target_well_pos[1])**2)
                
                # Make sure within well radius of target well 
                at_target = target_distance < self.well_radius
                # Make sure pipette below height of 1.5 before dispersing 
                down_enough = self.pipette_pos[2] < 1.5
                # Make sure we have extended the plunger more than 70% to release suction 
                plunger_dispensing = self.plunger_depth < 0.7
                
                if at_target and down_enough and plunger_dispensing:
                    self.droplets_in_target[i] = True
                    
                    # Place droplet in target well
                    angle = np.random.uniform(0, 2*np.pi)
                    radius = np.random.uniform(0, self.well_radius * 0.8)
                    x = self.target_well_pos[0] + radius * np.cos(angle)
                    y = self.target_well_pos[1] + radius * np.sin(angle)
                    z = self.target_well_pos[2] + 0.1
                    self.droplets[i] = np.array([x, y, z])
    

    # check if all the droplets have been transfered (full array of true)
    def _check_success(self):
        return all(self.droplets_in_target)
    
    # check what phase we are in 
    def _get_current_phase(self):
        # Count how many trues in each buffer for how many have been aspirated/dispensed
        aspirated_count = sum(self.aspirated_droplets)
        dispensed_count = sum(self.droplets_in_target)
        
        # The agent has picked up done 
        if aspirated_count == 0:
            return "approaching_source"
        # The agent has began aspirating 
        elif aspirated_count < 3:
            return "aspirating" 
        # The agent has began transferring completed aspirating  
        elif dispensed_count == 0:
            return "transferring"
        # The agent is now dispesning 
        elif dispensed_count < 3:
            return "dispensing"
        # 3 and 3 True aspirating/dispensing completed 
        else:
            return "complete"
    
    # Check if the pipette is in the environment boundaries 
    def _is_at_boundary(self):
        # 0.21 > 0.2 which is our clamp margin 
        margin = 0.21
        # Check X, Y, and Z boundaries 
        return (self.pipette_pos[0] <= margin or 
                self.pipette_pos[0] >= self.workspace_size - margin or
                self.pipette_pos[1] <= margin or 
                self.pipette_pos[1] >= self.workspace_size - margin or
                self.pipette_pos[2] <= 0.01 or 
                self.pipette_pos[2] >= 2.99)
    
    # Build Observation 14D vector
    def _get_observation(self):
        obs = []
        obs.extend(self.pipette_pos)
        obs.append(self.plunger_depth)
        for droplet in self.droplets:
            obs.extend(droplet)
        aspirated_or_not = float(all(self.aspirated_droplets))
        obs.append(aspirated_or_not)
        return np.array(obs, dtype=np.float32)
    # Note we don't need a dispensed_or_not because we get a termination signal when task is completed
    # So when the everything is dispensed the episode terminates 