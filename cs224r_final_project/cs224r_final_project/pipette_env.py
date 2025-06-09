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
        
        # Variable droplet count
        self.num_droplets = getattr(cfg.env, 'num_droplets', 3)
        print(f"🧪 Pipette Environment: {self.num_droplets} droplets")
        
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

        # Tell reward function about droplet count
        self.reward_function.set_droplet_count(self.num_droplets)
        
        # Initialize state variables
        self.reset()
    
    def reset(self):
        # Reset pipette to starting position above source well
        self.pipette_pos = np.array([2.0, 2.0, 3.0])  
        self.plunger_depth = 0.0 
        
        # Reset droplets to random positions in source well
        self.droplets = []

        # Only spawn num_droplets, then pad to 3
        for i in range(self.num_droplets):
            angle_rad = np.random.uniform(0, 2 * np.pi)
            dist_from_center = np.random.uniform(0, self.well_radius * 0.8) 
            
            droplet_x = self.source_well_pos[0] + dist_from_center * np.cos(angle_rad)
            droplet_y = self.source_well_pos[1] + dist_from_center * np.sin(angle_rad)
            droplet_z = self.source_well_pos[2] + 0.1
            
            self.droplets.append(np.array([droplet_x, droplet_y, droplet_z]))
        
        # Pad with dummy droplets for consistent observation space because our dimension hardcodes 3 droplets
        while len(self.droplets) < 3:
            self.droplets.append(np.array([0.0, 0.0, 0.0]))
        
        # State tracking adjusted for variable droplet count
        self.aspirated_droplets = [False] * self.num_droplets + [True] * (3 - self.num_droplets)
        self.droplets_in_target = [False] * self.num_droplets + [True] * (3 - self.num_droplets)
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
        # Only simulate physics for active droplets
        for i in range(self.num_droplets):
            droplet_pos = self.droplets[i]
            
            # Mimic aspirating - only aspirate droplets that haven't been picked up yet 
            if not self.aspirated_droplets[i]: 
                
                # Calculate XY distance (ignore Z for aspiration check)
                xy_distance = np.sqrt((droplet_pos[0] - self.pipette_pos[0])**2 + 
                                    (droplet_pos[1] - self.pipette_pos[1])**2)
                
                carried_droplets = sum(self.aspirated_droplets[:i])
                tolerance_bonus = carried_droplets * 0.3
                
                close_enough = xy_distance < (0.5 + tolerance_bonus)
                z_low_enough = self.pipette_pos[2] < 1.0 
                suction_on = self.plunger_depth > 0.3
                
                if close_enough and z_low_enough and suction_on:
                    self.aspirated_droplets[i] = True
            
            if self.aspirated_droplets[i] and not self.droplets_in_target[i]:
                # Arrange multiple droplets around pipette tip to avoid overlap
                carried_count = sum(self.aspirated_droplets[:i])
                
                if carried_count == 0:
                    # First droplet: directly below pipette
                    offset = np.array([0, 0, -0.2])
                elif carried_count == 1:
                    # Second droplet: slightly offset
                    offset = np.array([0.1, 0, -0.2])
                else:
                    # Third droplet: different offset
                    offset = np.array([-0.1, 0, -0.2])
                
                self.droplets[i] = self.pipette_pos + offset
            
            if self.aspirated_droplets[i] and not self.droplets_in_target[i]:
                
                target_distance = np.sqrt((self.pipette_pos[0] - self.target_well_pos[0])**2 + 
                                        (self.pipette_pos[1] - self.target_well_pos[1])**2)
                
                at_target = target_distance < self.well_radius
                down_enough = self.pipette_pos[2] < 1.5
                plunger_dispensing = self.plunger_depth < 0.7
                
                # This prevents all droplets from dispensing simultaneously
                droplets_already_dispensed = sum(self.droplets_in_target[:i])
                is_next_to_dispense = droplets_already_dispensed == i
                
                if at_target and down_enough and plunger_dispensing and is_next_to_dispense:
                    self.droplets_in_target[i] = True
                    
                    # Place droplet in target well with slight separation
                    angle = np.random.uniform(0, 2*np.pi)
                    radius = np.random.uniform(0, self.well_radius * 0.6)  # Slightly smaller radius for separation
                    x = self.target_well_pos[0] + radius * np.cos(angle)
                    y = self.target_well_pos[1] + radius * np.sin(angle)
                    z = self.target_well_pos[2] + 0.1
                    self.droplets[i] = np.array([x, y, z])
    

    # Check if all active droplets have been transferred
    def _check_success(self):
        return all(self.droplets_in_target[:self.num_droplets])
    
    # Check what phase we are in based on active droplets
    def _get_current_phase(self):
        # Count how many trues in each buffer for how many have been aspirated/dispensed
        aspirated_count = sum(self.aspirated_droplets[:self.num_droplets])
        dispensed_count = sum(self.droplets_in_target[:self.num_droplets])
        
        # The agent has picked up done 
        if aspirated_count == 0:
            return "approaching_source"
        # The agent has began aspirating 
        elif aspirated_count < self.num_droplets:
            return "aspirating" 
        # The agent has began transferring completed aspirating  
        elif dispensed_count == 0:
            return "transferring"
        # The agent is now dispesning 
        elif dispensed_count < self.num_droplets:
            return "dispensing"
        # All active droplets aspirated/dispensed - completed 
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
        # Aspirated progress based on active droplets
        aspirated_fraction = sum(self.aspirated_droplets[:self.num_droplets]) / max(1, self.num_droplets)
        obs.append(aspirated_fraction)
        return np.array(obs, dtype=np.float32)
    
    # when the everything is dispensed the episode terminates