# 6 reward weights 
# Aspirating and dispensing droplets 
# Completing the entire task 
# Source and target well guidance rewards 
# Hitting the boundaries 
# Jerkey movements 

import numpy as np

# R(t) = w₁·Δaspiration(t) + w₂·Δdispensing(t) + w₃·completion(t) + w₄·guidance(t) 
# - w₅·collision(t) - w₆·jerk(t)
class PipetteRewardFunction:

    def __init__(self):
        # Define reward weights dictionary 
        self.weights = {
            'aspiration': 50.0,     
            'dispensing': 100.0,   
            'completion': 200.0,  
            'guidance': 1.0,       
            'collision': 5.0,  
            'jerk': 0.5,            
        }
        
        # ADD: Task configuration
        self.num_droplets = 3  # Default
        
        # Internal tracking
        self._last_aspirated = 0
        self._last_dispensed = 0
    
    # ADD: Method to set droplet count
    def set_droplet_count(self, num_droplets):
        """Set the number of droplets for this reward function."""
        self.num_droplets = num_droplets
        print(f"🎯 Reward function set for {num_droplets} droplets")
    
    # Calculate the reward for the current step 
    def calculate_reward(self, env):
        reward = 0.0 
        
        # CHANGE: Get current state from environment count number of trues (only active droplets)
        aspirated_count = sum(env.aspirated_droplets[:env.num_droplets])
        dispensed_count = sum(env.droplets_in_target[:env.num_droplets])
        
        # Aspiration reward - reward for new aspirations = diff(aspirations) × weight
        # Only reward if this is a new aspiration 
        if aspirated_count > self._last_aspirated:
            aspiration_reward = (aspirated_count - self._last_aspirated) * self.weights['aspiration']
            reward += aspiration_reward
            self._last_aspirated = aspirated_count
                
        # Dispensing reward - reward for new dispensing = diff(dispensions) × weight
        if dispensed_count > self._last_dispensed:
            dispensing_reward = (dispensed_count - self._last_dispensed) * self.weights['dispensing']
            reward += dispensing_reward
            self._last_dispensed = dispensed_count
        
        # CHANGE: Task completion reward (only when ALL ACTIVE droplets transferred)
        if dispensed_count == env.num_droplets:
            completion_reward = self.weights['completion']
            reward += completion_reward
        
        # CHANGE: Source well guidance reward only guide toward source well during aspiration phase 
        if aspirated_count < env.num_droplets:
            # Calculate how close the pipette is to the source well 
            tip_to_source = np.linalg.norm(env.pipette_pos[:2] - env.source_well_pos[:2])
            # guidance_factor = max(0, 1 - distance/3) - Linear decay 
            guidance = max(0, 1.0 - tip_to_source/3.0) * self.weights['guidance']
            reward += guidance
            
        # CHANGE: Target well guidance reward only when dispensing
        elif dispensed_count < env.num_droplets:
            tip_to_target = np.linalg.norm(env.pipette_pos[:2] - env.target_well_pos[:2])
            guidance = max(0, 1.0 - tip_to_target/3.0) * self.weights['guidance']
            reward += guidance
        
        # Collision penalty
        if self.weights['collision'] > 0:
            collision_penalty = self._calculate_collision_penalty(env)
            reward -= self.weights['collision'] * collision_penalty
        
        # Jerk penalty for smooth movement
        if self.weights['jerk'] > 0:
            jerk_penalty = self._calculate_jerk_penalty(env)
            reward -= self.weights['jerk'] * jerk_penalty
        
        return reward
    
    def _calculate_collision_penalty(self, env):
        if env._is_at_boundary():
            return 1.0
        return 0.0
    
    def _calculate_jerk_penalty(self, env):
        # Can't calculate jerk without both current and previous actions
        if env.prev_action is None or env.current_action is None:
            return 0.0
        
        # Calculate action change jerk = Σ(|current_action - prev_action|)^2
        action_change = np.abs(env.current_action - env.prev_action)
        jerk = np.sum(action_change ** 2)
        
        # Penalty for excessive jerk
        if jerk > 1.0:
            penalty = jerk * 0.1
            return penalty
        
        return 0.0
    
    # Resetting the reward function 
    def reset(self):
        self._last_aspirated = 0
        self._last_dispensed = 0
    
    def get_weights(self):
        return self.weights.copy()
    
    # Merge any new reward weights dictionary into the existing one
    def set_weights(self, new_weights):
        self.weights.update(new_weights)