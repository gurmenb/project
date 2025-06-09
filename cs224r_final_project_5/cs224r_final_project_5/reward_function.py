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
        
        # Internal tracking
        self._last_aspirated = 0
        self._last_dispensed = 0
    
    # Calculate the reward for the current step 
    def calculate_reward(self, env):
        reward = 0.0 
        
        # Get current state from environment count number of trues
        aspirated_count = sum(env.aspirated_droplets)
        dispensed_count = sum(env.droplets_in_target)
        
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
        
        # Task completion reward
        if all(env.droplets_in_target):
            completion_reward = self.weights['completion']
            reward += completion_reward
        
        # Source well guidance reward only guide toward source well during aspiration phase 
        if aspirated_count < 3:
            # Calculate how close the pipette is to the source well 
            tip_to_source = np.linalg.norm(env.pipette_pos[:2] - env.source_well_pos[:2])
            # guidance_factor = max(0, 1 - distance/3) - Linear decay 
            guidance = max(0, 1.0 - tip_to_source/3.0) * self.weights['guidance']
            reward += guidance
            
        # Target well guidance reward only when dispensing
        elif dispensed_count < 3:
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