import numpy as np
import torch

class PipettingRewardFunction:
    def __init__(self, 
                 w_volume=10.0,      # w1: Volume accuracy weight
                 w_time=0.1,         # w2: Time efficiency weight  
                 w_completion=5.0,   # w3: Completion bonus weight
                 w_collision=2.0,    # w4: Collision penalty weight
                 w_drop=10.0,        # w5: Drop penalty weight
                 w_contamination=3.0, # w6: Contamination penalty weight
                 w_miss=5.0,         # w7: Miss penalty weight
                 w_jerk=1.0,         # Jerk penalty weight
                 target_volume=20):  # Target number of balls to transfer
        
        self.weights = {
            'volume': w_volume,
            'time': w_time,
            'completion': w_completion,
            'collision': w_collision,
            'drop': w_drop,
            'contamination': w_contamination,
            'miss': w_miss,
            'jerk': w_jerk
        }
        
        self.target_volume = target_volume
        self.prev_action = None
        
    def compute_reward(self, obs, action, info=None, done=False):
        """
        Compute reward based on observation and action
        
        Observation space (14D):
        0: liquid_in_plunger
        1: balls_in_plunger  
        2: source_well_amount
        3: target_well_amount
        4-6: source_well_position (x,y,z)
        7-9: target_well_position (x,y,z)
        10: aspiration_pressure
        11: dispersion_pressure
        12: task_phase (0=approach, 1=aspirate, 2=transfer, 3=dispense)
        13: submerged
        
        Action space (6D):
        0-2: pipette position (x,y,z)
        3: plunger z position
        4: plunger z force
        5: plunger z speed
        """
        
        reward = 0.0
        
        # Extract relevant observations
        balls_in_plunger = obs[1]
        source_amount = obs[2]
        target_amount = obs[3]
        task_phase = int(obs[12])
        submerged = obs[13]
        
        # 1. Volume accuracy rewards
        volume_reward = self._volume_accuracy_reward(balls_in_plunger, target_amount, task_phase, done)
        reward += self.weights['volume'] * volume_reward
        
        # 2. Time efficiency (small penalty per step)
        reward -= self.weights['time']
        
        # 3. Completion bonus
        if done and abs(target_amount - self.target_volume) <= 1:
            reward += self.weights['completion']
        
        # 4. Collision penalty (from info if available)
        if info and 'collision' in info and info['collision']:
            reward -= self.weights['collision']
        
        # 5. Drop penalty (balls lost from plunger unexpectedly)
        drop_penalty = self._drop_penalty(balls_in_plunger, task_phase)
        reward -= self.weights['drop'] * drop_penalty
        
        # 6. Miss penalty (spillage - liquid not going to target)
        miss_penalty = self._miss_penalty(obs, action, task_phase)
        reward -= self.weights['miss'] * miss_penalty
        
        # 7. Jerk penalty (excessive acceleration)
        jerk_penalty = self._jerk_penalty(action)
        reward -= self.weights['jerk'] * jerk_penalty
        
        # Update previous action for next computation
        self.prev_action = action.copy()
        
        return reward
    
    def _volume_accuracy_reward(self, balls_in_plunger, target_amount, task_phase, done):
        """Reward for correct volume handling"""
        reward = 0.0
        
        # Mid-check: Correct aspiration from source
        if task_phase == 1:  # Aspirating
            target_balls = min(self.target_volume, 50)  # Don't aspirate more than available
            aspiration_accuracy = 1.0 - abs(balls_in_plunger - target_balls) / target_balls
            reward += max(0, aspiration_accuracy)
        
        # Final check: Correct final volume in target
        if done:
            final_accuracy = 1.0 - abs(target_amount - self.target_volume) / self.target_volume
            reward += max(0, final_accuracy) * 2  # Double weight for final accuracy
        
        return reward
    
    def _drop_penalty(self, balls_in_plunger, task_phase):
        """Penalty for dropping balls unexpectedly"""
        # This would need more sophisticated tracking of expected vs actual ball count
        # For now, simple heuristic
        if task_phase == 2 and balls_in_plunger == 0:  # Transfer phase but no balls
            return 1.0
        return 0.0
    
    def _miss_penalty(self, obs, action, task_phase):
        """Penalty for missing target container (spilling)"""
        if task_phase != 3:  # Only care during dispensing
            return 0.0
        
        # Check if pipette is positioned over target well during dispensing
        pipette_pos = action[:3]  # x, y, z position
        target_pos = obs[7:10]    # target well position
        
        # Simple distance check (in real implementation, would need more sophisticated collision detection)
        distance = np.linalg.norm(pipette_pos[:2] - target_pos[:2])  # x, y distance only
        
        if distance > 0.1:  # Threshold for "over target"
            return distance * 10  # Scale penalty by distance
        
        return 0.0
    
    def _jerk_penalty(self, action):
        """Penalty for jerky movements (rapid acceleration)"""
        if self.prev_action is None:
            return 0.0
        
        # Compute acceleration (change in velocity approximated by change in position)
        acceleration = np.abs(action - self.prev_action)
        
        # Penalize high accelerations
        jerk = np.sum(acceleration**2)
        
        # Threshold to avoid penalizing normal movements
        if jerk > 0.5:
            return jerk
        
        return 0.0
    
    def reset(self):
        """Reset any internal state"""
        self.prev_action = None


def compute_reward(obs, action, info=None, done=False, reward_fn=None):
    """
    Standalone reward function that can be called from environment
    
    Args:
        obs: Current observation (14D numpy array)
        action: Action taken (6D numpy array) 
        info: Optional info dict from environment
        done: Whether episode is done
        reward_fn: Optional PipettingRewardFunction instance
    
    Returns:
        float: Computed reward
    """
    if reward_fn is None:
        reward_fn = PipettingRewardFunction()
    
    return reward_fn.compute_reward(obs, action, info, done)


# Default reward function instance
DEFAULT_REWARD_FUNCTION = PipettingRewardFunction()