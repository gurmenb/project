import numpy as np

class PipettingRewardFunction:
    """
    Comprehensive Pipetting Reward Function
    
    Observation Space (14-dimensional):
    0: liquid_in_plunger, 1: balls_in_plunger, 2: source_well_amount, 3: target_well_amount
    4-6: source_well_position (x,y,z), 7-9: target_well_position (x,y,z)
    10: aspiration_pressure, 11: dispersion_pressure
    12: task_phase (0=approach, 1=aspirate, 2=transfer, 3=dispense), 13: submerged
    
    Action Space (6-dimensional):
    0-2: pipette position (x,y,z), 3: plunger z position, 4: plunger z force, 5: plunger z speed
    """
    
    def __init__(self, 
                 w_volume=10.0,      # Volume accuracy weight
                 w_completion=50.0,  # Completion bonus weight
                 w_time=0.01,        # Time efficiency weight
                 w_collision=2.0,    # Collision penalty weight
                 w_drop=5.0,         # Drop penalty weight
                 w_contamination=3.0, # Contamination penalty weight
                 w_miss=2.0,         # Miss penalty weight
                 w_jerk=0.1,         # Jerk penalty weight
                 target_volume=20):  # Target number of balls to transfer
        
        self.weights = {
            'volume': w_volume,
            'completion': w_completion,
            'time': w_time,
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
        
        Args:
            obs: 14D observation array
            action: 6D action array
            info: Optional info dict from environment
            done: Whether episode is done
            
        Returns:
            float: Total reward for this step
        """
        reward = 0.0
        
        # Extract relevant observations
        balls_in_plunger = obs[1]
        source_amount = obs[2]
        target_amount = obs[3]
        task_phase = int(obs[12])
        
        # 1. Volume accuracy rewards
        volume_reward = self._volume_accuracy_reward(balls_in_plunger, target_amount, task_phase, done)
        reward += self.weights['volume'] * volume_reward
        
        # 2. Time efficiency penalty
        reward -= self.weights['time']
        
        # 3. Completion bonus
        if done:
            success_threshold = 5
            if abs(target_amount - self.target_volume) <= success_threshold:
                reward += self.weights['completion']
                print(f"SUCCESS! Target: {self.target_volume}, Actual: {target_amount}")
        
        # 4. Progressive rewards for good behavior
        progress_reward = self._progress_reward(task_phase, balls_in_plunger, target_amount)
        reward += progress_reward
        
        # 5. Collision penalty
        if info and info.get('collision', False):
            reward -= self.weights['collision']
        
        # 6. Drop penalty
        drop_penalty = self._drop_penalty(balls_in_plunger, task_phase)
        reward -= self.weights['drop'] * drop_penalty
        
        # 7. Miss penalty
        miss_penalty = self._miss_penalty(obs, action, task_phase)
        reward -= self.weights['miss'] * miss_penalty
        
        # 8. Jerk penalty
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
            if balls_in_plunger > 0:
                reward += 1.0
                if balls_in_plunger >= self.target_volume * 0.8:
                    reward += 2.0
        
        # Final check: Correct final volume in target
        if done:
            if target_amount > 0:
                reward += 5.0
                accuracy = 1.0 - abs(target_amount - self.target_volume) / max(self.target_volume, 1)
                reward += max(0, accuracy) * 10.0
        
        return reward
    
    def _progress_reward(self, task_phase, balls_in_plunger, target_amount):
        """Reward for making progress through the task phases"""
        reward = 0.0
        
        # Small rewards for reaching each phase
        if task_phase >= 1:
            reward += 0.5
        if task_phase >= 2:
            reward += 0.5
        if task_phase >= 3:
            reward += 0.5
        
        # Reward for having balls in plunger during transfer
        if task_phase == 2 and balls_in_plunger > 0:
            reward += 1.0
        
        # Reward for having dispensed some balls
        if target_amount > 0:
            reward += 2.0
        
        return reward
    
    def _drop_penalty(self, balls_in_plunger, task_phase):
        """Penalty for dropping balls unexpectedly"""
        if task_phase == 2 and balls_in_plunger == 0:
            return 0.5
        return 0.0
    
    def _miss_penalty(self, obs, action, task_phase):
        """Penalty for missing target container (spilling)"""
        if task_phase != 3:
            return 0.0
        
        pipette_pos = action[:3]
        target_pos = obs[7:10]
        
        distance = np.linalg.norm(pipette_pos[:2] - target_pos[:2])
        
        if distance > 0.2:
            return distance * 2
        
        return 0.0
    
    def _jerk_penalty(self, action):
        """Penalty for jerky movements (rapid acceleration)"""
        if self.prev_action is None:
            return 0.0
        
        acceleration = np.abs(action - self.prev_action)
        jerk = np.sum(acceleration**2)
        
        if jerk > 1.0:
            return jerk * 0.1
        
        return 0.0
    
    def reset(self):
        """Reset any internal state"""
        self.prev_action = None


def make_pipetting_reward_function(target_volume=20, **kwargs):
    """Factory function to create reward function with custom parameters"""
    return PipettingRewardFunction(target_volume=target_volume, **kwargs)


def compute_reward(obs, action, info=None, done=False, reward_fn=None):
    """Standalone reward computation function"""
    if reward_fn is None:
        reward_fn = PipettingRewardFunction()
    
    return reward_fn.compute_reward(obs, action, info, done)