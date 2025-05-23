import numpy as np
import torch
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RewardWeights:
    """Weights for different reward components."""
    # Positive rewards
    volume_accuracy: float = 10.0      # w1: Reward for transferring correct number of balls
    time_efficiency: float = 0.01      # w2: Small reward for faster completion
    completion_bonus: float = 50.0     # w3: Big bonus for successful task completion
    
    # Negative rewards (penalties)
    collision_penalty: float = 5.0     # w4: Penalty for hitting container walls
    drop_penalty: float = 10.0         # w5: Penalty for dropping balls (replaces splash)
    contamination_penalty: float = 3.0 # w6: Penalty for touching container sides with gripper
    miss_penalty: float = 5.0          # w7: Penalty for missing target container
    
    # Additional penalties for stability
    jerk_penalty: float = 0.1          # Penalty for jerky movements
    height_penalty: float = 0.5        # Penalty for going too high (inefficient)


class BallTransferReward:
    """Reward function for ball transfer task (simplified pipetting)."""
    
    def __init__(self, weights: Optional[RewardWeights] = None):
        self.weights = weights or RewardWeights()
        self.episode_steps = 0
        self.prev_action = None
        
    def reset(self):
        """Reset episode-specific counters."""
        self.episode_steps = 0
        self.prev_action = None
    
    def compute_reward(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        next_obs: Dict[str, np.ndarray],
        info: Dict[str, any]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute reward for ball transfer task.
        
        Args:
            obs: Current observation dictionary
            action: Action taken
            next_obs: Next observation dictionary
            info: Additional information from environment
            
        Returns:
            total_reward: Scalar reward
            reward_components: Dictionary of individual reward components for logging
        """
        reward_components = {}
        
        # 1. Volume Accuracy Reward (number of balls transferred correctly)
        balls_in_gripper = info.get('balls_in_gripper', 0)
        target_balls = info.get('target_balls', 0)
        balls_transferred = info.get('balls_transferred', 0)
        
        # Reward for picking up balls when needed
        if info.get('task_phase') == 'pick' and balls_in_gripper > 0:
            volume_reward = (balls_in_gripper / target_balls) * self.weights.volume_accuracy
        # Reward for successful transfer
        elif info.get('task_phase') == 'place' and balls_transferred > 0:
            accuracy = 1.0 - abs(balls_transferred - target_balls) / max(target_balls, 1)
            volume_reward = accuracy * self.weights.volume_accuracy
        else:
            volume_reward = 0.0
        
        reward_components['volume_accuracy'] = volume_reward
        
        # 2. Time Efficiency Penalty (negative reward for each step)
        self.episode_steps += 1
        time_penalty = -self.weights.time_efficiency * self.episode_steps
        reward_components['time_efficiency'] = time_penalty
        
        # 3. Completion Bonus
        if info.get('task_completed', False):
            completion_reward = self.weights.completion_bonus
            # Extra bonus for perfect accuracy
            if balls_transferred == target_balls:
                completion_reward *= 1.5
        else:
            completion_reward = 0.0
        
        reward_components['completion_bonus'] = completion_reward
        
        # 4. Collision Penalty
        if info.get('collision', False):
            collision_penalty = -self.weights.collision_penalty
        else:
            collision_penalty = 0.0
        
        reward_components['collision_penalty'] = collision_penalty
        
        # 5. Drop Penalty (replaces splash penalty)
        balls_dropped = info.get('balls_dropped', 0)
        if balls_dropped > 0:
            drop_penalty = -self.weights.drop_penalty * balls_dropped
        else:
            drop_penalty = 0.0
        
        reward_components['drop_penalty'] = drop_penalty
        
        # 6. Contamination Penalty (gripper touching container sides)
        if info.get('gripper_contamination', False):
            contamination_penalty = -self.weights.contamination_penalty
        else:
            contamination_penalty = 0.0
        
        reward_components['contamination_penalty'] = contamination_penalty
        
        # 7. Miss Penalty (balls falling outside target container)
        balls_missed = info.get('balls_missed', 0)
        if balls_missed > 0:
            miss_penalty = -self.weights.miss_penalty * balls_missed
        else:
            miss_penalty = 0.0
        
        reward_components['miss_penalty'] = miss_penalty
        
        # 8. Movement Quality Penalties
        # Jerk penalty for smooth movements
        if self.prev_action is not None:
            action_diff = np.linalg.norm(action[:3] - self.prev_action[:3])  # Position changes
            jerk_penalty = -self.weights.jerk_penalty * action_diff
        else:
            jerk_penalty = 0.0
        
        reward_components['jerk_penalty'] = jerk_penalty
        self.prev_action = action.copy()
        
        # Height penalty (encourage efficient paths)
        gripper_height = obs.get('gripper_height', 0.2)
        if gripper_height > 0.3 and info.get('task_phase') == 'transfer':
            height_penalty = -self.weights.height_penalty * (gripper_height - 0.3)
        else:
            height_penalty = 0.0
        
        reward_components['height_penalty'] = height_penalty
        
        # Calculate total reward
        total_reward = sum(reward_components.values())
        
        return total_reward, reward_components
    
    def compute_shaped_reward(
        self,
        obs: Dict[str, np.ndarray],
        info: Dict[str, any]
    ) -> float:
        """
        Compute additional shaped rewards to help learning.
        These are distance-based rewards to guide the agent.
        """
        shaped_reward = 0.0
        
        # Distance to source container when picking
        if info.get('task_phase') == 'approach_source':
            dist_to_source = info.get('distance_to_source', 1.0)
            shaped_reward += 0.1 * (1.0 - min(dist_to_source, 1.0))
        
        # Distance to target when placing
        elif info.get('task_phase') == 'approach_target':
            dist_to_target = info.get('distance_to_target', 1.0)
            shaped_reward += 0.1 * (1.0 - min(dist_to_target, 1.0))
        
        # Height reward when carrying balls
        elif info.get('task_phase') == 'transfer' and info.get('balls_in_gripper', 0) > 0:
            # Encourage maintaining safe height
            gripper_height = obs.get('gripper_height', 0.2)
            if 0.15 < gripper_height < 0.25:
                shaped_reward += 0.05
        
        return shaped_reward


def create_reward_function(config: Dict = None) -> BallTransferReward:
    """Factory function to create reward function with custom weights."""
    if config:
        weights = RewardWeights(**config)
    else:
        weights = RewardWeights()
    
    return BallTransferReward(weights)


# Example usage for testing
if __name__ == "__main__":
    # Create reward function
    reward_fn = create_reward_function()
    
    # Mock data for testing
    obs = {
        'gripper_height': 0.25,
        'gripper_pos': np.array([0.1, 0.2, 0.25])
    }
    
    action = np.array([0.01, 0.0, -0.02, 0.0, 0.0, 0.5, 0.8, 0.0])
    
    next_obs = {
        'gripper_height': 0.23,
        'gripper_pos': np.array([0.11, 0.2, 0.23])
    }
    
    info = {
        'task_phase': 'pick',
        'balls_in_gripper': 3,
        'target_balls': 5,
        'balls_transferred': 0,
        'collision': False,
        'balls_dropped': 0,
        'task_completed': False
    }
    
    # Compute reward
    total_reward, components = reward_fn.compute_reward(obs, action, next_obs, info)
    
    print(f"Total reward: {total_reward:.3f}")
    print("Reward components:")
    for name, value in components.items():
        print(f"  {name}: {value:.3f}")