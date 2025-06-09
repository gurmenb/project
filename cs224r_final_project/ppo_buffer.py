import numpy as np
import torch


class PPOBuffer:
    """
    Buffer for storing trajectories experienced by PPO agent.
    Handles experience collection and advantage computation.
    WITH ADVANTAGE NORMALIZATION (Your original 83% success version)
    """
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        # Buffers for storing trajectory data
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        
        # Hyperparameters
        self.gamma = gamma  # Discount factor
        self.lam = lam      # GAE lambda parameter
        
        # Buffer state
        self.ptr = 0                # Current position in buffer
        self.path_start_idx = 0     # Start of current trajectory
        self.max_size = size
    
    def store(self, obs, act, rew, val, logp):
        """
        Store one step of interaction.
        
        Args:
            obs: observation (14D for pipette env)
            act: action taken (4D for pipette env)
            rew: reward received
            val: value estimate from critic
            logp: log probability of action
        """
        assert self.ptr < self.max_size, "Buffer overflow!"
        
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1
    
    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory or when buffer fills up.
        Computes GAE advantages and rewards-to-go for the trajectory.
        
        Args:
            last_val: Value estimate for final state (0 if terminal)
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # Compute GAE (Generalized Advantage Estimation)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self._discount_cumsum(deltas, self.gamma * self.lam)
        
        # Compute rewards-to-go (targets for value function)
        self.ret_buf[path_slice] = self._discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr
    
    def get(self):
        """
        Get all data from buffer and prepare for training.
        Buffer must be full before calling this.
        
        Returns:
            dict: All buffer data as PyTorch tensors WITH normalized advantages
        """
        assert self.ptr == self.max_size, "Buffer not full!"
        
        # Reset buffer pointers
        self.ptr, self.path_start_idx = 0, 0
        
        # ✅ NORMALIZE ADVANTAGES (Your original working approach!)
        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        
        # Package data for training
        data = dict(
            obs=self.obs_buf,
            act=self.act_buf, 
            ret=self.ret_buf,
            adv=self.adv_buf,  # Normalized advantages
            logp=self.logp_buf
        )
        
        # Convert to PyTorch tensors
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}
    
    def _discount_cumsum(self, x, discount):
        """
        Helper function to compute discounted cumulative sums.
        Used for computing advantages and returns.
        """
        return np.array([
            np.sum(discount**np.arange(len(x)-i) * x[i:]) 
            for i in range(len(x))
        ])
    
    def is_full(self):
        """Check if buffer is full."""
        return self.ptr == self.max_size
    
    def size(self):
        """Current buffer size."""
        return self.ptr