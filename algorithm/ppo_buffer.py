import numpy as np
from typing import NamedTuple
from collections import defaultdict, deque
from dm_env import specs  # Used only for make_ppo_data_specs


class PPOBufferSimple:
    """
    Simple PPO Buffer for trajectory collection and advantage computation.
    """

    def __init__(self, obs_dim, act_dim, max_size=2048, gamma=0.99, lam=0.95):
        # Storage arrays
        self.obs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(max_size, dtype=np.float32)
        self.val_buf = np.zeros(max_size, dtype=np.float32)
        self.logp_buf = np.zeros(max_size, dtype=np.float32)
        self.ret_buf = np.zeros(max_size, dtype=np.float32)
        self.adv_buf = np.zeros(max_size, dtype=np.float32)

        # Params
        self.gamma = gamma
        self.lam = lam
        self.max_size = max_size

        # Pointers
        self.ptr = 0
        self.path_start_idx = 0

    def store(self, obs, act, rew, val, logp):
        """
        Store one timestep of agent-environment interaction.
        Returns False if buffer is already full.
        """
        if self.ptr >= self.max_size:
            return False

        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp

        self.ptr += 1
        return True

    def finish_path(self, last_val=0.0):
        """
        Call this at the end of a trajectory (or when the buffer fills mid-trajectory).
        Computes GAE advantages and rewards-to-go (returns).
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # GAE-Lambda advantage
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self._discount_cumsum(deltas, self.gamma * self.lam)

        # Compute returns (rewards-to-go)
        self.ret_buf[path_slice] = self._discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Retrieve all data from the buffer (normalizing advantages).
        Returns None if nothing is in the buffer.
        """
        if self.ptr == 0:
            return None

        data_slice = slice(0, self.ptr)
        adv_mean, adv_std = np.mean(self.adv_buf[data_slice]), np.std(self.adv_buf[data_slice])
        self.adv_buf[data_slice] = (self.adv_buf[data_slice] - adv_mean) / (adv_std + 1e-8)

        return {
            'observations': self.obs_buf[data_slice].copy(),
            'actions': self.act_buf[data_slice].copy(),
            'returns': self.ret_buf[data_slice].copy(),
            'advantages': self.adv_buf[data_slice].copy(),
            'old_log_probs': self.logp_buf[data_slice].copy()
        }

    def clear(self):
        """Clear the buffer for the next collection phase."""
        self.ptr = 0
        self.path_start_idx = 0

    def is_full(self):
        """Returns True if the buffer is full (ptr ≥ max_size)."""
        return self.ptr >= self.max_size

    def _discount_cumsum(self, x, discount):
        """Compute discounted cumulative sums of vector x."""
        return np.array([np.sum(discount ** np.arange(len(x) - i) * x[i:]) for i in range(len(x))])


class PPODataLoader:
    """Data loader for PPO mini-batch training"""

    def __init__(self, data, batch_size, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        dataset_size = len(self.data['observations'])
        indices = np.arange(dataset_size)
        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, dataset_size, self.batch_size):
            end = min(start + self.batch_size, dataset_size)
            batch_indices = indices[start:end]

            batch = {}
            for key, vals in self.data.items():
                batch[key] = vals[batch_indices]
            yield batch


def make_ppo_data_specs(obs_shape, action_shape):
    """Create dm_env specs for PPO (unused unless you need them)."""
    return (
        specs.Array(shape=obs_shape, dtype=np.float32, name='observation'),
        specs.Array(shape=action_shape, dtype=np.float32, name='action'),
        specs.Array(shape=(1,), dtype=np.float32, name='reward'),
        specs.Array(shape=(1,), dtype=np.float32, name='value'),
        specs.Array(shape=(1,), dtype=np.float32, name='log_prob'),
        specs.Array(shape=(1,), dtype=np.bool_,  name='done'),
    )
