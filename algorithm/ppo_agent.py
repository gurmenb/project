import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


class Actor(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_dim, std=0.1):
        super().__init__()
        self.std = std

        self.policy = nn.Sequential(
            nn.Linear(obs_shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_shape[0])
        )
        self.apply(utils.weight_init)

    def forward(self, obs):
        mu = self.policy(obs)
        mu = torch.tanh(mu)  # restrict outputs to [-1,1]
        std = torch.ones_like(mu) * self.std
        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, obs_shape, hidden_dim):
        super().__init__()
        self.value_net = nn.Sequential(
            nn.Linear(obs_shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(utils.weight_init)

    def forward(self, obs):
        return self.value_net(obs)


class PPOAgent:
    def __init__(self, obs_shape, action_shape, device, lr, hidden_dim,
                 clip_ratio=0.2, stddev_clip=0.3, use_tb=True):
        self.device = device
        self.clip_ratio = clip_ratio
        self.stddev_clip = stddev_clip
        self.use_tb = use_tb

        # Build actor & critic networks
        self.actor = Actor(obs_shape, action_shape, hidden_dim).to(device)
        self.critic = Critic(obs_shape, hidden_dim).to(device)

        # Optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.train()  # set networks to training mode

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, eval_mode=False):
        obs_tensor = torch.as_tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
        dist = self.actor(obs_tensor)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=self.stddev_clip)
        return action.cpu().numpy()[0]  # shape (4,)

    def get_value(self, obs):
        obs_tensor = torch.as_tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            value = self.critic(obs_tensor)
        return value.cpu().numpy()[0, 0]

    def update_critic(self, batch_data):
        """
        Update critic network using MSE between predicted values and returns.
        """
        metrics = {}
        obs = torch.as_tensor(batch_data['observations'], device=self.device, dtype=torch.float32)
        returns = torch.as_tensor(batch_data['returns'], device=self.device, dtype=torch.float32)

        current_values = self.critic(obs).squeeze(-1)
        critic_loss = F.mse_loss(current_values, returns)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        metrics['critic_loss'] = critic_loss.item()
        metrics['critic_target'] = returns.mean().item()
        return metrics

    def update_actor(self, batch_data):
        """
        Update actor network using the PPO clipped surrogate objective.
        """
        metrics = {}
        obs = torch.as_tensor(batch_data['observations'], device=self.device, dtype=torch.float32)
        actions = torch.as_tensor(batch_data['actions'], device=self.device, dtype=torch.float32)
        advantages = torch.as_tensor(batch_data['advantages'], device=self.device, dtype=torch.float32)
        old_log_probs = torch.as_tensor(batch_data['old_log_probs'], device=self.device, dtype=torch.float32)

        dist = self.actor(obs)
        new_log_probs = dist.log_prob(actions).sum(-1)

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        metrics['actor_loss'] = actor_loss.item()
        metrics['policy_ratio'] = ratio.mean().item()
        return metrics

    def evaluate_actions(self, obs, actions):
        """
        Compute log_probs and value estimates for given obs/actions.
        """
        dist = self.actor(obs)
        log_probs = dist.log_prob(actions).sum(-1)
        values = self.critic(obs).squeeze(-1)
        return log_probs, values

    def ppo_update(self, buffer_data, ppo_epochs=10, batch_size=64):
        """
        Legacy combined update across epochs & minibatches.
        (Not used by default since training_pipetting.py calls update_actor/critic directly.)
        """
        all_metrics = {'actor_loss': 0.0, 'critic_loss': 0.0, 'policy_ratio': 0.0, 'critic_target': 0.0}

        obs = torch.as_tensor(buffer_data['observations'], device=self.device, dtype=torch.float32)
        actions = torch.as_tensor(buffer_data['actions'], device=self.device, dtype=torch.float32)
        returns = torch.as_tensor(buffer_data['returns'], device=self.device, dtype=torch.float32)
        advantages = torch.as_tensor(buffer_data['advantages'], device=self.device, dtype=torch.float32)
        old_log_probs = torch.as_tensor(buffer_data['old_log_probs'], device=self.device, dtype=torch.float32)

        dataset_size = len(obs)
        num_updates = ppo_epochs * ((dataset_size + batch_size - 1) // batch_size)
        update_count = 0

        for _ in range(ppo_epochs):
            indices = torch.randperm(dataset_size)
            for start in range(0, dataset_size, batch_size):
                end = min(start + batch_size, dataset_size)
                batch_indices = indices[start:end]
                batch = {
                    'observations': obs[batch_indices],
                    'actions': actions[batch_indices],
                    'returns': returns[batch_indices],
                    'advantages': advantages[batch_indices],
                    'old_log_probs': old_log_probs[batch_indices]
                }
                c_metrics = self.update_critic(batch)
                a_metrics = self.update_actor(batch)
                for key in all_metrics:
                    if key in c_metrics:
                        all_metrics[key] += c_metrics[key]
                    if key in a_metrics:
                        all_metrics[key] += a_metrics[key]
                update_count += 1

        for key in all_metrics:
            all_metrics[key] /= max(1, update_count)
        return all_metrics

    def bc(self, replay_iter):
        """Legacy behavior cloning (not used in standard training)."""
        metrics = {}
        batch = next(replay_iter)
        obs, action, _, _, _ = utils.to_torch(batch, self.device)
        dist = self.actor(obs)
        loss = -dist.log_prob(action).sum(-1, keepdim=True).mean()

        self.actor_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()

        metrics['actor_loss'] = loss.item()
        return metrics
