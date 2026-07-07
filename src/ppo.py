import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def layer_init(layer, gain=np.sqrt(2), bias=0.0):
    """Orthogonal weight init + constant bias — the canonical PPO scheme
    (Huang et al. 2022, "The 37 Implementation Details of PPO").
    gain=sqrt(2) for hidden/conv layers, 0.01 for the policy head (near-uniform
    initial policy), 1.0 for the value head. Not a data-dependent prior — it
    only conditions activation/gradient variance at initialization."""
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    """Actor-Critic with shared MLP trunk for vector observations."""

    def __init__(self, num_states, num_actions, hidden_dim):
        super().__init__()
        self.shared = nn.Sequential(
            layer_init(nn.Linear(num_states, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
        )
        self.actor  = layer_init(nn.Linear(hidden_dim, num_actions), gain=0.01)
        self.critic = layer_init(nn.Linear(hidden_dim, 1), gain=1.0)

    def forward(self, x):
        features = self.shared(x)
        return self.actor(features), self.critic(features).squeeze(-1)

    def get_action(self, x, deterministic=False):
        logits, value = self(x)
        dist = Categorical(logits=logits)
        action = logits.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate_actions(self, x, actions):
        logits, values = self(x)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values


class CNNActorCritic(nn.Module):
    """Actor-Critic with shared CNN backbone for pixel observations.
    Uses the same conv stack as CNNDQN so Grad-CAM works identically."""

    def __init__(self, num_states, num_actions, hidden_dim, obs_size=80):
        super().__init__()
        self.conv1 = layer_init(nn.Conv2d(num_states, 32, kernel_size=8, stride=4, padding=2))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))

        with torch.no_grad():
            dummy = torch.zeros(1, num_states, obs_size, obs_size)
            flat_size = self._conv_features(dummy).shape[1]

        self.actor = nn.Sequential(
            layer_init(nn.Linear(flat_size, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, num_actions), gain=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(flat_size, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 1), gain=1.0),
        )

    def _conv_features(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.flatten(1)

    def forward(self, x):
        features = self._conv_features(x)
        return self.actor(features), self.critic(features).squeeze(-1)

    def get_action(self, x, deterministic=False):
        logits, value = self(x)
        dist = Categorical(logits=logits)
        action = logits.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate_actions(self, x, actions):
        logits, values = self(x)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values


PPO_NETWORK_REGISTRY = {
    "ppo":     ActorCritic,
    "ppo_cnn": CNNActorCritic,
}


def ppo_optimize(actor_critic, optimizer, batch, clip_eps, vf_coef, ent_coef, max_grad_norm):
    """Single PPO gradient update step.

    Loss = L_clip + vf_coef * L_vf + ent_coef * L_entropy
    Returns (total_loss, policy_loss, value_loss, mean_entropy) as plain floats.
    """
    states, actions, old_log_probs, advantages, returns = batch

    log_probs, entropy, values = actor_critic.evaluate_actions(states, actions)

    ratio = (log_probs - old_log_probs).exp()
    policy_loss = -torch.min(
        ratio * advantages,
        ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages,
    ).mean()

    value_loss  = 0.5 * (values - returns).pow(2).mean()
    entropy_loss = -entropy.mean()

    loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(actor_critic.parameters(), max_grad_norm)
    optimizer.step()

    return loss.item(), policy_loss.item(), value_loss.item(), entropy.mean().item()
