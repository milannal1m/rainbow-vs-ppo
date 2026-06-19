import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQN, self).__init__()
        self.fc1    = nn.Linear(state_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.output(x)


class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_value      = nn.Linear(hidden_dim, hidden_dim)
        self.value         = nn.Linear(hidden_dim, 1)
        self.fc_advantages = nn.Linear(hidden_dim, hidden_dim)
        self.advantages    = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        V = self.value(F.relu(self.fc_value(x)))
        A = self.advantages(F.relu(self.fc_advantages(x)))
        return V + A - torch.mean(A, dim=1, keepdim=True)


class CNNDQN(nn.Module):
    # Based on https://github.com/yenchenlin/DeepLearningFlappyBird/blob/master/deep_q_network.py
    def __init__(self, in_channels, action_dim, hidden_dim=512, obs_size=80):
        super(CNNDQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, obs_size, obs_size)
            dummy = F.relu(self.conv1(dummy))
            dummy = self.pool1(dummy)
            dummy = F.relu(self.conv2(dummy))
            dummy = F.relu(self.conv3(dummy))
            flat_size = dummy.view(1, -1).shape[1]
        self.fc1    = nn.Linear(flat_size, hidden_dim)
        self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.output(x)


# ---------------------------------------------------------------------------
# Rainbow extensions
# ---------------------------------------------------------------------------

class NoisyLinear(nn.Module):
    """Noisy linear layer with factorised Gaussian noise (Fortunato et al. 2017)."""
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu    = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        bound = 1.0 / math.sqrt(in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(sigma_init / math.sqrt(in_features))
        self.bias_mu.data.uniform_(-bound, bound)
        self.bias_sigma.data.fill_(sigma_init / math.sqrt(out_features))
        self.sample_noise()

    @staticmethod
    def _f(x):
        return x.sign().mul_(x.abs().sqrt_())

    def sample_noise(self):
        eps_in  = self._f(torch.randn(self.in_features,  device=self.weight_mu.device))
        eps_out = self._f(torch.randn(self.out_features, device=self.weight_mu.device))
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        else:
            w, b = self.weight_mu, self.bias_mu
        return F.linear(x, w, b)


class RainbowCNNDQN(nn.Module):
    """CNN DQN with modular Rainbow extensions.

    Supported flags (all default False / off):
      use_noisy         — replace fc layers with NoisyLinear (Noisy Nets)
      use_dueling       — value + advantage streams (Dueling architecture)
      use_distributional— categorical distribution over n_atoms return atoms (C51)

    Double DQN and PER are handled at the optimize() / agent level, not here.
    n-step returns are also agent-level.

    Other model types (DQN, DuelingDQN) can be extended similarly in the future;
    for now only RainbowCNNDQN implements these flags.

    Following Rainbow DQN paper (Hessel et al. 2017)
    """
    def __init__(self, in_channels, action_dim, hidden_dim=512, obs_size=80,
                 use_noisy=False, sigma_init=0.5,
                 use_dueling=False,
                 use_distributional=False, n_atoms=51, v_min=-10.0, v_max=10.0):
        super().__init__()
        self.action_dim         = action_dim
        self.use_noisy          = use_noisy
        self.use_dueling        = use_dueling
        self.use_distributional = use_distributional
        self.n_atoms            = n_atoms
        self.v_min              = v_min
        self.v_max              = v_max

        # Shared CNN encoder (same architecture as CNNDQN)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, obs_size, obs_size)
            dummy = F.relu(self.conv1(dummy)); dummy = self.pool1(dummy)
            dummy = F.relu(self.conv2(dummy)); dummy = F.relu(self.conv3(dummy))
            flat_size = dummy.view(1, -1).shape[1]

        def _lin(in_f, out_f):
            return NoisyLinear(in_f, out_f, sigma_init) if use_noisy else nn.Linear(in_f, out_f)

        self.fc1 = _lin(flat_size, hidden_dim)

        if use_distributional:
            self.register_buffer('support', torch.linspace(v_min, v_max, n_atoms))

        if use_dueling:
            self.fc_value = _lin(hidden_dim, hidden_dim)
            self.value    = _lin(hidden_dim, n_atoms if use_distributional else 1)
            self.fc_adv   = _lin(hidden_dim, hidden_dim)
            self.adv      = _lin(hidden_dim, action_dim * n_atoms if use_distributional else action_dim)
        else:
            self.output = _lin(hidden_dim, action_dim * n_atoms if use_distributional else action_dim)

    def _conv_features(self, x):
        x = F.relu(self.conv1(x)); x = self.pool1(x)
        x = F.relu(self.conv2(x)); x = F.relu(self.conv3(x))
        return x.view(x.size(0), -1)

    def _logits(self, x):
        """Returns logits: [B, action_dim, n_atoms] if distributional, else [B, action_dim]."""
        x = F.relu(self.fc1(self._conv_features(x)))
        if self.use_dueling:
            v = self.value(F.relu(self.fc_value(x)))
            a = self.adv(F.relu(self.fc_adv(x)))
            if self.use_distributional:
                v = v.view(-1, 1, self.n_atoms)
                a = a.view(-1, self.action_dim, self.n_atoms)
            return v + a - a.mean(dim=1, keepdim=True)
        else:
            out = self.output(x)
            if self.use_distributional:
                return out.view(-1, self.action_dim, self.n_atoms)
            return out

    def forward(self, x):
        """Returns Q-values [B, action_dim] for action selection."""
        logits = self._logits(x)
        if self.use_distributional:
            return (F.softmax(logits, dim=-1) * self.support).sum(-1)
        return logits

    def dist(self, x):
        """Returns (probs, log_probs) each [B, action_dim, n_atoms].
           Only valid when use_distributional=True."""
        logits    = self._logits(x)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.exp(), log_probs

    def sample_noise(self):
        """Resample factorised Gaussian noise for all NoisyLinear layers."""
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.sample_noise()


def mean_sigma(model):
    """Mean absolute weight_sigma across all NoisyLinear layers. Returns 0.0 for non-noisy models."""
    sigmas = [m.weight_sigma.abs().mean().item()
              for m in model.modules() if isinstance(m, NoisyLinear)]
    return float(sum(sigmas) / len(sigmas)) if sigmas else 0.0


NETWORK_REGISTRY = {
    "dqn":             DQN,
    "dueling_dqn":     DuelingDQN,
    "cnn_dqn":         CNNDQN,
    "rainbow_cnn_dqn": RainbowCNNDQN,
}


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

def _standard_loss(states, actions, new_states, rewards, terminations,
                   policy_dqn, target_dqn, discount_factor_g, enable_double_dqn):
    with torch.no_grad():
        if enable_double_dqn:
            best_a   = policy_dqn(new_states).max(dim=1)[1]
            target_q = rewards + (1 - terminations.float()) * discount_factor_g * \
                       target_dqn(new_states).gather(1, best_a.unsqueeze(1)).squeeze(1)
        else:
            target_q = rewards + (1 - terminations.float()) * discount_factor_g * \
                       target_dqn(new_states).max(dim=1)[0]

    current_q     = policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    element_losses = F.mse_loss(current_q, target_q, reduction='none')
    td_errors      = (current_q - target_q).detach().abs().cpu().numpy()
    return element_losses, td_errors


def _distributional_loss(states, actions, new_states, rewards, terminations,
                         policy_dqn, target_dqn, discount_factor_g, enable_double_dqn):
    """Cross-entropy loss for C51 distributional RL (Bellemare et al. 2017)."""
    B       = states.shape[0]
    n_atoms = policy_dqn.n_atoms
    support = policy_dqn.support
    delta_z = (policy_dqn.v_max - policy_dqn.v_min) / (n_atoms - 1)

    with torch.no_grad():
        # Bootstrap action: online net (Double) or target net
        next_q       = policy_dqn(new_states) if enable_double_dqn else target_dqn(new_states)
        next_actions = next_q.argmax(dim=1)                           # [B]

        # Target distribution from target network
        next_probs, _ = target_dqn.dist(new_states)                  # [B, A, n_atoms]
        next_p        = next_probs[range(B), next_actions]           # [B, n_atoms]

        # Project onto support: Tz_j = clip(r + gamma * z_j, v_min, v_max)
        Tz = rewards.unsqueeze(1) + \
             (1 - terminations.float().unsqueeze(1)) * discount_factor_g * support
        Tz = Tz.clamp(policy_dqn.v_min, policy_dqn.v_max)
        b  = (Tz - policy_dqn.v_min) / delta_z                      # [B, n_atoms]
        l  = b.floor().long().clamp(0, n_atoms - 1)
        u  = b.ceil().long().clamp(0, n_atoms - 1)

        target_p = torch.zeros(B, n_atoms, device=states.device)
        offset   = torch.arange(B, device=states.device).unsqueeze(1) * n_atoms
        target_p.view(-1).scatter_add_(0, (l + offset).view(-1),
                                       (next_p * (u.float() - b)).view(-1))
        target_p.view(-1).scatter_add_(0, (u + offset).view(-1),
                                       (next_p * (b - l.float())).view(-1))

    # Log-probs for selected actions
    _, log_p_all  = policy_dqn.dist(states)                          # [B, A, n_atoms]
    log_p         = log_p_all[range(B), actions]                     # [B, n_atoms]
    element_losses = -(target_p * log_p).sum(dim=1)                  # [B]

    with torch.no_grad():
        q_pred    = (log_p.exp() * support).sum(dim=1)
        q_tgt     = (target_p   * support).sum(dim=1)
        td_errors = (q_pred - q_tgt).abs().cpu().numpy()

    return element_losses, td_errors


def optimize(batch, policy_dqn, target_dqn, optimizer, discount_factor_g,
             enable_double_dqn, device, weights=None):
    """
    Args:
        weights: Optional IS weights tensor [batch_size] from PER (or None).
    Returns:
        (loss_scalar, mean_max_q, td_errors_numpy)
    """
    states, actions, new_states, rewards, terminations = zip(*batch)
    states       = torch.stack(states)
    actions      = torch.stack(actions)
    new_states   = torch.stack(new_states)
    rewards      = torch.stack(rewards)
    terminations = torch.tensor(terminations, dtype=torch.bool).to(device)

    if getattr(policy_dqn, 'use_distributional', False):
        element_losses, td_errors = _distributional_loss(
            states, actions, new_states, rewards, terminations,
            policy_dqn, target_dqn, discount_factor_g, enable_double_dqn)
    else:
        element_losses, td_errors = _standard_loss(
            states, actions, new_states, rewards, terminations,
            policy_dqn, target_dqn, discount_factor_g, enable_double_dqn)

    with torch.no_grad():
        mean_max_q = policy_dqn(states).max(dim=1)[0].mean().item()

    loss = (element_losses * weights).mean() if weights is not None else element_losses.mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), 1)
    optimizer.step()
    return loss.item(), mean_max_q, td_errors
