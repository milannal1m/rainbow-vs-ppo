import torch


class RolloutBuffer:
    """On-policy rollout buffer with GAE advantage estimation.

    Usage:
        buffer.reset()
        for each env step:
            buffer.push(state, action, reward, log_prob, value, done)
        buffer.finalize(last_value)   # compute GAE
        for batch in buffer.get_minibatches(minibatch_size, device):
            ...  # (states, actions, log_probs, advantages, returns)
    """

    def __init__(self, capacity, gamma, gae_lambda):
        self.capacity   = capacity
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.reset()

    def reset(self):
        self.states    = []
        self.actions   = []
        self.rewards   = []
        self.log_probs = []
        self.values    = []
        self.dones     = []
        self.advantages = None
        self.returns    = None

    def push(self, state, action, reward, log_prob, value, done):
        self.states.append(state.cpu())
        self.actions.append(action.cpu() if torch.is_tensor(action) else action)
        self.rewards.append(float(reward))
        self.log_probs.append(log_prob.cpu() if torch.is_tensor(log_prob) else log_prob)
        self.values.append(float(value.item()) if torch.is_tensor(value) else float(value))
        self.dones.append(bool(done))

    def finalize(self, last_value):
        """Compute GAE advantages and discounted returns in-place.

        last_value: V(s_T) bootstrapped from the state after the last rollout step.
        When done[t]=True the episode terminated at step t, so the bootstrap for
        the next step is zeroed out via the mask.
        """
        T = len(self.rewards)
        last_val = float(last_value.item()) if torch.is_tensor(last_value) else float(last_value)

        advantages = [0.0] * T
        gae = 0.0

        for t in reversed(range(T)):
            next_val  = last_val if t == T - 1 else self.values[t + 1]
            mask      = 1.0 - float(self.dones[t])
            delta     = self.rewards[t] + self.gamma * next_val * mask - self.values[t]
            gae       = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae

        self.advantages = advantages
        self.returns    = [adv + val for adv, val in zip(advantages, self.values)]

    def get_minibatches(self, minibatch_size, device):
        """Yield shuffled minibatches of (states, actions, log_probs, advantages, returns).
        Advantages are normalised per full buffer before splitting."""
        assert self.advantages is not None, "Call finalize() before get_minibatches()"

        T = len(self.states)
        indices = torch.randperm(T)

        states     = torch.stack(self.states).to(device)
        actions    = torch.stack(self.actions).to(device) if torch.is_tensor(self.actions[0]) \
                     else torch.tensor(self.actions, dtype=torch.int64).to(device)
        log_probs  = torch.stack(self.log_probs).to(device)
        advantages = torch.tensor(self.advantages, dtype=torch.float32).to(device)
        returns    = torch.tensor(self.returns,    dtype=torch.float32).to(device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for start in range(0, T, minibatch_size):
            idx = indices[start : start + minibatch_size]
            yield states[idx], actions[idx], log_probs[idx], advantages[idx], returns[idx]
