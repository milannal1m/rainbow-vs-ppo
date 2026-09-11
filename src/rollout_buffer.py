import torch


class RolloutBuffer:
    """On-policy rollout buffer with GAE advantage estimation.

    reset() -> push() per env step -> finalize(last_value) to compute GAE -> get_minibatches().
    """

    def __init__(self, capacity, gamma, gae_lambda):
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
        self.bootstraps = []
        self.advantages = None
        self.returns    = None

    def push(self, state, action, reward, log_prob, value, done, bootstrap_value=0.0):
        """`bootstrap_value` is the future value to use when done[t] is True: 0.0 for a real
        terminal state, V(s_final) for a truncation (step cap), which was cut short rather than
        lost — zeroing it would teach the agent that being truncated is worthless."""
        self.states.append(state.cpu())
        self.actions.append(action.cpu() if torch.is_tensor(action) else action)
        self.rewards.append(float(reward))
        self.log_probs.append(log_prob.cpu() if torch.is_tensor(log_prob) else log_prob)
        self.values.append(float(value.item()) if torch.is_tensor(value) else float(value))
        self.dones.append(bool(done))
        self.bootstraps.append(
            float(bootstrap_value.item()) if torch.is_tensor(bootstrap_value) else float(bootstrap_value)
        )

    def finalize(self, last_value):
        """Compute GAE advantages and discounted returns in-place.

        last_value: V(s_T) after the last rollout step. At an episode boundary there is no link
        to t+1, so the recursion is cut and the future value comes from bootstraps[t]. With the
        default 0.0 this is arithmetically identical to the old mask-based version.
        """
        T = len(self.rewards)
        last_val = float(last_value.item()) if torch.is_tensor(last_value) else float(last_value)

        advantages = [0.0] * T
        gae = 0.0

        for t in reversed(range(T)):
            if self.dones[t]:
                next_val = self.bootstraps[t]
                mask     = 0.0
            else:
                next_val = last_val if t == T - 1 else self.values[t + 1]
                mask     = 1.0
            delta     = self.rewards[t] + self.gamma * next_val - self.values[t]
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
