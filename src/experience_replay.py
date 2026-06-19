from collections import deque
import random
import numpy as np
import torch


class ReplayMemory:
    def __init__(self, capacity, seed=None):
        self._rng = random.Random(seed)
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return self._rng.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class SumTree:
    """Binary sum tree for O(log n) priority-proportional sampling."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.empty(capacity, dtype=object)
        self.n_entries = 0
        self.ptr = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        return self._retrieve(left, s) if s <= self.tree[left] else self._retrieve(right, s - self.tree[left])

    @property
    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.ptr + self.capacity - 1
        self.data[self.ptr] = data
        self.update(idx, priority)
        self.ptr = (self.ptr + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        self._propagate(idx, priority - self.tree[idx])
        self.tree[idx] = priority

    def get(self, s):
        idx = self._retrieve(0, s)
        return idx, self.tree[idx], self.data[idx - self.capacity + 1]


class PrioritizedReplayMemory:
    """Proportional prioritized experience replay (Schaul et al. 2015)."""
    def __init__(self, capacity, alpha=0.5, epsilon=1e-6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_priority = 1.0

    def push(self, transition):
        self.tree.add(self.max_priority, transition)

    def sample(self, batch_size, beta=0.4):
        batch, idxs, priorities = [], [], []
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(max(priority, 1e-8))

        priorities = np.array(priorities, dtype=np.float64)
        probs = priorities / (self.tree.total + 1e-8)
        weights = (self.tree.n_entries * probs + 1e-8) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)
        return batch, idxs, weights

    def update_priorities(self, idxs, td_errors):
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        for idx, p in zip(idxs, priorities):
            self.tree.update(idx, float(p))
        self.max_priority = max(self.max_priority, float(priorities.max()))

    def __len__(self):
        return self.tree.n_entries


class NStepBuffer:
    """Computes n-step discounted returns before pushing to the main replay buffer."""
    def __init__(self, n_step, gamma, memory):
        self.n_step = n_step
        self.gamma  = gamma
        self.memory = memory
        self.buffer = deque()

    def push(self, transition):
        self.buffer.append(transition)
        if len(self.buffer) == self.n_step:
            self._commit_oldest()

    def flush(self):
        """Push all remaining transitions at episode end (with shorter horizons)."""
        while self.buffer:
            self._commit_oldest()

    def _commit_oldest(self):
        state_0, action_0 = self.buffer[0][0], self.buffer[0][1]

        n_step_reward  = 0.0
        final_ns       = None
        final_done     = False

        for i, (_, _, ns, r, done) in enumerate(self.buffer):
            r_val = r.item() if hasattr(r, 'item') else float(r)
            n_step_reward += (self.gamma ** i) * r_val
            final_ns = ns
            if done:
                final_done = True
                break

        device = state_0.device if hasattr(state_0, 'device') else 'cpu'
        r_tensor = torch.tensor(n_step_reward, dtype=torch.float32, device=device)
        self.memory.push((state_0, action_0, final_ns, r_tensor, final_done))
        self.buffer.popleft()
