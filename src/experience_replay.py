from collections import deque
import random

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
