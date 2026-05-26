from collections import deque
import random

class ReplayMemory:
    def __init__(self, capacity, seed=None):
        if seed is not None:
            random.seed(seed)
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)