import random
from collections import deque, namedtuple
import numpy as np
from config import CONFIG

Transition = namedtuple('Transition', ('s','a','r','s2','done','goal','pos2'))

class Replay:
    def __init__(self, capacity=CONFIG["REPLAY_CAP"]):
        self.buffer = deque(maxlen=capacity)
    def push(self, t):
        self.buffer.append(t)
    def sample(self, batch_size):
        n = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, n)
    def __len__(self):
        return len(self.buffer)

class PrioritizedReplay:
    def __init__(self, capacity=CONFIG["PRIO_CAP"], alpha=CONFIG["PRIO_ALPHA"], beta=CONFIG["PRIO_BETA"]):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = [None] * capacity
        self.priorities = np.zeros(capacity)
        self.position = 0
        self.size = 0

    def push(self, trans, priority=None):
        if priority is None:
            priority = np.max(self.priorities) if self.size > 0 else 1.0

        self.buffer[self.position] = trans
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if self.size == 0:
            return [], [], []

        prios = self.priorities[:self.size] ** self.alpha
        probs = prios / prios.sum()

        idxs = np.random.choice(self.size, size=min(batch_size, self.size), p=probs, replace=False)

        samples = [self.buffer[i] for i in idxs]
        weights = (self.size * probs[idxs]) ** (-self.beta)
        weights /= weights.max()

        return samples, idxs, weights

    def update_priorities(self, idxs, priorities):
        for i, p in zip(idxs, priorities):
            if i < self.size:
                self.priorities[i] = float(p) + 1e-6

    def __len__(self):
        return self.size
