# replay_buffer.py
import numpy as np
import random
from collections import deque, namedtuple
from config import CONFIG

Transition = namedtuple('Transition', ('s','a','r','s2','done','goal','pos2'))

class PrioritizedReplay:
    def __init__(self, capacity=CONFIG["PRIO_CAP"], alpha=CONFIG["PRIO_ALPHA"], beta=CONFIG["PRIO_BETA"]):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = []

    def push(self, trans, priority=None):
        if len(self.buffer) < self.capacity:
            self.buffer.append(trans)
            self.priorities.append(priority if priority is not None else (max(self.priorities) if self.priorities else 1.0))
        else:
            idx = len(self.buffer) % self.capacity
            self.buffer[idx] = trans
            self.priorities[idx] = priority if priority is not None else max(self.priorities)

    def sample(self, batch_size):
        if len(self.buffer)==0:
            return [], [], []
        prios = np.array(self.priorities) ** self.alpha
        probs = prios / prios.sum()
        idxs = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), p=probs, replace=False)
        samples = [self.buffer[i] for i in idxs]
        weights = (len(self.buffer) * probs[idxs]) ** (-self.beta)
        weights = weights / (weights.max() + 1e-8)
        return samples, list(idxs), weights

    def update_priorities(self, idxs, priorities):
        for i,p in zip(idxs, priorities):
            if i < len(self.priorities):
                self.priorities[i] = float(p) + 1e-6

    def __len__(self):
        return len(self.buffer)

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
