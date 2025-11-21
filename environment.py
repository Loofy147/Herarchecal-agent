# environment.py
import numpy as np
import random
from config import CONFIG

class Nav1D:
    def __init__(self, size=CONFIG["ENV_SIZE"], forbidden_frac=CONFIG["FORBIDDEN_FRAC"], max_steps=CONFIG["MAX_STEPS"]):
        self.size = size
        self.max_steps = max_steps
        self.forbidden_frac = forbidden_frac
        self.actions = [-5, -1, 0, +1, +5]
        self.reset()

    def reset(self, random_forbidden=True):
        n_forbidden = max(1, int(self.size * self.forbidden_frac))
        if random_forbidden:
            self.forbidden = set(random.sample(range(self.size), n_forbidden))
        else:
            self.forbidden = set([self.size//2])
        while True:
            self.agent = random.randrange(self.size)
            if self.agent not in self.forbidden:
                break
        while True:
            self.goal = random.randrange(self.size)
            if self.goal not in self.forbidden and self.goal != self.agent:
                break
        self.steps = 0
        return self.obs()

    def obs(self):
        return (self.agent, self.goal, tuple(sorted(self.forbidden)))

    def step(self, action_idx):
        move = self.actions[action_idx]
        self.steps += 1
        newpos = int(np.clip(self.agent + move, 0, self.size-1))
        if newpos in self.forbidden:
            reward = -1.0
            done = False
            newpos = self.agent
        else:
            self.agent = newpos
            reward = 0.0
            done = False
            if self.agent == self.goal:
                reward = 1.0
                done = True
        if self.steps >= self.max_steps:
            done = True
        return self.obs(), reward, done, {}

    def render(self):
        s = ['.'] * self.size
        for f in self.forbidden:
            s[f] = 'X'
        s[self.goal] = 'G'
        s[self.agent] = 'A'
        print(''.join(s))
