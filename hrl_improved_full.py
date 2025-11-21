# hrl_improved_full.py
# Improved HRL 1D navigation — fixed HER, positional info, prioritized replay, manager+worker (UVFA).
# Paste into Colab and run. Example usage at bottom.

import numpy as np, random, time, json, os
from collections import deque, namedtuple, defaultdict
import matplotlib.pyplot as plt

# ---------------------------
# CONFIG (tune these)
# ---------------------------
CONFIG = {
    "ENV_SIZE": 21,
    "FORBIDDEN_FRAC": 0.12,
    "MAX_STEPS": 50,
    "EPISODES": 400,
    "MANAGER_INTERVAL": 5,
    "WORKER_LR": 0.16,
    "GAMMA": 0.95,
    "EPS_START": 0.4,
    "EPS_END": 0.05,
    "EPS_DECAY": 0.9995,
    "REPLAY_CAP": 5000,
    "PRIO_CAP": 4000,
    "HER_K": 6,
    "PRIO_ALPHA": 0.6,
    "PRIO_BETA": 0.4,
    "SEED": 1
}

# ---------------------------
# Simple 1D Nav environment
# ---------------------------
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

# ---------------------------
# Relational encoder
# ---------------------------
def encode_relational(agent, goal, forbidden, size):
    agent_oh = np.zeros(size); agent_oh[agent] = 1.0
    goal_oh  = np.zeros(size); goal_oh[goal] = 1.0
    dist = (goal - agent) / (size-1)
    dir_sign = np.sign(goal - agent)
    left = [agent - f for f in forbidden if f < agent]
    right = [f - agent for f in forbidden if f > agent]
    dl = (min(left)/ (size-1)) if left else 1.0
    dr = (min(right)/ (size-1)) if right else 1.0
    feat = np.concatenate([agent_oh, goal_oh, np.array([dist, dir_sign, dl, dr])])
    norm = np.linalg.norm(feat)
    if norm>0: feat = feat / norm
    return feat

# ---------------------------
# Linear UVFA approximator
# ---------------------------
class LinearUVFA:
    def __init__(self, n_actions, feat_size):
        self.n_actions = n_actions
        self.feat_size = feat_size
        self.W = np.zeros((n_actions, feat_size), dtype=float)
    def q(self, feat):
        return self.W.dot(feat)
    def update(self, feat, a, target, alpha=0.1):
        qv = float(self.W[a].dot(feat))
        td = target - qv
        self.W[a] += alpha * td * feat
        return abs(td)

# ---------------------------
# Prioritized replay (proportional)
# ---------------------------
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

# ---------------------------
# Uniform replay
# ---------------------------
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

# ---------------------------
# HRL Trainer (Manager + Worker) with correct HER
# ---------------------------
class HRLTrainer:
    def __init__(self, env, feat_size, config=CONFIG):
        self.env = env
        self.config = config
        self.n_actions = len(env.actions)
        self.worker = LinearUVFA(self.n_actions, feat_size)
        self.gamma = config["GAMMA"]
        self.eps = config["EPS_START"]
        self.eps_end = config["EPS_END"]
        self.eps_decay = config["EPS_DECAY"]
        self.coarse = 5
        self.manager_interval = config["MANAGER_INTERVAL"]
        self.manager_q = defaultdict(lambda: defaultdict(float))
        self.replay = Replay(config["REPLAY_CAP"])
        self.her_k = config["HER_K"]
        self.worker_lr = config["WORKER_LR"]

    def state_key(self, agent, goal, forbidden):
        ca = agent * self.coarse // self.env.size
        cg = goal  * self.coarse // self.env.size
        return (ca,cg, tuple(sorted(forbidden)))

    def manager_select(self, key, eps=0.2):
        qdict = self.manager_q[key]
        candidates = list(range(self.env.size))
        if random.random() < eps or len(qdict)==0:
            return random.choice(candidates)
        best = max(candidates, key=lambda c: qdict.get(c, 0.0))
        return best

    def manager_update(self, key, subgoal, ret, alpha=0.1):
        old = self.manager_q[key].get(subgoal, 0.0)
        self.manager_q[key][subgoal] = old + alpha * (ret - old)

    def run_episode(self, use_prioritized=False, prio_buffer=None):
        obs = self.env.reset()
        agent, goal, forbidden = obs
        total_reward = 0.0
        t = 0
        episode = []
        manager_key = self.state_key(agent, goal, forbidden)
        subgoal = self.manager_select(manager_key, eps=0.2)
        manager_ret = 0.0
        subgoal_timer = 0

        while True:
            w_feat = encode_relational(agent, subgoal, forbidden, self.env.size)
            if random.random() < self.eps:
                a = random.randrange(self.n_actions)
            else:
                qv = self.worker.q(w_feat)
                a = int(np.argmax(qv))
            (agent2, goal2, forbidden2), r, done, _ = self.env.step(a)
            total_reward += r
            # store raw pos for HER
            exp = Transition((agent,goal,tuple(sorted(forbidden))), a, r, (agent2,goal2,tuple(sorted(forbidden2))), done, subgoal, agent2)
            episode.append(exp)
            intrinsic = 1.0 if agent2 == subgoal else 0.0
            trans = Transition((agent,goal,tuple(sorted(forbidden))), a, intrinsic, (agent2,goal2,tuple(sorted(forbidden2))), intrinsic>0, subgoal, agent2)
            if use_prioritized and prio_buffer is not None:
                prio_buffer.push(trans)
            else:
                self.replay.push(trans)
            manager_ret += r
            subgoal_timer += 1

            # small uniform update pass to keep training continuous
            if len(self.replay) >= 4:
                batch = self.replay.sample(4)
                for tr in batch:
                    sfeat = encode_relational(tr.s[0], tr.goal, tr.s[2], self.env.size)
                    s2feat = encode_relational(tr.s2[0], tr.goal, tr.s2[2], self.env.size)
                    q2 = self.worker.q(s2feat)
                    target = tr.r + (0.0 if tr.done else self.gamma * np.max(q2))
                    self.worker.update(sfeat, tr.a, target, alpha=self.worker_lr)

            if intrinsic>0 or subgoal_timer >= self.manager_interval or done:
                self.manager_update(manager_key, subgoal, manager_ret, alpha=0.1)
                manager_ret = 0.0
                subgoal_timer = 0
                manager_key = self.state_key(agent2, goal2, forbidden2)
                subgoal = self.manager_select(manager_key, eps=0.1)

            agent, goal, forbidden = agent2, goal2, forbidden2
            t += 1
            self.eps = max(self.eps_end, self.eps * self.eps_decay)
            if done or t >= self.env.max_steps:
                final_pos = agent
                samples = episode if len(episode)<=self.her_k else random.sample(episode, self.her_k)
                for tr in samples:
                    rel_r = 1.0 if tr.s2[0] == final_pos else 0.0
                    rel_trans = Transition(tr.s, tr.a, rel_r, tr.s2, rel_r>0, final_pos, tr.pos2)
                    self.replay.push(rel_trans)
                break
        return total_reward, t, episode

# ---------------------------
# Experiment runner + plotting helper
# ---------------------------
def run_experiment(seed=0, episodes=CONFIG["EPISODES"], use_prioritized=False, tuned=False):
    random.seed(seed); np.random.seed(seed)
    env = Nav1D(size=CONFIG["ENV_SIZE"], forbidden_frac=CONFIG["FORBIDDEN_FRAC"], max_steps=CONFIG["MAX_STEPS"])
    feat_size = 2*env.size + 4
    trainer = HRLTrainer(env, feat_size, CONFIG)
    prio = PrioritizedReplay(capacity=CONFIG["PRIO_CAP"]) if use_prioritized else None
    rewards=[]; lengths=[]; wins=[]
    for ep in range(episodes):
        total, length, epdata = trainer.run_episode(use_prioritized=use_prioritized, prio_buffer=prio)
        rewards.append(total); lengths.append(length); wins.append(1 if total>0 else 0)
        if use_prioritized and prio is not None and len(prio)>=8:
            samples, idxs, weights = prio.sample(8)
            td_list=[]
            for tr in samples:
                sfeat = encode_relational(tr.s[0], tr.goal, tr.s[2], env.size)
                s2feat = encode_relational(tr.s2[0], tr.goal, tr.s2[2], env.size)
                q2 = trainer.worker.q(s2feat)
                target = tr.r + (0.0 if tr.done else trainer.gamma * np.max(q2))
                td = trainer.worker.update(sfeat, tr.a, target, alpha=trainer.worker_lr)
                td_list.append(td)
            prio.update_priorities(idxs, td_list)
        if (ep+1)%50==0 or ep==0:
            recent = np.mean(wins[-50:]) if len(wins)>=50 else np.mean(wins)
            print(f"[{'PRIO' if use_prioritized else 'BASE'}{'-T' if tuned else ''}] Ep {ep+1}/{episodes} recent_win:{recent:.3f} replay:{len(trainer.replay)} prio:{len(prio) if prio is not None else 0}")
    return {'rewards':rewards,'lengths':lengths,'wins':wins,'win_rate':sum(wins)/len(wins)}

# ---------------------------
# Example run (uncomment to run in Colab)
# ---------------------------
if __name__ == "__main__":
    print("Example quick run: baseline then prioritized+tuned")
    base = run_experiment(seed=1, episodes=200, use_prioritized=False, tuned=False)
    prio = run_experiment(seed=2, episodes=200, use_prioritized=True, tuned=True)
    print("BASE win_rate:", base['win_rate'])
    print("PRIO win_rate:", prio['win_rate'])
