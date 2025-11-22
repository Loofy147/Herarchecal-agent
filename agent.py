import numpy as np
import random
from collections import defaultdict

from config import CONFIG
from models import encode_relational, LinearUVFA
from replay_buffer import Replay, PrioritizedReplay, Transition

class HRLTrainer:
    def __init__(self, env, worker_model, config=CONFIG):
        self.env = env
        self.config = config
        self.n_actions = len(env.actions)
        self.worker = worker_model
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
        cg = goal * self.coarse // self.env.size
        return (ca, cg, tuple(sorted(forbidden)))

    def manager_select(self, key, eps=0.2):
        qdict = self.manager_q[key]
        candidates = list(range(self.env.size))
        if random.random() < eps or len(qdict) == 0:
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
            exp = Transition((agent, goal, tuple(sorted(forbidden))), a, r, (agent2, goal2, tuple(sorted(forbidden2))), done, subgoal, agent2)
            episode.append(exp)
            intrinsic = 1.0 if agent2 == subgoal else 0.0
            trans = Transition((agent, goal, tuple(sorted(forbidden))), a, intrinsic, (agent2, goal2, tuple(sorted(forbidden2))), intrinsic > 0, subgoal, agent2)
            if use_prioritized and prio_buffer is not None:
                prio_buffer.push(trans)
            else:
                self.replay.push(trans)
            manager_ret += r
            subgoal_timer += 1

            if len(self.replay) >= 4:
                batch = self.replay.sample(4)
                for tr in batch:
                    sfeat = encode_relational(tr.s[0], tr.goal, tr.s[2], self.env.size)
                    s2feat = encode_relational(tr.s2[0], tr.goal, tr.s2[2], self.env.size)
                    q2 = self.worker.q(s2feat)
                    target = tr.r + (0.0 if tr.done else self.gamma * np.max(q2))
                    self.worker.update(sfeat, tr.a, target, alpha=self.worker_lr)

            if intrinsic > 0 or subgoal_timer >= self.manager_interval or done:
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
                samples = episode if len(episode) <= self.her_k else random.sample(episode, self.her_k)
                for tr in samples:
                    rel_r = 1.0 if tr.s2[0] == final_pos else 0.0
                    rel_trans = Transition(tr.s, tr.a, rel_r, tr.s2, rel_r > 0, final_pos, tr.pos2)
                    self.replay.push(rel_trans)
                break
        return total_reward, t, episode
