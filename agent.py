# agent.py
import numpy as np
import random
import torch
from collections import defaultdict
from config import CONFIG
from replay_buffer import Replay, Transition
from models import LinearUVFA, NeuralUVFA, ManagerNN, encode_relational, encode_manager_state

class HRLTrainer:
    def __init__(self, env, worker, config=CONFIG, use_manager_nn=False):
        self.env = env
        self.config = config
        self.n_actions = len(env.actions)
        self.worker = worker
        self.gamma = config["GAMMA"]
        self.eps = config["EPS_START"]
        self.eps_end = config["EPS_END"]
        self.eps_decay = config["EPS_DECAY"]
        self.manager_interval = config["MANAGER_INTERVAL"]
        if use_manager_nn:
            self.manager = ManagerNN(3 * env.size, env.size)
        else:
            self.manager = defaultdict(lambda: defaultdict(float))
        self.replay = Replay(config["REPLAY_CAP"])
        self.her_k = config["HER_K"]
        self.worker_lr = config["WORKER_LR"]

    def manager_select(self, state, eps=None):
        if eps is None:
            eps = self.config["MANAGER_EPS"]

        if random.random() < eps:
            return random.randrange(self.env.size)

        if isinstance(self.manager, ManagerNN):
            q_values = self.manager.q(state)
            return int(torch.argmax(q_values))
        else:
            qdict = self.manager[state]
            candidates = list(range(self.env.size))
            if len(qdict)==0:
                return random.choice(candidates)
            best = max(candidates, key=lambda c: qdict.get(c, 0.0))
            return best

    def manager_update(self, state, subgoal, ret, alpha=None):
        if alpha is None:
            alpha = self.config["MANAGER_LR"]

        if isinstance(self.manager, ManagerNN):
            self.manager.update(state, subgoal, ret)
        else:
            old = self.manager[state].get(subgoal, 0.0)
            self.manager[state][subgoal] = old + alpha * (ret - old)

    def run_episode(self, use_prioritized=False, prio_buffer=None):
        obs = self.env.reset()
        agent, goal, forbidden = obs
        total_reward = 0.0
        t = 0
        episode = []

        if isinstance(self.manager, ManagerNN):
            manager_state = encode_manager_state(agent, goal, forbidden, self.env.size)
        else:
            manager_state = (agent * self.config["MANAGER_STATE_QUANTIZATION"] // self.env.size,
                             goal * self.config["MANAGER_STATE_QUANTIZATION"] // self.env.size,
                             tuple(sorted(forbidden)))

        subgoal = self.manager_select(manager_state)
        manager_ret = 0.0
        subgoal_timer = 0

        while True:
            w_feat = encode_relational(agent, subgoal, forbidden, self.env.size)
            if random.random() < self.eps:
                a = random.randrange(self.n_actions)
            else:
                qv = self.worker.q(w_feat)
                if isinstance(self.worker, NeuralUVFA):
                    a = int(torch.argmax(qv))
                else:
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

            # prioritized replay update
            if use_prioritized and prio_buffer is not None and len(prio_buffer) >= self.config["PRIO_BATCH_SIZE"]:
                samples, idxs, weights = prio_buffer.sample(self.config["PRIO_BATCH_SIZE"])
                td_list=[]
                for tr in samples:
                    sfeat = encode_relational(tr.s[0], tr.goal, tr.s[2], self.env.size)
                    s2feat = encode_relational(tr.s2[0], tr.goal, tr.s2[2], self.env.size)
                    q2 = self.worker.q(s2feat)
                    if isinstance(self.worker, NeuralUVFA):
                        with torch.no_grad():
                            target = tr.r + (0.0 if tr.done else self.gamma * torch.max(q2))
                        td = self.worker.update(sfeat, tr.a, target)
                    else:
                        target = tr.r + (0.0 if tr.done else self.gamma * np.max(q2))
                        td = self.worker.update(sfeat, tr.a, target, alpha=self.worker_lr)
                    td_list.append(td)
                prio_buffer.update_priorities(idxs, td_list)
            # small uniform update pass to keep training continuous
            elif len(self.replay) >= self.config["ONLINE_BATCH_SIZE"]:
                batch = self.replay.sample(self.config["ONLINE_BATCH_SIZE"])
                for tr in batch:
                    sfeat = encode_relational(tr.s[0], tr.goal, tr.s[2], self.env.size)
                    s2feat = encode_relational(tr.s2[0], tr.goal, tr.s2[2], self.env.size)
                    q2 = self.worker.q(s2feat)
                    if isinstance(self.worker, NeuralUVFA):
                        with torch.no_grad():
                            target = tr.r + (0.0 if tr.done else self.gamma * torch.max(q2))
                        self.worker.update(sfeat, tr.a, target)
                    else:
                        target = tr.r + (0.0 if tr.done else self.gamma * np.max(q2))
                        self.worker.update(sfeat, tr.a, target, alpha=self.worker_lr)

            if intrinsic>0 or subgoal_timer >= self.manager_interval or done:
                self.manager_update(manager_state, subgoal, manager_ret)
                manager_ret = 0.0
                subgoal_timer = 0
                if isinstance(self.manager, ManagerNN):
                    manager_state = encode_manager_state(agent2, goal2, forbidden2, self.env.size)
                else:
                    manager_state = (agent2 * self.config["MANAGER_STATE_QUANTIZATION"] // self.env.size,
                                     goal2 * self.config["MANAGER_STATE_QUANTIZATION"] // self.env.size,
                                     tuple(sorted(forbidden2)))
                subgoal = self.manager_select(manager_state)

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
