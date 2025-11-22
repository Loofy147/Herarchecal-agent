import numpy as np
import random
import matplotlib.pyplot as plt

from config import CONFIG
from environment import Nav1D
from models import LinearUVFA, encode_relational
from agent import HRLTrainer
from replay_buffer import PrioritizedReplay

def run_experiment(seed=0, episodes=CONFIG["EPISODES"], use_prioritized=False, tuned=False):
    random.seed(seed)
    np.random.seed(seed)
    env = Nav1D(size=CONFIG["ENV_SIZE"], forbidden_frac=CONFIG["FORBIDDEN_FRAC"], max_steps=CONFIG["MAX_STEPS"])
    feat_size = 2 * env.size + 4
    worker_model = LinearUVFA(len(env.actions), feat_size)
    trainer = HRLTrainer(env, worker_model, CONFIG)
    prio = PrioritizedReplay(capacity=CONFIG["PRIO_CAP"]) if use_prioritized else None
    rewards = []
    lengths = []
    wins = []
    for ep in range(episodes):
        total, length, epdata = trainer.run_episode(use_prioritized=use_prioritized, prio_buffer=prio)
        rewards.append(total)
        lengths.append(length)
        wins.append(1 if total > 0 else 0)
        if use_prioritized and prio is not None and len(prio) >= 8:
            samples, idxs, weights = prio.sample(8)
            td_list = []
            for tr in samples:
                sfeat = encode_relational(tr.s[0], tr.goal, tr.s[2], env.size)
                s2feat = encode_relational(tr.s2[0], tr.goal, tr.s2[2], env.size)
                q2 = trainer.worker.q(s2feat)
                target = tr.r + (0.0 if tr.done else trainer.gamma * np.max(q2))
                td = trainer.worker.update(sfeat, tr.a, target, alpha=trainer.worker_lr)
                td_list.append(td)
            prio.update_priorities(idxs, td_list)
        if (ep + 1) % 50 == 0 or ep == 0:
            recent = np.mean(wins[-50:]) if len(wins) >= 50 else np.mean(wins)
            print(f"[{'PRIO' if use_prioritized else 'BASE'}{'-T' if tuned else ''}] Ep {ep + 1}/{episodes} recent_win:{recent:.3f} replay:{len(trainer.replay)} prio:{len(prio) if prio is not None else 0}")
    return {'rewards': rewards, 'lengths': lengths, 'wins': wins, 'win_rate': sum(wins) / len(wins)}

if __name__ == "__main__":
    print("Example quick run: baseline then prioritized+tuned")
    base = run_experiment(seed=1, episodes=200, use_prioritized=False, tuned=False)
    prio = run_experiment(seed=2, episodes=200, use_prioritized=True, tuned=True)
    print("BASE win_rate:", base['win_rate'])
    print("PRIO win_rate:", prio['win_rate'])
