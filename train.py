# train.py
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from config import CONFIG
from environment import Nav1D
from agent import HRLTrainer
from replay_buffer import PrioritizedReplay
from models import encode_relational, NeuralUVFA, LinearUVFA

def run_experiment(seed=0, episodes=CONFIG["EPISODES"], use_prioritized=False, tuned=False, use_nn=False, use_manager_nn=False):
    random.seed(seed); np.random.seed(seed)
    env = Nav1D(size=CONFIG["ENV_SIZE"], forbidden_frac=CONFIG["FORBIDDEN_FRAC"], max_steps=CONFIG["MAX_STEPS"])
    feat_size = 2*env.size + 4
    if use_nn:
        worker = NeuralUVFA(len(env.actions), feat_size)
    else:
        worker = LinearUVFA(len(env.actions), feat_size)
    trainer = HRLTrainer(env, worker, CONFIG, use_manager_nn=use_manager_nn)
    prio = PrioritizedReplay(capacity=CONFIG["PRIO_CAP"]) if use_prioritized else None
    rewards=[]; lengths=[]; wins=[]
    for ep in range(episodes):
        total, length, epdata = trainer.run_episode(use_prioritized=use_prioritized, prio_buffer=prio)
        rewards.append(total); lengths.append(length); wins.append(1 if total>0 else 0)
        if (ep+1)%50==0 or ep==0:
            recent = np.mean(wins[-50:]) if len(wins)>=50 else np.mean(wins)
            print(f"[{'PRIO' if use_prioritized else 'BASE'}{'-T' if tuned else ''}{'-NN' if use_nn else ''}] Ep {ep+1}/{episodes} recent_win:{recent:.3f} replay:{len(trainer.replay)} prio:{len(prio) if prio is not None else 0}")
    return {'rewards':rewards,'lengths':lengths,'wins':wins,'win_rate':sum(wins)/len(wins)}

if __name__ == "__main__":
    print("Example quick run: baseline then prioritized+tuned")
    base = run_experiment(seed=1, episodes=200, use_prioritized=False, tuned=False)
    prio = run_experiment(seed=2, episodes=200, use_prioritized=True, tuned=True)
    print("BASE win_rate:", base['win_rate'])
    print("PRIO win_rate:", prio['win_rate'])
