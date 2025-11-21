# models.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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

class NeuralUVFA(nn.Module):
    def __init__(self, n_actions, feat_size, hidden_size=64):
        super(NeuralUVFA, self).__init__()
        self.n_actions = n_actions
        self.feat_size = feat_size
        self.model = nn.Sequential(
            nn.Linear(feat_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

    def q(self, feat):
        if not isinstance(feat, torch.Tensor):
            feat = torch.from_numpy(feat).float()
        return self.model(feat)

    def update(self, feat, a, target):
        if not isinstance(feat, torch.Tensor):
            feat = torch.from_numpy(feat).float()

        q_values = self.q(feat)

        q_value = q_values[a]

        target_tensor = torch.tensor([target]).float()

        loss = self.loss_fn(q_value, target_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
