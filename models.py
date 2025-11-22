import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class ManagerNN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ManagerNN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class HierarchicalQNetwork(nn.Module):
    def __init__(self, state_dim=12, action_dim=3, hidden_dim=256):
        super(HierarchicalQNetwork, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.exploration_head = self._build_head(hidden_dim, action_dim)
        self.navigation_head = self._build_head(hidden_dim, action_dim)
        self.precision_head = self._build_head(hidden_dim, action_dim)
        self.phase_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Softmax(dim=-1)
        )

    def _build_head(self, hidden_dim, action_dim):
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        features = self.feature_extractor(state)
        phase_probs = self.phase_classifier(features)
        exploration_q = self.exploration_head(features)
        navigation_q = self.navigation_head(features)
        precision_q = self.precision_head(features)
        q_values = (phase_probs[:, 0:1] * exploration_q +
                    phase_probs[:, 1:2] * navigation_q +
                    phase_probs[:, 2:3] * precision_q)
        return q_values, phase_probs
