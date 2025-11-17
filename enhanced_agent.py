import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple

from model import HierarchicalQNetwork
from environment import PuzzleEnvironment
from core import get_hierarchical_state_representation, get_shaped_reward, decompose_target

# Enhanced transition structure with priority
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class SumTree:
    """Sum Tree data structure for efficient prioritized sampling."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer with importance sampling."""
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta_start = beta_start  # Importance sampling weight
        self.beta_frames = beta_frames
        self.frame = 1
        self.epsilon = 0.01  # Small constant to avoid zero priority
        self.abs_err_upper = 1.0  # Clipped abs error

    def _get_priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.alpha

    def push(self, *args):
        """Add experience with max priority."""
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = self.abs_err_upper
        
        transition = Transition(*args)
        self.tree.add(max_priority, transition)

    def sample(self, batch_size):
        """Sample batch with priorities."""
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total() / batch_size

        # Annealing importance sampling weight
        self.beta = np.min([1., self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames])
        self.frame += 1

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            
            priorities.append(priority)
            batch.append(data)
            idxs.append(idx)

        # Calculate importance sampling weights
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()  # Normalize

        return batch, idxs, is_weights

    def update_priorities(self, idxs, errors):
        """Update priorities based on TD errors."""
        for idx, error in zip(idxs, errors):
            priority = self._get_priority(error)
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.n_entries


class RewardNormalizer:
    """Running reward normalization for stable training."""
    def __init__(self, epsilon=1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.epsilon = epsilon

    def update(self, reward):
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        delta2 = reward - self.mean
        self.var += (delta * delta2 - self.var) / self.count

    def normalize(self, reward):
        return reward / (np.sqrt(self.var) + self.epsilon)


class EnhancedDQNAgent:
    """Enhanced DQN Agent with Double DQN, PER, and reward normalization."""
    def __init__(self, state_dim, action_dim, learning_rate=0.0005, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, 
                 buffer_size=10000, batch_size=64, tau=0.005,
                 use_double_dqn=True, use_per=True, use_reward_norm=True):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = tau
        
        # Feature flags
        self.use_double_dqn = use_double_dqn
        self.use_per = use_per
        self.use_reward_norm = use_reward_norm

        self.policy_net = HierarchicalQNetwork(state_dim, action_dim)
        self.target_net = HierarchicalQNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Choose replay buffer type
        if self.use_per:
            self.memory = PrioritizedReplayBuffer(buffer_size)
        else:
            from agent import ReplayBuffer
            self.memory = ReplayBuffer(buffer_size)
        
        # Reward normalization
        if self.use_reward_norm:
            self.reward_normalizer = RewardNormalizer()
        
        # Metrics tracking
        self.metrics = {
            'td_errors': [],
            'losses': [],
            'q_values': [],
            'phase_distributions': []
        }

    def select_action(self, state):
        """Selects an action using an epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values, phase_probs = self.policy_net(state_tensor)
                
                # Track metrics
                self.metrics['q_values'].append(q_values.max().item())
                self.metrics['phase_distributions'].append(phase_probs.cpu().numpy()[0])
                
                return q_values.argmax().item()

    def train_step(self):
        """Enhanced training step with Double DQN and PER."""
        if len(self.memory) < self.batch_size:
            return None

        # Sample from replay buffer
        if self.use_per:
            transitions, idxs, is_weights = self.memory.sample(self.batch_size)
            is_weights = torch.FloatTensor(is_weights)
        else:
            transitions = self.memory.sample(self.batch_size)
            idxs = None
            is_weights = torch.ones(self.batch_size)

        batch = Transition(*zip(*transitions))

        state_batch = torch.FloatTensor(np.array(batch.state))
        action_batch = torch.LongTensor(batch.action).unsqueeze(1)
        reward_batch = torch.FloatTensor(batch.reward)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state))
        done_batch = torch.BoolTensor(batch.done)

        # Get current Q-values from the policy network
        current_q_values, _ = self.policy_net(state_batch)
        current_q_values = current_q_values.gather(1, action_batch)

        # Compute next Q-values using Double DQN or standard DQN
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: use policy network to select action, target network to evaluate
                next_q_policy, _ = self.policy_net(next_state_batch)
                next_actions = next_q_policy.argmax(1).unsqueeze(1)
                
                next_q_target, _ = self.target_net(next_state_batch)
                max_next_q_values = next_q_target.gather(1, next_actions).squeeze(1)
            else:
                # Standard DQN
                next_q_values, _ = self.target_net(next_state_batch)
                max_next_q_values = next_q_values.max(1)[0]
            
            # Zero out Q-values for terminal states
            max_next_q_values[done_batch] = 0.0

        # Compute expected Q-values
        expected_q_values = reward_batch + (self.gamma * max_next_q_values)

        # Compute TD errors for PER
        td_errors = (expected_q_values.unsqueeze(1) - current_q_values).detach()
        
        # Compute loss with importance sampling weights
        element_wise_loss = F.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(1), reduction='none')
        loss = (element_wise_loss * is_weights.unsqueeze(1)).mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # Update priorities in PER
        if self.use_per and idxs is not None:
            self.memory.update_priorities(idxs, td_errors.squeeze().cpu().numpy())

        # Track metrics
        self.metrics['td_errors'].append(td_errors.abs().mean().item())
        self.metrics['losses'].append(loss.item())

        return loss.item()

    def update_target_network(self):
        """Performs a soft update of the target network's weights."""
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def update_epsilon(self):
        """Decays the epsilon value."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def get_metrics_summary(self):
        """Returns summary statistics of recent metrics."""
        if not self.metrics['losses']:
            return {}
        
        return {
            'avg_loss': np.mean(self.metrics['losses'][-100:]),
            'avg_td_error': np.mean(self.metrics['td_errors'][-100:]),
            'avg_q_value': np.mean(self.metrics['q_values'][-100:]) if self.metrics['q_values'] else 0,
            'avg_phase_dist': np.mean(self.metrics['phase_distributions'][-100:], axis=0) if self.metrics['phase_distributions'] else [0, 0, 0]
        }
