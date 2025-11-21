import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple

from model import HierarchicalQNetwork
from environment import PuzzleEnvironment
from core import get_hierarchical_state_representation, get_shaped_reward, decompose_target

# Define the structure for a single transition in the replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    """A simple replay buffer for storing and sampling experiences."""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Sample a batch of transitions."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class HierarchicalAgent:
    """
    A hierarchical agent that uses constraint-aware subgoal decomposition to solve complex tasks.
    """
    def __init__(self, state_dim, action_dim, learning_rate=0.0005, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64, tau=0.005):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = tau

        # --- Thresholds ---
        self.proximity_threshold = 10
        self.distance_threshold = 50
        self.exploration_threshold = 0.5
        self.precision_threshold = 5
        self.min_zone_size = 10 # For subgoal decomposition

        self.policy_net = HierarchicalQNetwork(state_dim, action_dim)
        self.target_net = HierarchicalQNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size)

    def select_action(self, state, valid_actions):
        """
        Selects an action using an epsilon-greedy policy, but with constraint awareness.
        """
        if not valid_actions:
            return self.emergency_backtrack_action()

        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values, _ = self.policy_net(state_tensor)

                # Mask out invalid actions
                masked_q_values = q_values.clone()
                for i in range(self.action_dim):
                    if i not in valid_actions:
                        masked_q_values[0, i] = -float('inf')

                return masked_q_values.argmax().item()

    def emergency_backtrack_action(self):
        """
        Defines a safe action to take when no other valid moves are available.
        """
        # A simple but effective strategy: move back by the smallest possible step.
        # This assumes that the action space is ordered [move_1, move_3, move_5, ...]
        return 0 # Corresponds to the smallest action

    def decompose_target_hierarchically(self, current, target, forbidden_states):
        """
        Adaptive decomposition that respects constraints by creating subgoals in safe zones.
        """
        if not forbidden_states or abs(target - current) < self.distance_threshold:
            return [target]

        safe_zones = self._identify_safe_zones(current, target, forbidden_states)

        subgoals = []
        last_pos = current

        for zone_start, zone_end in safe_zones:
            # Add the start of the safe zone as a subgoal
            if zone_start != last_pos:
                subgoals.append(zone_start)

            # Decompose within the safe zone if it's large enough
            zone_size = abs(zone_end - zone_start)
            if zone_size > self.min_zone_size:
                # Add a subgoal in the middle of the safe zone
                subgoal = (zone_start + zone_end) // 2
                subgoals.append(subgoal)

            # The end of the zone will be the start of the next or the target
            last_pos = zone_end

        subgoals.append(target)

        # Remove redundant subgoals and ensure sorted order
        final_subgoals = []
        for subgoal in sorted(subgoals):
            if subgoal != current and (not final_subgoals or subgoal != final_subgoals[-1]):
                final_subgoals.append(subgoal)

        return final_subgoals

    def _identify_safe_zones(self, current, target, forbidden):
        """Identifies contiguous safe zones between current and target."""
        direction = 1 if target > current else -1
        path = range(min(current, target), max(current, target) + 1)

        sorted_forbidden = sorted([f for f in forbidden if f in path])

        if not sorted_forbidden:
            return [(current, target)]

        zones = []
        zone_start = current

        for f_state in sorted_forbidden:
            if zone_start * direction < f_state * direction:
                zones.append((zone_start, f_state - direction))
            zone_start = f_state + direction

        if zone_start * direction <= target * direction:
            zones.append((zone_start, target))

        return zones

    def train_step(self):
        """Performs a single training step on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.FloatTensor(np.array(batch.state))
        action_batch = torch.LongTensor(batch.action).unsqueeze(1)
        reward_batch = torch.FloatTensor(batch.reward)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state))
        done_batch = torch.BoolTensor(batch.done)

        # Get current Q-values from the policy network
        current_q_values, _ = self.policy_net(state_batch)
        current_q_values = current_q_values.gather(1, action_batch)

        # Get next Q-values from the target network
        with torch.no_grad():
            next_q_values, _ = self.target_net(next_state_batch)
            max_next_q_values = next_q_values.max(1)[0]
            # Zero out the Q-values for terminal states
            max_next_q_values[done_batch] = 0.0

        # Compute the expected Q-values
        expected_q_values = reward_batch + (self.gamma * max_next_q_values)

        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
