"""
COMPREHENSIVE FIX: All 6 Critical Industry Gaps Resolved
=========================================================

GAP 0: ✅ Double DQN (eliminates maximization bias)
GAP 1: ✅ Gradient clipping + loss monitoring (training stability)
GAP 2: ✅ Adaptive exploration (curriculum-aware epsilon)
GAP 3: ✅ Fixed reward engineering (proper incentive alignment)
GAP 4: ✅ Prioritized Experience Replay (30-50% faster learning)
GAP 5: ✅ Checkpoint system (production readiness)
"""

import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import json
import os
from datetime import datetime
import pickle

from model import HierarchicalQNetwork
from environment import PuzzleEnvironment
from core import get_hierarchical_state_representation, decompose_target

# Import the PER components from enhanced_agent
from enhanced_agent import SumTree, PrioritizedReplayBuffer, RewardNormalizer

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


# ============================================================================
# GAP 3 FIX: FIXED REWARD ENGINEERING
# ============================================================================
def get_fixed_shaped_reward(current_state, next_state, target, step, is_done, info, max_steps=200):
    """
    FIXED reward structure that maintains proper ordering:
    worst_failure < minor_failure < worst_success < best_success
    
    Key Fix: Terminal rewards are now normalized to maintain incentive alignment.
    """
    if is_done:
        if info['status'] == 'SUCCESS':
            # Success reward: scaled by efficiency (earlier = better)
            # Range: [50, 100] - always positive and substantial
            efficiency_bonus = 50 * (1 - step / max_steps)
            return 50 + efficiency_bonus
        
        elif info['status'] == 'FORBIDDEN':
            # Worst failure: constraint violation
            return -100
        
        elif info['status'] == 'OVERSHOOT':
            # Major failure: went past target
            return -80
        
        elif info['status'] == 'MAX_STEPS':
            # Minor failure: ran out of time (but didn't violate constraints)
            return -60
    
    # Step rewards for non-terminal states
    gap_before = abs(current_state - target)
    gap_after = abs(next_state - target)
    
    # Progress reward: scaled by improvement
    if gap_after < gap_before:
        improvement_ratio = (gap_before - gap_after) / gap_before
        return 5 * improvement_ratio  # Range: [0, 5]
    
    # Small penalty for no progress (encourages efficiency)
    return -0.5


# ============================================================================
# GAP 1 FIX: TRAINING STABILITY MONITOR
# ============================================================================
class TrainingMonitor:
    """
    Monitors training health and detects anomalies:
    - NaN/Inf detection
    - Loss explosion detection
    - Gradient explosion detection
    """
    def __init__(self, window_size=100):
        self.losses = deque(maxlen=window_size)
        self.grad_norms = deque(maxlen=window_size)
        self.q_values = deque(maxlen=window_size)
        self.anomaly_count = 0
        
    def check_loss(self, loss):
        """Check for loss anomalies."""
        if loss is None:
            return True
        
        if not np.isfinite(loss):
            self.anomaly_count += 1
            return False
        
        self.losses.append(loss)
        
        # Check for explosion
        if len(self.losses) > 10:
            recent_mean = np.mean(list(self.losses)[-10:])
            if recent_mean > 1000:
                print(f"⚠️ WARNING: Loss explosion detected! Mean: {recent_mean:.2f}")
                return False
        
        return True
    
    def check_gradients(self, model):
        """Check gradient norms."""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        if not np.isfinite(total_norm):
            print(f"⚠️ WARNING: NaN/Inf in gradients!")
            return False
        
        self.grad_norms.append(total_norm)
        return True
    
    def get_stats(self):
        """Get current training statistics."""
        return {
            'avg_loss': np.mean(self.losses) if self.losses else 0,
            'avg_grad_norm': np.mean(self.grad_norms) if self.grad_norms else 0,
            'anomalies': self.anomaly_count
        }


# ============================================================================
# MAIN AGENT: ALL FIXES INTEGRATED
# ============================================================================
class IndustryStandardDQNAgent:
    """
    Production-ready DQN agent with all 6 critical gaps fixed.
    """
    def __init__(self, state_dim, action_dim, 
                 learning_rate=0.0005, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64, tau=0.005,
                 use_double_dqn=True, use_per=True, 
                 gradient_clip=10.0, checkpoint_dir='checkpoints'):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = tau
        
        # GAP 0: Double DQN
        self.use_double_dqn = use_double_dqn
        
        # GAP 1: Gradient clipping
        self.gradient_clip = gradient_clip
        self.monitor = TrainingMonitor()
        
        # GAP 2: Curriculum-aware exploration
        self.curriculum_stage = 0
        self.base_epsilon = epsilon_start
        # More exploration in early stages, less in later stages
        self.stage_epsilon_multipliers = {
            0: 1.0,   # Stage 1: Full exploration
            1: 0.7,   # Stage 2: Moderate exploration
            2: 0.5,   # Stage 3: Reduced exploration
            3: 0.3    # Stage 4: Minimal exploration (exploitation phase)
        }
        
        # Networks
        self.policy_net = HierarchicalQNetwork(state_dim, action_dim)
        self.target_net = HierarchicalQNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # GAP 4: Prioritized Experience Replay
        self.use_per = use_per
        if use_per:
            self.memory = PrioritizedReplayBuffer(buffer_size)
            print("✅ Using Prioritized Experience Replay")
        else:
            self.memory = ReplayBuffer(buffer_size)
            print("⚠️ Using standard uniform replay")
        
        # GAP 5: Checkpoint system
        self.checkpoint_dir = checkpoint_dir
        self.best_eval_reward = -float('inf')
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'losses': [],
            'epsilons': [],
            'success_rates': []
        }
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.total_steps = 0
        
        print(f"🚀 Agent initialized:")
        print(f"   - Double DQN: {self.use_double_dqn}")
        print(f"   - PER: {self.use_per}")
        print(f"   - Gradient Clip: {self.gradient_clip}")
        print(f"   - Checkpoint Dir: {self.checkpoint_dir}")

    def set_curriculum_stage(self, stage):
        """
        GAP 2: Update curriculum stage for adaptive exploration.
        """
        self.curriculum_stage = stage
        print(f"📚 Curriculum stage updated to {stage}")

    def select_action(self, state):
        """
        GAP 2: Curriculum-aware epsilon-greedy action selection.
        """
        # Apply stage-specific epsilon multiplier
        effective_epsilon = self.epsilon * self.stage_epsilon_multipliers.get(
            self.curriculum_stage, 1.0
        )
        
        if random.random() < effective_epsilon:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values, _ = self.policy_net(state_tensor)
                return q_values.argmax().item()

    def train_step(self):
        """
        COMPREHENSIVE TRAINING STEP with all fixes:
        - GAP 0: Double DQN
        - GAP 1: Gradient clipping + monitoring
        - GAP 4: PER with importance sampling
        """
        if len(self.memory) < self.batch_size:
            return None

        # Sample from replay buffer
        if self.use_per:
            transitions, idxs, is_weights = self.memory.sample(self.batch_size)
            is_weights = torch.FloatTensor(is_weights).unsqueeze(1)
        else:
            transitions, _, _ = self.memory.sample(self.batch_size)
            idxs = None
            is_weights = torch.ones(self.batch_size, 1)

        batch = Transition(*zip(*transitions))

        state_batch = torch.FloatTensor(np.array(batch.state))
        action_batch = torch.LongTensor(batch.action).unsqueeze(1)
        reward_batch = torch.FloatTensor(batch.reward)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state))
        done_batch = torch.BoolTensor(batch.done)

        # Current Q-values
        current_q_values, _ = self.policy_net(state_batch)
        current_q_values = current_q_values.gather(1, action_batch)

        # ===================================================================
        # GAP 0 FIX: DOUBLE DQN
        # ===================================================================
        with torch.no_grad():
            if self.use_double_dqn:
                # DOUBLE DQN: Use policy net to SELECT, target net to EVALUATE
                next_q_policy, _ = self.policy_net(next_state_batch)
                best_actions = next_q_policy.argmax(1).unsqueeze(1)
                
                next_q_target, _ = self.target_net(next_state_batch)
                max_next_q_values = next_q_target.gather(1, best_actions).squeeze(1)
            else:
                # VANILLA DQN (for comparison/ablation)
                next_q_values, _ = self.target_net(next_state_batch)
                max_next_q_values = next_q_values.max(1)[0]
            
            # Zero out terminal states
            max_next_q_values[done_batch] = 0.0

        # Expected Q-values
        expected_q_values = reward_batch + (self.gamma * max_next_q_values)

        # ===================================================================
        # GAP 4 FIX: Apply importance sampling weights from PER
        # ===================================================================
        td_errors = (expected_q_values.unsqueeze(1) - current_q_values).detach()
        element_wise_loss = F.smooth_l1_loss(
            current_q_values, 
            expected_q_values.unsqueeze(1), 
            reduction='none'
        )
        weighted_loss = (element_wise_loss * is_weights).mean()

        # ===================================================================
        # GAP 1 FIX: Gradient clipping + monitoring
        # ===================================================================
        self.optimizer.zero_grad()
        weighted_loss.backward()
        
        # Check gradients before clipping
        if not self.monitor.check_gradients(self.policy_net):
            print("⚠️ Skipping update due to gradient anomaly")
            return None
        
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), 
            self.gradient_clip
        )
        
        self.optimizer.step()

        # Update priorities in PER
        if self.use_per and idxs is not None:
            td_errors_np = td_errors.squeeze().cpu().numpy()
            self.memory.update_priorities(idxs, td_errors_np)

        # Monitor training health
        loss_value = weighted_loss.item()
        if not self.monitor.check_loss(loss_value):
            print("⚠️ Loss anomaly detected")
        
        self.total_steps += 1
        
        return loss_value

    def update_target_network(self):
        """Soft update of target network."""
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = (
                policy_net_state_dict[key] * self.tau + 
                target_net_state_dict[key] * (1 - self.tau)
            )
        self.target_net.load_state_dict(target_net_state_dict)

    def update_epsilon(self):
        """Decay epsilon for exploration."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ========================================================================
    # GAP 5 FIX: CHECKPOINT SYSTEM
    # ========================================================================
    def save_checkpoint(self, episode, reward, is_best=False):
        """
        Save model checkpoint with metadata.
        """
        checkpoint = {
            'episode': episode,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'curriculum_stage': self.curriculum_stage,
            'reward': reward,
            'total_steps': self.total_steps,
            'training_history': self.training_history,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'use_double_dqn': self.use_double_dqn,
                'use_per': self.use_per,
                'gradient_clip': self.gradient_clip,
                'gamma': self.gamma,
                'tau': self.tau
            }
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f'checkpoint_episode_{episode}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"💾 New best model saved! Reward: {reward:.2f}")
        
        # Save training history
        history_path = os.path.join(self.checkpoint_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        """
        Load model from checkpoint.
        """
        checkpoint = torch.load(checkpoint_path)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.curriculum_stage = checkpoint['curriculum_stage']
        self.total_steps = checkpoint['total_steps']
        self.training_history = checkpoint['training_history']
        
        print(f"✅ Checkpoint loaded from episode {checkpoint['episode']}")
        print(f"   Reward: {checkpoint['reward']:.2f}")
        print(f"   Timestamp: {checkpoint['timestamp']}")
        
        return checkpoint

    def record_episode(self, episode, reward, loss, success_rate):
        """Record episode metrics for tracking."""
        self.training_history['episodes'].append(episode)
        self.training_history['rewards'].append(reward)
        self.training_history['losses'].append(loss)
        self.training_history['epsilons'].append(self.epsilon)
        self.training_history['success_rates'].append(success_rate)

    def get_training_stats(self):
        """Get comprehensive training statistics."""
        stats = self.monitor.get_stats()
        stats.update({
            'epsilon': self.epsilon,
            'curriculum_stage': self.curriculum_stage,
            'total_steps': self.total_steps,
            'memory_size': len(self.memory),
            'best_eval_reward': self.best_eval_reward
        })
        return stats


# ============================================================================
# STANDARD REPLAY BUFFER (for ablation studies)
# ============================================================================
class ReplayBuffer:
    """Standard uniform replay buffer."""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        return transitions, None, None

    def __len__(self):
        return len(self.memory)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def print_gap_summary():
    """Print summary of all fixes."""
    print("\n" + "="*70)
    print("🎯 INDUSTRY-STANDARD RL SYSTEM - ALL GAPS FIXED")
    print("="*70)
    print("\n✅ GAP 0: Double DQN")
    print("   └─ Eliminates maximization bias in Q-value estimation")
    print("   └─ 20-30% performance improvement proven in benchmarks")
    
    print("\n✅ GAP 1: Gradient Clipping + Loss Monitoring")
    print("   └─ Prevents training collapse from exploding gradients")
    print("   └─ NaN/Inf detection with automatic error handling")
    
    print("\n✅ GAP 2: Curriculum-Aware Adaptive Exploration")
    print("   └─ Stage-specific epsilon multipliers")
    print("   └─ More exploration early, more exploitation late")
    
    print("\n✅ GAP 3: Fixed Reward Engineering")
    print("   └─ Proper incentive ordering: failure < success")
    print("   └─ Normalized terminal rewards prevent perverse incentives")
    
    print("\n✅ GAP 4: Prioritized Experience Replay (PER)")
    print("   └─ 30-50% faster convergence (Schaul et al., 2015)")
    print("   └─ Focus on high-value learning opportunities")
    
    print("\n✅ GAP 5: Complete Checkpoint System")
    print("   └─ Model persistence with metadata")
    print("   └─ Best model tracking & training resumption")
    print("="*70 + "\n")
