import math
import numpy as np

def identify_phase(gap, target):
    """
    Identifies the current problem-solving phase based on the gap to the target.
    Returns a numerical representation of the phase.
    - 0: Exploration (far from target)
    - 1: Navigation (medium distance)
    - 2: Precision (close to target)
    """
    if gap > target * 0.5:
        return 0  # Exploration: Far from target
    elif gap > 10:
        return 1  # Navigation: Medium distance
    else:
        return 2  # Precision: Close to target

def get_hierarchical_state_representation(current, target, step, max_steps, forbidden_states, action_space):
    """
    Computes the multi-scale relational state representation as described in the paper.

    Args:
        current (int): The current number/state.
        target (int): The target number.
        step (int): The current step in the episode.
        max_steps (int): The maximum steps allowed.
        forbidden_states (set): A set of forbidden numbers.
        action_space (list): The list of possible actions.

    Returns:
        np.ndarray: A NumPy array representing the state vector.
    """
    # Scale-invariant core features
    progress_ratio = current / target if target > 0 else 0
    remaining_ratio = (target - current) / target if target > 0 else 0
    time_ratio = step / max_steps if max_steps > 0 else 0
    
    # Multi-scale gap analysis
    gap = abs(target - current)
    log_gap = np.log(gap + 1) / np.log(target + 1) if target > 0 else 0
    gap_magnitude = gap / target if target > 0 else 0
    
    # Strategic features
    is_close = 1.0 if gap <= 10 else 0.0
    is_far = 1.0 if gap >= target * 0.5 else 0.0
    
    # Constraint features
    min_dist_to_forbidden = min([abs(current - f) for f in forbidden_states]) if forbidden_states else target
    danger_proximity = 1.0 / (min_dist_to_forbidden + 1)
    
    constraints_in_path = sum(1 for f in forbidden_states if current < f < target)
    constraint_pressure = constraints_in_path / (gap + 1)
    
    # Phase identification
    phase = identify_phase(gap, target)
    
    # Efficiency features
    max_action = max(action_space)
    theoretical_min_steps = math.ceil(gap / max_action) if max_action > 0 else float('inf')
    remaining_steps = max_steps - step + 1
    efficiency_ratio = theoretical_min_steps / remaining_steps if remaining_steps > 0 else 0

    state_vector = np.array([
        progress_ratio, remaining_ratio, time_ratio, log_gap, gap_magnitude,
        is_close, is_far, danger_proximity, constraint_pressure, 
        phase, theoretical_min_steps, efficiency_ratio
    ])
    
    return state_vector

def decompose_target(current, target):
    """
    Implements the hierarchical goal decomposition from the paper.

    Args:
        current (int): The current number.
        target (int): The final target number.

    Returns:
        list: A list of subgoals, including the final target.
    """
    gap = abs(target - current)
    
    if gap <= 50:
        return [target]  # Direct approach for small gaps
    
    # Create strategic waypoints
    num_subgoals = max(2, gap // 75)
    step_size = gap // num_subgoals
    direction = 1 if target > current else -1
    
    subgoals = []
    for i in range(1, num_subgoals):
        subgoal = current + (step_size * i * direction)
        subgoals.append(subgoal)
    
    subgoals.append(target)
    return subgoals

def get_shaped_reward(current_state, next_state, target, step, is_done, info):
    """
    Calculates the shaped reward based on the paper's specification.

    Args:
        current_state (int): The state before the action.
        next_state (int): The state after the action.
        target (int): The target number.
        step (int): The current step number.
        is_done (bool): Whether the episode has terminated.
        info (dict): Information about the transition from the environment.

    Returns:
        float: The calculated reward.
    """
    if is_done:
        if info['status'] == 'SUCCESS':
            return 100 - step  # Earlier success = higher reward
        elif info['status'] == 'FORBIDDEN':
            return -50  # Constraint violation penalty
        elif info['status'] == 'OVERSHOOT':
            return -20  # Overshoot penalty
        elif info['status'] == 'MAX_STEPS':
            return -10 # Penalty for running out of time, less severe than overshoot
    
    # Progress reward
    if abs(next_state - target) < abs(current_state - target):
        return 10
    
    # Time penalty for every other step
    return -1
