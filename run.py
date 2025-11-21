import random
import numpy as np
import torch
import time

from agent import HierarchicalAgent
from environment import PuzzleEnvironment
from core import get_hierarchical_state_representation, get_shaped_reward

def train_agent():
    """Main function to run the training and evaluation."""
    print("Starting Hierarchical Relational Reinforcement Learning training...")

    # Training configuration from the paper
    TRAINING_EPISODES = 5000
    STATE_DIM = 12  # Based on our state representation
    ACTION_DIM = 3  # Actions: 1, 3, 5
    MAX_STEPS_PER_EPISODE = 200 # A reasonable limit

    agent = HierarchicalAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)

    # Curriculum Design from the paper
    curriculum = {
        1: (5, 10),
        1000: (12, 18),
        2000: (20, 30),
        3000: (5, 100) # Stage 4: Mixed training on full range
    }

    start_time = time.time()
    
    current_stage = 1
    total_steps = 0
    for episode in range(TRAINING_EPISODES):
        if episode + 1 in curriculum:
            current_stage = episode + 1
            print(f"Episode {episode + 1}: Entering curriculum stage with targets {curriculum[current_stage]}")

        # Episode setup
        target_range = curriculum[current_stage]
        target = random.randint(*target_range)
        num_forbidden = random.randint(2, 5)
        forbidden_states = set(random.sample(range(1, target), min(num_forbidden, target - 1)))
        
        env = PuzzleEnvironment(target=target, forbidden_states=forbidden_states, max_steps=MAX_STEPS_PER_EPISODE)
        
        subgoals = agent.decompose_target_hierarchically(0, target, forbidden_states)
        total_reward = 0
        
        # This variable will track if the episode ends due to failure
        episode_failed = False

        for subgoal in subgoals:
            # If the episode failed on a previous subgoal, stop.
            if episode_failed:
                break

            env.target = subgoal  # Aim for the current subgoal
            
            subgoal_reached = False
            while not subgoal_reached:
                current_numeric_state = env.current_state
                env_state = env.get_state()

                state_representation = get_hierarchical_state_representation(
                    current=env_state['current'], target=subgoal,
                    step=env_state['step'], max_steps=env_state['max_steps'],
                    forbidden_states=env_state['forbidden_states'], action_space=env.action_space
                )
                
                valid_actions = env.get_valid_actions()
                action_idx = agent.select_action(state_representation, valid_actions)
                action_val = env.action_space[action_idx]

                # The environment tells us if the episode is terminal
                next_numeric_state, episode_done, info = env.step(action_val)
                
                next_env_state = env.get_state()
                next_state_representation = get_hierarchical_state_representation(
                    current=next_env_state['current'], target=subgoal,
                    step=next_env_state['step'], max_steps=next_env_state['max_steps'],
                    forbidden_states=next_env_state['forbidden_states'], action_space=env.action_space
                )

                reward = get_shaped_reward(current_numeric_state, next_numeric_state, subgoal, env.current_step, episode_done, info)
                
                agent.memory.push(state_representation, action_idx, next_state_representation, reward, episode_done)
                agent.train_step()
                total_reward += reward
                total_steps += 1

                if total_steps % 200 == 0:
                    agent.update_target_network()
                
                if episode_done:
                    episode_failed = True
                    break # Exit subgoal loop

                # Check if subgoal is reached
                if env.current_state >= subgoal:
                    subgoal_reached = True

        agent.update_epsilon()

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{TRAINING_EPISODES}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time / 60:.2f} minutes.")
    return agent

def evaluate_agent(agent):
    """Evaluates the trained agent on the test cases from the paper."""
    print("\n--- Evaluating Agent ---")
    
    test_cases = [
        {"target": 123, "forbidden": {23, 45, 67, 89}},
        {"target": 278, "forbidden": {51, 102, 177, 203, 234}},
        {"target": 431, "forbidden": {78, 156, 234, 312, 389}}
    ]

    agent.epsilon = 0.0 # Set to greedy mode for evaluation

    for case in test_cases:
        target = case['target']
        forbidden = case['forbidden']
        
        env = PuzzleEnvironment(target=target, forbidden_states=forbidden, max_steps=target * 2)
        env.reset()
        
        path = [0]
        final_state = 0
        evaluation_failed = False
        
        subgoals = agent.decompose_target_hierarchically(0, target, forbidden)

        for subgoal in subgoals:
            env.target = subgoal
            
            subgoal_reached = False
            while not subgoal_reached:
                env_state = env.get_state()
                state_representation = get_hierarchical_state_representation(
                    current=env_state['current'], target=subgoal,
                    step=env_state['step'], max_steps=env_state['max_steps'],
                    forbidden_states=env_state['forbidden_states'], action_space=env.action_space
                )
                
                valid_actions = env.get_valid_actions()
                action_idx = agent.select_action(state_representation, valid_actions)
                action_val = env.action_space[action_idx]
                
                next_state, is_terminal, info = env.step(action_val)
                path.append(next_state)
                final_state = next_state

                if is_terminal and info['status'] != 'SUCCESS':
                    evaluation_failed = True
                    break

                if env.current_state >= subgoal:
                    subgoal_reached = True
            
            if evaluation_failed:
                break

        success = final_state == target
        print(f"\nTarget: {target}, Forbidden: {forbidden}")
        print(f"Result: {'Success' if success else 'Failure'} in {env.current_step} steps.")
        print(f"Path taken: {' -> '.join(map(str, path))}")


if __name__ == "__main__":
    trained_agent = train_agent()
    evaluate_agent(trained_agent)
