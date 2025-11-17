"""
PRODUCTION TRAINING SCRIPT
===========================
Training script with all 6 industry-standard fixes applied.
Ready for deployment with full monitoring and checkpointing.
"""

import random
import numpy as np
import torch
import time
import json
import logging
import csv
from datetime import datetime

from mas.hr_rl.comprehensive_fix import (
    IndustryStandardDQNAgent,
    get_fixed_shaped_reward,
    print_gap_summary
)
from mas.hr_rl.environment import PuzzleEnvironment
from mas.hr_rl.core import get_hierarchical_state_representation, decompose_target


def load_config(config_path='config.json'):
    """Loads configuration from a JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def log_experiment(config, metrics, checkpoint_path):
    """Logs the results of an experiment to a CSV file."""
    with open('experiments.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            int(time.time()),
            datetime.now().isoformat(),
            'Jules',
            'main',
            'PuzzleEnvironment',
            json.dumps(config['agent']),
            42,
            json.dumps(metrics),
            checkpoint_path
        ])


def train_production_agent():
    """
    Production-grade training with all fixes and comprehensive monitoring.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("🚀 PRODUCTION TRAINING - INDUSTRY-STANDARD RL SYSTEM")
    
    print_gap_summary()
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    CONFIG = load_config()
    
    # ========================================================================
    # CURRICULUM DESIGN (from paper)
    # ========================================================================
    curriculum = {
        0: {'name': 'Stage 1: Basics', 'target_range': (5, 10), 'forbidden_count': (1, 2)},
        1000: {'name': 'Stage 2: Intermediate', 'target_range': (12, 18), 'forbidden_count': (2, 3)},
        2000: {'name': 'Stage 3: Advanced', 'target_range': (20, 30), 'forbidden_count': (3, 5)},
        3000: {'name': 'Stage 4: Mixed', 'target_range': (5, 100), 'forbidden_count': (2, 5)}
    }
    
    # ========================================================================
    # INITIALIZE AGENT
    # ========================================================================
    logging.info("📋 Configuration:")
    for key, value in CONFIG.items():
        if isinstance(value, dict):
            logging.info(f"   {key}:")
            for sub_key, sub_value in value.items():
                logging.info(f"     {sub_key}: {sub_value}")
        else:
            logging.info(f"   {key}: {value}")

    agent_config = CONFIG['agent']
    
    agent = IndustryStandardDQNAgent(
        state_dim=CONFIG['state_dim'],
        action_dim=CONFIG['action_dim'],
        learning_rate=agent_config['learning_rate'],
        gamma=agent_config['gamma'],
        epsilon_start=agent_config['epsilon_start'],
        epsilon_end=agent_config['epsilon_end'],
        epsilon_decay=agent_config['epsilon_decay'],
        buffer_size=agent_config['buffer_size'],
        batch_size=agent_config['batch_size'],
        tau=agent_config['tau'],
        use_double_dqn=agent_config['use_double_dqn'],
        use_per=agent_config['use_per'],
        gradient_clip=agent_config['gradient_clip'],
        checkpoint_dir='production_checkpoints'
    )
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    start_time = time.time()
    
    current_curriculum = curriculum[0]
    current_stage = 0
    agent.set_curriculum_stage(current_stage)
    
    # Metrics tracking
    success_count = 0
    recent_rewards = []
    recent_steps = []
    recent_losses = []
    
    for episode in range(CONFIG['training_episodes']):
        
        # ====================================================================
        # CURRICULUM STAGE TRANSITION
        # ====================================================================
        if episode in curriculum:
            current_curriculum = curriculum[episode]
            current_stage = list(curriculum.keys()).index(episode)
            agent.set_curriculum_stage(current_stage)
            logging.info(f"📚 Episode {episode}: {current_curriculum['name']}")
        
        # ====================================================================
        # EPISODE SETUP
        # ====================================================================
        target_range = current_curriculum['target_range']
        target = random.randint(*target_range)
        
        forbidden_count_range = current_curriculum['forbidden_count']
        num_forbidden = random.randint(*forbidden_count_range)
        
        # Ensure we don't sample more forbidden states than exist
        available_states = list(range(1, target))
        num_forbidden = min(num_forbidden, len(available_states))
        
        if num_forbidden > 0:
            forbidden_states = set(random.sample(available_states, num_forbidden))
        else:
            forbidden_states = set()
        
        env = PuzzleEnvironment(
            target=target,
            forbidden_states=forbidden_states,
            max_steps=CONFIG['max_steps_per_episode']
        )
        
        # Goal decomposition
        subgoals = decompose_target(0, target)
        
        # ====================================================================
        # EPISODE EXECUTION
        # ====================================================================
        total_reward = 0
        episode_losses = []
        episode_failed = False
        
        for subgoal_idx, subgoal in enumerate(subgoals):
            if episode_failed:
                break
            
            env.target = subgoal
            subgoal_reached = False
            
            while not subgoal_reached and not episode_failed:
                # Get current state
                env_state = env.get_state()
                state_representation = get_hierarchical_state_representation(
                    current=env_state['current'],
                    target=subgoal,
                    step=env_state['step'],
                    max_steps=env_state['max_steps'],
                    forbidden_states=env_state['forbidden_states'],
                    action_space=env.action_space
                )
                
                # Select action
                action_idx = agent.select_action(state_representation)
                action_val = env.action_space[action_idx]
                
                # Execute action
                current_numeric_state = env.current_state
                next_numeric_state, episode_done, info = env.step(action_val)
                
                # Get next state representation
                next_env_state = env.get_state()
                next_state_representation = get_hierarchical_state_representation(
                    current=next_env_state['current'],
                    target=subgoal,
                    step=next_env_state['step'],
                    max_steps=next_env_state['max_steps'],
                    forbidden_states=next_env_state['forbidden_states'],
                    action_space=env.action_space
                )
                
                # Calculate reward (using fixed reward function)
                reward = get_fixed_shaped_reward(
                    current_numeric_state,
                    next_numeric_state,
                    subgoal,
                    env.current_step,
                    episode_done,
                    info,
                    max_steps=CONFIG['max_steps_per_episode']
                )
                
                # Store transition
                agent.memory.push(
                    state_representation,
                    action_idx,
                    next_state_representation,
                    reward,
                    episode_done
                )
                
                # Train
                loss = agent.train_step()
                if loss is not None:
                    episode_losses.append(loss)
                
                # Update target network periodically
                if agent.total_steps % 200 == 0:
                    agent.update_target_network()
                
                total_reward += reward
                
                # Check episode termination
                if episode_done:
                    if info['status'] == 'SUCCESS' and subgoal_idx == len(subgoals) - 1:
                        success_count += 1
                    else:
                        episode_failed = True
                    break
                
                # Check subgoal completion
                if env.current_state >= subgoal:
                    subgoal_reached = True
        
        # ====================================================================
        # EPISODE CLEANUP
        # ====================================================================
        agent.update_epsilon()
        
        recent_rewards.append(total_reward)
        recent_steps.append(env.current_step)
        if episode_losses:
            recent_losses.append(np.mean(episode_losses))
        
        # Keep only recent history
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)
            recent_steps.pop(0)
        if len(recent_losses) > 100:
            recent_losses.pop(0)
        
        # ====================================================================
        # LOGGING
        # ====================================================================
        if (episode + 1) % CONFIG['monitoring']['log_interval'] == 0:
            avg_reward = np.mean(recent_rewards[-10:]) if recent_rewards else 0
            avg_steps = np.mean(recent_steps[-10:]) if recent_steps else 0
            avg_loss = np.mean(recent_losses[-10:]) if recent_losses else 0
            success_rate = success_count / (episode + 1) * 100
            
            stats = agent.get_training_stats()
            
            logging.info(f"📊 Episode {episode + 1}/{CONFIG['training_episodes']} | Reward: {total_reward:.2f} (avg: {avg_reward:.2f}) | Steps: {env.current_step} (avg: {avg_steps:.1f}) | Loss: {avg_loss:.4f} | Success Rate: {success_rate:.1f}% | Epsilon: {agent.epsilon:.4f} | Memory: {len(agent.memory)}/{agent_config['buffer_size']} | Training Steps: {agent.total_steps}")
        
        # ====================================================================
        # CHECKPOINTING
        # ====================================================================
        if (episode + 1) % CONFIG['monitoring']['checkpoint_interval'] == 0:
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            is_best = avg_reward > agent.best_eval_reward
            
            if is_best:
                agent.best_eval_reward = avg_reward
            
            success_rate = success_count / (episode + 1)
            agent.record_episode(episode + 1, avg_reward, avg_loss, success_rate)
            
            checkpoint_path = agent.save_checkpoint(
                episode=episode + 1,
                reward=avg_reward,
                is_best=is_best
            )
            
            logging.info(f"💾 Checkpoint saved: {checkpoint_path}")
    
    # ========================================================================
    # TRAINING COMPLETE
    # ========================================================================
    training_time = time.time() - start_time
    
    logging.info("🎉 TRAINING COMPLETED")
    logging.info(f"Total time: {training_time / 60:.2f} minutes")
    logging.info(f"Total episodes: {CONFIG['training_episodes']}")
    logging.info(f"Total steps: {agent.total_steps}")
    logging.info(f"Final success rate: {success_count / CONFIG['training_episodes'] * 100:.1f}%")
    logging.info(f"Best evaluation reward: {agent.best_eval_reward:.2f}")
    logging.info(f"Final epsilon: {agent.epsilon:.4f}")
    
    # Get final training stats
    final_stats = agent.get_training_stats()
    logging.info("📈 Final Statistics:")
    for key, value in final_stats.items():
        logging.info(f"   {key}: {value}")

    log_experiment(CONFIG, final_stats, f"production_checkpoints/checkpoint_episode_{CONFIG['training_episodes']}.pt")
    
    return agent


def evaluate_production_agent(agent):
    """
    Comprehensive evaluation on test cases.
    """
    logging.info("🎯 EVALUATION - Test Cases from Paper")
    
    test_cases = [
        {"name": "Easy", "target": 123, "forbidden": {23, 45, 67, 89}},
        {"name": "Medium", "target": 278, "forbidden": {51, 102, 177, 203, 234}},
        {"name": "Hard", "target": 431, "forbidden": {78, 156, 234, 312, 389}}
    ]
    
    agent.epsilon = 0.0  # Greedy evaluation
    
    results = []
    
    for case in test_cases:
        target = case['target']
        forbidden = case['forbidden']
        
        env = PuzzleEnvironment(
            target=target,
            forbidden_states=forbidden,
            max_steps=target * 2
        )
        env.reset()
        
        path = [0]
        subgoals = decompose_target(0, target)
        evaluation_failed = False
        
        for subgoal in subgoals:
            env.target = subgoal
            subgoal_reached = False
            
            while not subgoal_reached and not evaluation_failed:
                env_state = env.get_state()
                state_representation = get_hierarchical_state_representation(
                    current=env_state['current'],
                    target=subgoal,
                    step=env_state['step'],
                    max_steps=env_state['max_steps'],
                    forbidden_states=env_state['forbidden_states'],
                    action_space=env.action_space
                )
                
                action_idx = agent.select_action(state_representation)
                action_val = env.action_space[action_idx]
                
                next_state, is_terminal, info = env.step(action_val)
                path.append(next_state)
                
                if is_terminal and info['status'] != 'SUCCESS':
                    evaluation_failed = True
                    break
                
                if env.current_state >= subgoal:
                    subgoal_reached = True
            
            if evaluation_failed:
                break
        
        success = env.current_state == target and not evaluation_failed
        
        result = {
            'name': case['name'],
            'target': target,
            'success': success,
            'steps': env.current_step,
            'path_length': len(path),
            'final_state': env.current_state
        }
        results.append(result)
        
        logging.info(f"Test Case: {case['name']} | Result: {'✅ SUCCESS' if success else '❌ FAILURE'} | Steps: {env.current_step} | Final State: {env.current_state}")
    
    # Summary
    success_count = sum(1 for r in results if r['success'])
    logging.info("📊 EVALUATION SUMMARY")
    logging.info(f"Success Rate: {success_count}/{len(test_cases)} ({success_count/len(test_cases)*100:.1f}%)")
    logging.info(f"Average Steps: {np.mean([r['steps'] for r in results]):.1f}")
    
    return results


def run_ablation_study():
    """
    Compare performance with and without each fix.
    """
    logging.info("🔬 ABLATION STUDY - Measuring Impact of Each Fix")
    
    configs = [
        {'name': 'All Fixes', 'double_dqn': True, 'per': True, 'clip': 10.0},
        {'name': 'No Double DQN', 'double_dqn': False, 'per': True, 'clip': 10.0},
        {'name': 'No PER', 'double_dqn': True, 'per': False, 'clip': 10.0},
        {'name': 'No Clipping', 'double_dqn': True, 'per': True, 'clip': None},
        {'name': 'Baseline (No Fixes)', 'double_dqn': False, 'per': False, 'clip': None},
    ]
    
    logging.info("⚠️ Note: Full ablation study requires ~30 minutes.")
    logging.info("Running quick comparison (100 episodes each)...")
    
    results = {}
    
    for config in configs:
        logging.info(f"📝 Testing: {config['name']}")
        
        agent = IndustryStandardDQNAgent(
            state_dim=12,
            action_dim=3,
            use_double_dqn=config['double_dqn'],
            use_per=config['per'],
            gradient_clip=config['clip'] if config['clip'] else 1000.0
        )
        
        rewards = []
        
        for episode in range(100):
            target = random.randint(10, 30)
            forbidden = set(random.sample(range(1, target), min(2, target-1)))
            
            env = PuzzleEnvironment(target, forbidden, max_steps=100)
            env.reset()
            
            total_reward = 0
            
            while not env.is_done:
                env_state = env.get_state()
                state_rep = get_hierarchical_state_representation(
                    current=env_state['current'],
                    target=env_state['target'],
                    step=env_state['step'],
                    max_steps=env_state['max_steps'],
                    forbidden_states=env_state['forbidden_states'],
                    action_space=env.action_space
                )
                
                action_idx = agent.select_action(state_rep)
                action_val = env.action_space[action_idx]
                
                current_state = env.current_state
                next_state, done, info = env.step(action_val)
                
                next_env_state = env.get_state()
                next_state_rep = get_hierarchical_state_representation(
                    current=next_env_state['current'],
                    target=next_env_state['target'],
                    step=next_env_state['step'],
                    max_steps=next_env_state['max_steps'],
                    forbidden_states=next_env_state['forbidden_states'],
                    action_space=env.action_space
                )
                
                reward = get_fixed_shaped_reward(
                    current_state, next_state, target,
                    env.current_step, done, info
                )
                
                agent.memory.push(state_rep, action_idx, next_state_rep, reward, done)
                agent.train_step()
                
                total_reward += reward
            
            rewards.append(total_reward)
            agent.update_epsilon()
        
        avg_reward = np.mean(rewards)
        results[config['name']] = avg_reward
        
        logging.info(f"Average Reward: {avg_reward:.2f}")
    
    # Print comparison
    logging.info("📊 ABLATION RESULTS")
    
    baseline = results.get('Baseline (No Fixes)', 0)
    
    for name, reward in sorted(results.items(), key=lambda x: x[1], reverse=True):
        improvement = ((reward - baseline) / abs(baseline) * 100) if baseline != 0 else 0
        logging.info(f"{name:30s}: {reward:8.2f}  ({improvement:+.1f}%)")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Production RL Training')
    parser.add_argument('--mode', choices=['train', 'eval', 'ablation', 'all'],
                       default='all', help='Execution mode')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint path for evaluation')
    
    args = parser.parse_args()
    
    if args.mode in ['train', 'all']:
        trained_agent = train_production_agent()
        
        if args.mode in ['eval', 'all']:
            evaluate_production_agent(trained_agent)
    
    elif args.mode == 'eval' and args.checkpoint:
        agent = IndustryStandardDQNAgent(
            state_dim=12,
            action_dim=3
        )
        agent.load_checkpoint(args.checkpoint)
        evaluate_production_agent(agent)
    
    elif args.mode == 'ablation':
        run_ablation_study()
    
    logging.info("✅ Production run complete!")
