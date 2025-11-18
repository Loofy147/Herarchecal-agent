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
from datetime import datetime

from comprehensive_fix import (
    IndustryStandardDQNAgent,
    get_fixed_shaped_reward,
    print_gap_summary
)
from environment import PuzzleEnvironment
from core import get_hierarchical_state_representation, decompose_target


def train_production_agent():
    """
    Production-grade training with all fixes and comprehensive monitoring.
    """
    print("\n" + "="*70)
    print("🚀 PRODUCTION TRAINING - INDUSTRY-STANDARD RL SYSTEM")
    print("="*70)
    
    print_gap_summary()
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    CONFIG = {
        'training_episodes': 5000,
        'state_dim': 12,
        'action_dim': 3,
        'max_steps_per_episode': 200,
        
        # Agent hyperparameters (tuned for production)
        'learning_rate': 0.0005,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'buffer_size': 50000,  # Larger buffer for better PER
        'batch_size': 64,
        'tau': 0.005,
        
        # Feature flags (all enabled for production)
        'use_double_dqn': True,
        'use_per': True,
        'gradient_clip': 10.0,
        
        # Monitoring & checkpointing
        'checkpoint_interval': 100,  # Save every 100 episodes
        'eval_interval': 50,  # Evaluate every 50 episodes
        'log_interval': 10,   # Log every 10 episodes
    }
    
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
    print(f"\n📋 Configuration:")
    for key, value in CONFIG.items():
        print(f"   {key}: {value}")
    
    agent = IndustryStandardDQNAgent(
        state_dim=CONFIG['state_dim'],
        action_dim=CONFIG['action_dim'],
        learning_rate=CONFIG['learning_rate'],
        gamma=CONFIG['gamma'],
        epsilon_start=CONFIG['epsilon_start'],
        epsilon_end=CONFIG['epsilon_end'],
        epsilon_decay=CONFIG['epsilon_decay'],
        buffer_size=CONFIG['buffer_size'],
        batch_size=CONFIG['batch_size'],
        tau=CONFIG['tau'],
        use_double_dqn=CONFIG['use_double_dqn'],
        use_per=CONFIG['use_per'],
        gradient_clip=CONFIG['gradient_clip'],
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
            print(f"\n{'='*70}")
            print(f"📚 Episode {episode}: {current_curriculum['name']}")
            print(f"{'='*70}")
        
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
        if (episode + 1) % CONFIG['log_interval'] == 0:
            avg_reward = np.mean(recent_rewards[-10:]) if recent_rewards else 0
            avg_steps = np.mean(recent_steps[-10:]) if recent_steps else 0
            avg_loss = np.mean(recent_losses[-10:]) if recent_losses else 0
            success_rate = success_count / (episode + 1) * 100
            
            stats = agent.get_training_stats()
            
            print(f"\n📊 Episode {episode + 1}/{CONFIG['training_episodes']}")
            print(f"   Reward: {total_reward:.2f} (avg: {avg_reward:.2f})")
            print(f"   Steps: {env.current_step} (avg: {avg_steps:.1f})")
            print(f"   Loss: {avg_loss:.4f}")
            print(f"   Success Rate: {success_rate:.1f}%")
            print(f"   Epsilon: {agent.epsilon:.4f}")
            print(f"   Memory: {len(agent.memory)}/{CONFIG['buffer_size']}")
            print(f"   Training Steps: {agent.total_steps}")
        
        # ====================================================================
        # CHECKPOINTING
        # ====================================================================
        if (episode + 1) % CONFIG['checkpoint_interval'] == 0:
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
            
            print(f"\n💾 Checkpoint saved: {checkpoint_path}")
    
    # ========================================================================
    # TRAINING COMPLETE
    # ========================================================================
    training_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETED")
    print("="*70)
    print(f"Total time: {training_time / 60:.2f} minutes")
    print(f"Total episodes: {CONFIG['training_episodes']}")
    print(f"Total steps: {agent.total_steps}")
    print(f"Final success rate: {success_count / CONFIG['training_episodes'] * 100:.1f}%")
    print(f"Best evaluation reward: {agent.best_eval_reward:.2f}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    
    # Get final training stats
    final_stats = agent.get_training_stats()
    print(f"\n📈 Final Statistics:")
    for key, value in final_stats.items():
        print(f"   {key}: {value}")
    
    return agent


def evaluate_production_agent(agent):
    """
    Comprehensive evaluation on test cases.
    """
    print("\n" + "="*70)
    print("🎯 EVALUATION - Test Cases from Paper")
    print("="*70)
    
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
        
        print(f"\n{'='*70}")
        print(f"Test Case: {case['name']}")
        print(f"Target: {target}, Forbidden: {forbidden}")
        print(f"Result: {'✅ SUCCESS' if success else '❌ FAILURE'}")
        print(f"Steps: {env.current_step}")
        print(f"Final State: {env.current_state}")
        print(f"Path: {' → '.join(map(str, path[:20]))}")
        if len(path) > 20:
            print(f"      ... ({len(path) - 20} more steps)")
    
    # Summary
    success_count = sum(1 for r in results if r['success'])
    print(f"\n{'='*70}")
    print(f"📊 EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"Success Rate: {success_count}/{len(test_cases)} ({success_count/len(test_cases)*100:.1f}%)")
    print(f"Average Steps: {np.mean([r['steps'] for r in results]):.1f}")
    
    return results


def run_ablation_study():
    """
    Compare performance with and without each fix.
    """
    print("\n" + "="*70)
    print("🔬 ABLATION STUDY - Measuring Impact of Each Fix")
    print("="*70)
    
    configs = [
        {'name': 'All Fixes', 'double_dqn': True, 'per': True, 'clip': 10.0},
        {'name': 'No Double DQN', 'double_dqn': False, 'per': True, 'clip': 10.0},
        {'name': 'No PER', 'double_dqn': True, 'per': False, 'clip': 10.0},
        {'name': 'No Clipping', 'double_dqn': True, 'per': True, 'clip': None},
        {'name': 'Baseline (No Fixes)', 'double_dqn': False, 'per': False, 'clip': None},
    ]
    
    print("\n⚠️ Note: Full ablation study requires ~30 minutes.")
    print("Running quick comparison (100 episodes each)...\n")
    
    results = {}
    
    for config in configs:
        print(f"\n📝 Testing: {config['name']}")
        print("-" * 60)
        
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
        
        print(f"Average Reward: {avg_reward:.2f}")
    
    # Print comparison
    print(f"\n{'='*70}")
    print("📊 ABLATION RESULTS")
    print(f"{'='*70}")
    
    baseline = results.get('Baseline (No Fixes)', 0)
    
    for name, reward in sorted(results.items(), key=lambda x: x[1], reverse=True):
        improvement = ((reward - baseline) / abs(baseline) * 100) if baseline != 0 else 0
        print(f"{name:30s}: {reward:8.2f}  ({improvement:+.1f}%)")
    
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
    
    print("\n✅ Production run complete!")
