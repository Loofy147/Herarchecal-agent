# Runbook

This document provides operational guidance for the Hierarchical Reinforcement Learning (RL) agent project.

## 1. System Overview

-   **Purpose**: To train and evaluate a hierarchical RL agent for solving mathematical puzzles.
-   **Key Components**: `PuzzleEnvironment`, `HierarchicalQNetwork`, `IndustryStandardDQNAgent`.
-   **Primary Entrypoint**: `production_run.py`.

## 2. Common Incidents and Playbooks

### Incident: Training Fails with `NaN` Loss

-   **Symptom**: The training script crashes or logs `NaN` values for the loss.
-   **Probable Cause**: This is often due to exploding gradients, which can be caused by unstable reward signals or a high learning rate.
-   **Playbook**:
    1.  **Check the Logs**: Review the structured logs to identify the point at which the `NaN` values first appeared.
    2.  **Review Configuration**: Check the `learning_rate` in `config.json`. If it has been recently increased, consider lowering it.
    3.  **Check Gradient Clipping**: Ensure that `gradient_clip` is enabled and set to a reasonable value in `config.json`.
    4.  **Analyze Reward Function**: Examine the `get_fixed_shaped_reward` function in `mas/hr_rl/comprehensive_fix.py`. A recent change may have introduced instability.

### Incident: Low Training Success Rate

-   **Symptom**: The agent's success rate is not improving or is significantly lower than the target SLO (> 90%).
-   **Probable Cause**: This could be due to a number of factors, including suboptimal hyperparameters, a flawed reward function, or an issue with the training curriculum.
-   **Playbook**:
    1.  **Review Experiment Registry**: Check the `experiments.csv` file to see if a recent change to the configuration has resulted in a drop in performance.
    2.  **Tune Hyperparameters**: Adjust the hyperparameters in `config.json`, such as `learning_rate`, `gamma`, and the epsilon decay schedule.
    3.  **Analyze Curriculum**: Review the curriculum stages defined in `production_run.py`. The difficulty may be increasing too quickly for the agent to learn effectively.

## 3. On-Call Responsibilities

-   **Primary On-Call**: The engineer who most recently made changes to the `mas/hr_rl/` directory.
-   **Secondary On-Call**: The tech lead for the project.

## 4. Secrets Management

-   **Secrets**: This project does not currently handle any secrets. If, in the future, it is integrated with a service that requires API keys or other credentials, they should be managed using a secure secrets management system (e.g., HashiCorp Vault, AWS Secrets Manager).
-   **Rotation**: Any secrets should be rotated on a regular basis (e.g., every 90 days).
