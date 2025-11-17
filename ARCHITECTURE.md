# Architecture

This document describes the architecture of the Hierarchical Reinforcement Learning (RL) agent project.

## 1. Component Diagram

The system is composed of the following key components:

-   **`PuzzleEnvironment`**: A custom environment that simulates the mathematical puzzle. It manages the state, actions, and rewards.
-   **`HierarchicalQNetwork`**: A PyTorch neural network that serves as the function approximator for the agent. It features a phase-adaptive architecture with a shared feature extractor and three specialized heads for different problem-solving phases.
-   **`DQNAgent` / `IndustryStandardDQNAgent`**: The RL agents responsible for learning and decision-making. These agents include implementations of Double DQN, Prioritized Experience Replay (PER), and other industry-standard enhancements.
-   **Core Logic (`core.py`)**: A collection of pure functions that provide key services, such as state representation, reward shaping, and goal decomposition.
-   **Training Script (`production_run.py`)**: The main entry point for training and evaluating the agent. It manages the training loop, curriculum learning, and experiment execution.

```
+---------------------------+       +--------------------------------+
|     Training Script       |------>|   IndustryStandardDQNAgent     |
| (production_run.py)       |       | (comprehensive_fix.py)         |
+---------------------------+       +--------------------------------+
             |                                    /|\\
             |                                     |
             |                                     |
             v                                     |
+---------------------------+       +--------------------------------+
|    PuzzleEnvironment      |<----->|      HierarchicalQNetwork      |
|    (environment.py)       |       |           (model.py)           |
+---------------------------+       +--------------------------------+
             |                                     /|\\
             |                                     |
             |                                     |
             v                                     |
+---------------------------+                      |
|        Core Logic         |<---------------------+
|        (core.py)          |
+---------------------------+

```

## 2. Data and Control Flow

1.  **Initialization**:
    -   The `production_run.py` script initializes the `PuzzleEnvironment` and the `IndustryStandardDQNAgent`.
    -   The agent, in turn, initializes the `HierarchicalQNetwork` (both a policy and a target network) and a replay buffer.

2.  **Training Loop**:
    -   The training script manages a curriculum of increasing difficulty.
    -   For each episode, it resets the environment and begins the training loop.
    -   At each step, the agent receives the current state from the environment.
    -   The `get_hierarchical_state_representation` function from `core.py` is called to create a feature vector from the raw state.
    -   The agent's `select_action` method uses the policy network to choose an action based on the state vector (using an epsilon-greedy strategy).
    -   The chosen action is sent to the environment's `step` method, which returns the next state, a reward, and a `done` flag.
    -   The `get_fixed_shaped_reward` function from `comprehensive_fix.py` is used to calculate a shaped reward that encourages efficient learning.
    -   The transition (state, action, next_state, reward, done) is stored in the agent's replay buffer.
    -   The agent samples a batch of transitions from the replay buffer and performs a training step, updating the policy network's weights.

3.  **Shutdown**:
    -   The training loop continues for a configured number of episodes.
    -   Throughout the process, the agent saves checkpoints of the model's state to disk, allowing for resumption of training and for later evaluation.

## 3. Key Design Principles

-   **Modularity**: The separation of the environment, agent, model, and core logic into distinct components allows for easier maintenance and testing.
-   **Hierarchical Reinforcement Learning**: The use of goal decomposition in `core.py` and a hierarchical agent structure allows the system to tackle more complex problems than a traditional "flat" RL agent could.
-   **Config-over-Code**: (To be implemented) The system will be refactored to load all hyperparameters from an external configuration file, enabling more flexible experimentation.
-   **Observability**: (To be implemented) The system will be instrumented with structured logging to provide better insights into its behavior during training and evaluation.
