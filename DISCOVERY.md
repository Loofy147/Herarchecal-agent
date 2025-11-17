# Discovery Brief

This document summarizes the findings from the discovery phase for the Hierarchical Reinforcement Learning (RL) agent project.

## 1. Summary of Findings

The project is a Hierarchical Reinforcement Learning system designed to solve a mathematical puzzle. It is implemented in Python and uses PyTorch for its neural network components.

-   **Architecture**: The system uses a hierarchical agent with a phase-adaptive neural network. It decomposes complex problems into subgoals and uses specialized network heads for different problem-solving phases (exploration, navigation, precision).
-   **Code Quality**: The codebase was recently refactored from a flat structure into a modular Python package (`mas.hr_rl`). However, it still has several areas for improvement:
    -   **Configuration**: Hyperparameters and settings are hardcoded directly into the training scripts (`production_run.py`), making it difficult to run experiments without modifying the code.
    -   **Observability**: The system relies on `print()` statements for output, which is insufficient for production-grade monitoring and debugging. Structured logging is needed.
    -   **Testing**: While the project has a good set of integration-style tests in `red_team_tests (1).py`, it lacks granular unit tests for core logic, particularly for the functions in `core.py`.
-   **Dependencies**: The project depends on `numpy`, `torch`, and `pytest`, which are now listed in a `requirements.txt` file.

## 2. Success Criteria & SLOs

The following success criteria are inferred from the project's goal of creating a production-ready RL agent:

| Metric                  | Description                                                                                              | Threshold        |
| ----------------------- | -------------------------------------------------------------------------------------------------------- | ---------------- |
| **Training Success Rate** | The agent must achieve a high success rate on the "Hard" evaluation case after a full training cycle.  | > 90%            |
| **Training Performance**  | A full training run of 5000 episodes should complete within a reasonable time on a standard CPU.       | < 15 minutes     |
| **Experiment Reproducibility** | Given the same config and seed, final agent performance should be consistent.                          | < 5% variance    |
| **Maintainability**       | A new engineer can set up the project and run all tests with minimal effort.                             | < 10 minutes     |

## 3. Risk Register

| Risk ID | Title                           | Impact | Probability | Owner      | Mitigation                                                                                                      | Status |
| ------- | ------------------------------- | ------ | ----------- | ---------- | --------------------------------------------------------------------------------------------------------------- | ------ |
| R-01    | Hardcoded Configuration         | High   | High        | Tech Lead  | Externalize all hyperparameters and environment settings into a dedicated config file (e.g., `config.json`).      | Open   |
| R-02    | Lack of Observability           | High   | High        | Engineer   | Replace all `print()` statements with a structured logging library to enable proper monitoring and debugging.     | Open   |
| R-03    | Incomplete Test Coverage        | Medium | High        | Engineer   | Implement unit tests for critical business logic in `core.py` to ensure correctness and prevent regressions.    | Open   |
| R-04    | Lack of Performance Benchmarks  | Medium | Medium      | Engineer   | Create a benchmarking script to measure training performance and track regressions.                             | Open   |
| R-05    | No CI/CD Pipeline               | High   | High        | SRE/Ops    | Implement a CI pipeline to automate linting, testing, and vulnerability scanning on every commit.             | Open   |

## 4. Recommended Direction

The immediate priority is to address the risks outlined above to improve the project's maintainability, observability, and reliability. The recommended direction is to proceed with the following:

1.  **Externalize Configuration**: Move all hardcoded settings to a `config.json` file.
2.  **Implement Structured Logging**: Integrate Python's `logging` module throughout the codebase.
3.  **Expand Test Coverage**: Add unit tests for the `core.py` module.
4.  **Establish a CI Pipeline**: Create a basic CI workflow to automate quality checks.
5.  **Create an Experiment Registry**: To ensure reproducibility and track results.
