import random

class PuzzleEnvironment:
    """
    Implements the mathematical puzzle environment as described in the research paper.

    The agent starts at 0 and must reach a target number by applying actions
    from a fixed set, while avoiding a list of forbidden numbers.
    """

    def __init__(self, target, forbidden_states, max_steps):
        """
        Initializes a new puzzle environment for a single episode.

        Args:
            target (int): The target number to reach.
            forbidden_states (set): A set of integers that the agent must avoid.
            max_steps (int): The maximum number of steps allowed for the episode.
        """
        self.final_target = target
        self.target = target # The current (sub)goal
        self.forbidden_states = set(forbidden_states)
        self.max_steps = max_steps
        self.action_space = [1, 3, 5]
        
        self.current_state = 0
        self.current_step = 0
        self.is_done = False

    def reset(self):
        """
        Resets the environment to its initial state.

        Returns:
            int: The initial state of the environment (always 0).
        """
        self.current_state = 0
        self.current_step = 0
        self.is_done = False
        return self.current_state

    def step(self, action):
        """
        Executes a single step in the environment.

        Args:
            action (int): The action to apply, which must be in the action_space.

        Returns:
            tuple: A tuple containing (next_state, done, info), where:
                - next_state (int): The new state after applying the action.
                - done (bool): True if the episode has ended, False otherwise.
                - info (dict): A dictionary with auxiliary information, including
                               'status' which can be 'SUCCESS', 'FORBIDDEN',
                               'MAX_STEPS', or 'IN_PROGRESS'.
        """
        if self.is_done:
            raise Exception("Cannot step in a completed episode. Please reset the environment.")

        if action not in self.action_space:
            raise ValueError(f"Invalid action: {action}. Must be one of {self.action_space}")

        self.current_step += 1
        self.current_state += action

        info = {'status': 'IN_PROGRESS'}
        
        # Check for success on the current target
        if self.current_state == self.target:
            info['status'] = 'SUCCESS'
            # Only terminate if the final target is reached
            if self.current_state == self.final_target:
                self.is_done = True

        # Failure conditions always terminate the episode
        elif self.current_state in self.forbidden_states:
            self.is_done = True
            info['status'] = 'FORBIDDEN'
        elif self.current_step >= self.max_steps:
            self.is_done = True
            info['status'] = 'MAX_STEPS'
        # The overshoot penalty should only apply to the final target
        elif self.current_state > self.final_target:
            self.is_done = True
            info['status'] = 'OVERSHOOT'

        return self.current_state, self.is_done, info

    def get_state(self):
        """Returns the current state of the environment."""
        return {
            'current': self.current_state,
            'target': self.target,
            'step': self.current_step,
            'max_steps': self.max_steps,
            'forbidden_states': self.forbidden_states
        }
