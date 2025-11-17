import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalQNetwork(nn.Module):
    """
    Implements the Phase-Adaptive Neural Architecture from the research paper.

    This network uses a shared feature extractor and three specialized heads
    for different phases of problem-solving: exploration, navigation, and precision.
    The final Q-values are a weighted combination of the outputs of these heads,
    determined by a phase classifier.
    """
    def __init__(self, state_dim=12, action_dim=3, hidden_dim=256):
        """
        Initializes the neural network.

        Args:
            state_dim (int): The dimensionality of the input state vector.
            action_dim (int): The number of possible actions.
            hidden_dim (int): The number of units in the hidden layers.
        """
        super(HierarchicalQNetwork, self).__init__()

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Phase-specific heads
        self.exploration_head = self._build_head(hidden_dim, action_dim)
        self.navigation_head = self._build_head(hidden_dim, action_dim)  
        self.precision_head = self._build_head(hidden_dim, action_dim)
        
        # Phase classifier
        self.phase_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # 3 phases: exploration, navigation, precision
            nn.Softmax(dim=-1)
        )
    
    def _build_head(self, hidden_dim, action_dim):
        """Builds a single Q-value head."""
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, state):
        """
        Performs the forward pass through the network.

        Args:
            state (torch.Tensor): The input state tensor.

        Returns:
            tuple: A tuple containing:
                - q_values (torch.Tensor): The final, weighted Q-values for each action.
                - phase_probs (torch.Tensor): The probabilities for each of the three phases.
        """
        # Ensure state is a tensor
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        
        # Process input through the shared feature extractor
        features = self.feature_extractor(state)
        
        # Get phase probabilities from the classifier
        phase_probs = self.phase_classifier(features)
        
        # Compute Q-values for each phase-specific head
        exploration_q = self.exploration_head(features)
        navigation_q = self.navigation_head(features)
        precision_q = self.precision_head(features)
        
        # Combine the Q-values from the heads using the phase probabilities as weights
        # The phase_probs need to be reshaped to correctly broadcast over the Q-values
        q_values = (phase_probs[:, 0:1] * exploration_q + 
                    phase_probs[:, 1:2] * navigation_q + 
                    phase_probs[:, 2:3] * precision_q)
        
        return q_values, phase_probs
