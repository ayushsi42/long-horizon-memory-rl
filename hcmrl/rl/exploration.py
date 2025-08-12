from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical, Normal

from ..memory import HierarchicalMemory


class MemoryGuidedExploration:
    """Exploration strategy guided by hierarchical memory."""

    def __init__(
        self,
        memory: HierarchicalMemory,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        continuous_actions: bool = False,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        temperature: float = 1.0,
        novelty_weight: float = 0.1,
    ):
        """Initialize memory-guided exploration.

        Args:
            memory: Hierarchical memory instance
            state_dim: State space dimension
            action_dim: Action space dimension
            hidden_dim: Hidden layer dimension
            continuous_actions: Whether action space is continuous
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exploration decay rate
            temperature: Temperature for softmax
            novelty_weight: Weight for novelty bonus
        """
        self.memory = memory
        self.continuous_actions = continuous_actions
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.temperature = temperature
        self.novelty_weight = novelty_weight

        # Networks
        self.device = next(memory.immediate.parameters()).device
        
        # Novelty estimation network
        self.novelty_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(self.device)

        # Memory confidence network
        self.confidence_network = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.novelty_network.parameters()) +
            list(self.confidence_network.parameters()),
            lr=3e-4,
        )

    def compute_novelty(self, state: Tensor) -> Tensor:
        """Compute novelty score for state.

        Args:
            state: Input state tensor

        Returns:
            Novelty score tensor
        """
        # Get memory retrievals
        memory_query = self.memory.immediate.query_network(state)
        memory_output, memory_meta = self.memory.read(memory_query)

        # Compute memory-based novelty
        if memory_meta is not None and "attention_weights" in memory_meta:
            attention_weights = torch.tensor(
                memory_meta["attention_weights"],
                device=self.device,
            )
            max_similarity = attention_weights.max(dim=-1)[0]
            memory_novelty = 1.0 - max_similarity
        else:
            memory_novelty = torch.ones(
                state.shape[0],
                device=self.device,
            )

        # Compute learned novelty
        learned_novelty = self.novelty_network(state)

        # Combine novelties
        novelty = (memory_novelty.unsqueeze(-1) + learned_novelty) / 2

        return novelty

    def compute_confidence(
        self,
        state: Tensor,
        memory_output: Tensor,
    ) -> Tensor:
        """Compute confidence in memory retrieval.

        Args:
            state: Input state tensor
            memory_output: Retrieved memory tensor

        Returns:
            Confidence score tensor
        """
        combined = torch.cat([state, memory_output], dim=-1)
        return self.confidence_network(combined)

    def update_exploration(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        next_state: Tensor,
        done: Tensor,
    ) -> Dict[str, float]:
        """Update exploration networks.

        Args:
            state: Current state
            action: Taken action
            reward: Received reward
            next_state: Next state
            done: Done flag

        Returns:
            Dictionary of training metrics
        """
        # Get memory retrievals
        memory_query = self.memory.immediate.query_network(state)
        memory_output, _ = self.memory.read(memory_query)

        # Compute current novelty and confidence
        novelty = self.compute_novelty(state)
        confidence = self.compute_confidence(state, memory_output)

        # Compute next state novelty
        next_novelty = self.compute_novelty(next_state)

        # Novelty loss: predict decrease in novelty
        novelty_loss = nn.MSELoss()(
            novelty,
            next_novelty * (1 - done.unsqueeze(-1)) + done.unsqueeze(-1),
        )

        # Confidence loss: predict reward
        confidence_loss = nn.MSELoss()(
            confidence,
            torch.sigmoid(reward.unsqueeze(-1)),
        )

        # Total loss
        loss = novelty_loss + confidence_loss

        # Update networks
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay,
        )

        return {
            "novelty_loss": novelty_loss.item(),
            "confidence_loss": confidence_loss.item(),
            "epsilon": self.epsilon,
        }

    def get_exploration_bonus(
        self,
        state: Tensor,
        base_value: Tensor,
    ) -> Tensor:
        """Compute exploration bonus for value function.

        Args:
            state: Input state tensor
            base_value: Base value prediction

        Returns:
            Exploration bonus tensor
        """
        with torch.no_grad():
            novelty = self.compute_novelty(state)
            memory_query = self.memory.immediate.query_network(state)
            memory_output, _ = self.memory.read(memory_query)
            confidence = self.compute_confidence(state, memory_output)

            # Compute bonus based on novelty and confidence
            exploration_bonus = (
                self.novelty_weight * novelty * (1 - confidence)
            )

        return base_value + exploration_bonus

    def modify_action_distribution(
        self,
        state: Tensor,
        action_dist: Union[Categorical, Normal],
    ) -> Union[Categorical, Normal]:
        """Modify action distribution for exploration.

        Args:
            state: Input state tensor
            action_dist: Base action distribution

        Returns:
            Modified action distribution
        """
        with torch.no_grad():
            # Get memory-based information
            memory_query = self.memory.immediate.query_network(state)
            memory_output, memory_meta = self.memory.read(memory_query)
            confidence = self.compute_confidence(state, memory_output)

            if self.continuous_actions:
                # For continuous actions, adjust distribution variance
                if isinstance(action_dist, Normal):
                    exploration_std = (
                        action_dist.scale * (1 + self.epsilon * (1 - confidence))
                    )
                    return Normal(action_dist.loc, exploration_std)
            else:
                # For discrete actions, adjust logits
                if isinstance(action_dist, Categorical):
                    logits = action_dist.logits
                    temperature = self.temperature * (1 + self.epsilon * (1 - confidence))
                    modified_logits = logits / temperature
                    return Categorical(logits=modified_logits)

        return action_dist

    def should_explore(self, state: Tensor) -> bool:
        """Determine if exploration should be used.

        Args:
            state: Input state tensor

        Returns:
            Boolean indicating whether to explore
        """
        with torch.no_grad():
            memory_query = self.memory.immediate.query_network(state)
            memory_output, _ = self.memory.read(memory_query)
            confidence = self.compute_confidence(state, memory_output)

            # Explore with probability epsilon * (1 - confidence)
            explore_prob = self.epsilon * (1 - confidence.item())
            return np.random.random() < explore_prob
