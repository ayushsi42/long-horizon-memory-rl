from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical, Normal

from ..memory import HierarchicalMemory


class MemoryAugmentedPolicy(nn.Module):
    """Policy network augmented with hierarchical memory."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        memory: HierarchicalMemory,
        hidden_dim: int = 256,
        memory_dim: int = 128,
        continuous_actions: bool = False,
        action_std: float = 0.5,
    ):
        """Initialize memory-augmented policy network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            memory: Hierarchical memory instance
            hidden_dim: Hidden layer dimension
            memory_dim: Memory embedding dimension
            continuous_actions: Whether action space is continuous
            action_std: Standard deviation for continuous actions
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = memory
        self.continuous_actions = continuous_actions
        self.action_std = action_std

        # Memory query network
        self.query_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, memory_dim),
        )

        # State processing network
        self.state_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Memory processing network
        self.memory_network = nn.Sequential(
            nn.Linear(memory_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # LSTM for temporal integration
        self.lstm = nn.LSTM(
            input_size=hidden_dim * 2,  # State + Memory
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Action head
        if continuous_actions:
            self.action_mean = nn.Linear(hidden_dim, action_dim)
            self.action_std = nn.Parameter(torch.ones(action_dim) * action_std)
        else:
            self.action_head = nn.Linear(hidden_dim, action_dim)

        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize network weights.

        Args:
            module: Neural network module
        """
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param, gain=np.sqrt(2))
                elif "bias" in name:
                    nn.init.zeros_(param)

    def get_memory_query(self, state: Tensor) -> Tensor:
        """Generate memory query from state.

        Args:
            state: Input state tensor

        Returns:
            Memory query tensor
        """
        return self.query_network(state)

    def forward(
        self,
        state: Tensor,
        memory_state: Optional[Tuple[Tensor, Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[Union[Tensor, Tuple[Tensor, Tensor]], Tensor, Dict, Optional[Tuple[Tensor, Tensor]]]:
        """Forward pass through the policy network.

        Args:
            state: Input state tensor
            memory_state: Optional LSTM memory state
            deterministic: Whether to sample actions deterministically

        Returns:
            Tuple of (action, value, extra_info, next_memory_state)
        """
        batch_size = state.shape[0]

        # Generate memory query and retrieve from memory
        memory_query = self.get_memory_query(state)
        memory_output, memory_meta = self.memory.read(memory_query)

        # Process state and memory
        state_features = self.state_network(state)
        memory_features = self.memory_network(memory_output)

        # Combine features
        combined = torch.cat([state_features, memory_features], dim=-1)
        combined = combined.unsqueeze(1)  # Add time dimension for LSTM

        # LSTM integration
        if memory_state is None:
            lstm_out, next_memory_state = self.lstm(combined)
        else:
            lstm_out, next_memory_state = self.lstm(combined, memory_state)

        lstm_out = lstm_out.squeeze(1)  # Remove time dimension

        # Value prediction
        value = self.value_head(lstm_out)

        # Action prediction
        if self.continuous_actions:
            action_mean = self.action_mean(lstm_out)
            action_std = self.action_std.expand_as(action_mean)
            dist = Normal(action_mean, action_std)

            if deterministic:
                action = action_mean
            else:
                action = dist.sample()

            action_log_prob = dist.log_prob(action).sum(-1)
            entropy = dist.entropy().sum(-1)
        else:
            action_logits = self.action_head(lstm_out)
            dist = Categorical(logits=action_logits)

            if deterministic:
                action = torch.argmax(action_logits, dim=-1)
            else:
                action = dist.sample()

            action_log_prob = dist.log_prob(action)
            entropy = dist.entropy()

        # Extra information
        extra_info = {
            "log_prob": action_log_prob,
            "entropy": entropy,
            "memory_meta": memory_meta,
        }

        return action, value, extra_info, next_memory_state

    def evaluate_actions(
        self,
        state: Tensor,
        action: Tensor,
        memory_state: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Dict]:
        """Evaluate actions for PPO updates.

        Args:
            state: Input state tensor
            action: Actions to evaluate
            memory_state: Optional LSTM memory state

        Returns:
            Tuple of (log_prob, entropy, value, extra_info)
        """
        # Generate memory query and retrieve from memory
        memory_query = self.get_memory_query(state)
        memory_output, memory_meta = self.memory.read(memory_query)

        # Process state and memory
        state_features = self.state_network(state)
        memory_features = self.memory_network(memory_output)

        # Combine features
        combined = torch.cat([state_features, memory_features], dim=-1)
        combined = combined.unsqueeze(1)  # Add time dimension for LSTM

        # LSTM integration
        if memory_state is None:
            lstm_out, _ = self.lstm(combined)
        else:
            lstm_out, _ = self.lstm(combined, memory_state)

        lstm_out = lstm_out.squeeze(1)  # Remove time dimension

        # Value prediction
        value = self.value_head(lstm_out)

        # Action evaluation
        if self.continuous_actions:
            action_mean = self.action_mean(lstm_out)
            action_std = self.action_std.expand_as(action_mean)
            dist = Normal(action_mean, action_std)
            log_prob = dist.log_prob(action).sum(-1)
            entropy = dist.entropy().sum(-1)
        else:
            action_logits = self.action_head(lstm_out)
            dist = Categorical(logits=action_logits)
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

        # Extra information
        extra_info = {
            "memory_meta": memory_meta,
        }

        return log_prob, entropy, value, extra_info

    def update_memory(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        next_state: Tensor,
        done: Tensor,
    ) -> None:
        """Update memory with new experience.

        Args:
            state: Current state
            action: Taken action
            reward: Received reward
            next_state: Next state
            done: Done flag
        """
        # Create experience embedding
        if self.continuous_actions:
            action_embedding = action
        else:
            action_embedding = torch.zeros(
                (action.shape[0], self.action_dim),
                device=action.device,
            )
            action_embedding.scatter_(1, action.unsqueeze(-1), 1)

        experience = torch.cat([
            state,
            action_embedding,
            reward.unsqueeze(-1),
            next_state,
            done.unsqueeze(-1),
        ], dim=-1)

        # Store in immediate memory
        self.memory.write(experience, memory_type="immediate")

        # Trigger memory consolidation
        if done.any():
            self.memory.consolidate()
