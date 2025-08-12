from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from ..models.policy import MemoryAugmentedPolicy


class PPOMemory:
    """Memory buffer for PPO."""

    def __init__(self):
        self.states: List[Tensor] = []
        self.actions: List[Tensor] = []
        self.rewards: List[Tensor] = []
        self.values: List[Tensor] = []
        self.log_probs: List[Tensor] = []
        self.dones: List[Tensor] = []
        self.lstm_states: List[Tuple[Tensor, Tensor]] = []

    def push(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        value: Tensor,
        log_prob: Tensor,
        done: Tensor,
        lstm_state: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> None:
        """Add transition to memory.

        Args:
            state: State tensor
            action: Action tensor
            reward: Reward tensor
            value: Value prediction
            log_prob: Action log probability
            done: Done flag
            lstm_state: LSTM hidden state
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        if lstm_state is not None:
            self.lstm_states.append(lstm_state)

    def get(self) -> Tuple[Tensor, ...]:
        """Get all stored transitions.

        Returns:
            Tuple of (states, actions, rewards, values, log_probs, dones)
        """
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        rewards = torch.stack(self.rewards)
        values = torch.stack(self.values)
        log_probs = torch.stack(self.log_probs)
        dones = torch.stack(self.dones)
        
        return states, actions, rewards, values, log_probs, dones

    def clear(self) -> None:
        """Clear memory buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.lstm_states.clear()


class PPOTrainer:
    """PPO trainer with memory integration."""

    def __init__(
        self,
        policy: MemoryAugmentedPolicy,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        value_clip_range: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        num_epochs: int = 4,
        batch_size: int = 64,
    ):
        """Initialize PPO trainer.

        Args:
            policy: Memory-augmented policy network
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_range: PPO clip range
            value_clip_range: Value function clip range
            entropy_coef: Entropy coefficient
            value_coef: Value loss coefficient
            max_grad_norm: Maximum gradient norm
            num_epochs: Number of PPO epochs
            batch_size: Batch size for updates
        """
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.value_clip_range = value_clip_range
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        self.memory = PPOMemory()
        self.device = next(policy.parameters()).device

    def compute_gae(
        self,
        rewards: Tensor,
        values: Tensor,
        dones: Tensor,
        next_value: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Compute Generalized Advantage Estimation.

        Args:
            rewards: Reward tensor
            values: Value predictions
            dones: Done flags
            next_value: Next state value prediction

        Returns:
            Tuple of (returns, advantages)
        """
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        last_gae = 0
        last_return = next_value

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = next_value
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_value = values[t + 1]

            delta = (
                rewards[t]
                + self.gamma * next_value * next_non_terminal
                - values[t]
            )
            last_gae = (
                delta
                + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            )
            advantages[t] = last_gae
            last_return = (
                rewards[t]
                + self.gamma * next_non_terminal * last_return
            )
            returns[t] = last_return

        return returns, advantages

    def update(self, next_value: Tensor) -> Dict[str, float]:
        """Update policy using PPO.

        Args:
            next_value: Next state value for GAE computation

        Returns:
            Dictionary of training metrics
        """
        # Get data from memory
        states, actions, rewards, values, old_log_probs, dones = self.memory.get()

        # Compute returns and advantages
        returns, advantages = self.compute_gae(rewards, values, dones, next_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update epochs
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_approx_kl = 0
        total_clip_fraction = 0
        total_batches = 0

        for _ in range(self.num_epochs):
            # Generate random permutation for batching
            indices = torch.randperm(len(states))

            for start_idx in range(0, len(states), self.batch_size):
                total_batches += 1
                batch_indices = indices[start_idx:start_idx + self.batch_size]

                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_values = values[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]

                # Get current policy outputs
                log_probs, entropy, values, _ = self.policy.evaluate_actions(
                    batch_states,
                    batch_actions,
                )

                # Policy loss
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_range,
                    1.0 + self.clip_range,
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                values_pred = values.squeeze(-1)
                if self.value_clip_range is not None:
                    values_clipped = batch_old_values + torch.clamp(
                        values_pred - batch_old_values,
                        -self.value_clip_range,
                        self.value_clip_range,
                    )
                    value_loss = torch.max(
                        (values_pred - batch_returns) ** 2,
                        (values_clipped - batch_returns) ** 2,
                    ).mean()
                else:
                    value_loss = ((values_pred - batch_returns) ** 2).mean()

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # Update policy
                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(
                        self.policy.parameters(),
                        self.max_grad_norm,
                    )
                self.optimizer.step()

                # Record metrics
                with torch.no_grad():
                    total_loss += loss.item()
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy_loss += entropy_loss.item()
                    total_approx_kl += (
                        (batch_old_log_probs - log_probs).mean().item()
                    )
                    clip_fraction = (
                        (ratio - 1.0).abs() > self.clip_range
                    ).float().mean().item()
                    total_clip_fraction += clip_fraction

        # Clear memory
        self.memory.clear()

        # Return average metrics
        return {
            "loss": total_loss / total_batches,
            "policy_loss": total_policy_loss / total_batches,
            "value_loss": total_value_loss / total_batches,
            "entropy_loss": total_entropy_loss / total_batches,
            "approx_kl": total_approx_kl / total_batches,
            "clip_fraction": total_clip_fraction / total_batches,
        }

    def train_step(
        self,
        state: Tensor,
        lstm_state: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, float, bool, Dict, Optional[Tuple[Tensor, Tensor]]]:
        """Execute single training step.

        Args:
            state: Current state
            lstm_state: LSTM hidden state

        Returns:
            Tuple of (action, value, done, info, next_lstm_state)
        """
        with torch.no_grad():
            action, value, info, next_lstm_state = self.policy(
                state,
                lstm_state,
            )

        return action, value.item(), False, info, next_lstm_state

    def store_transition(
        self,
        state: Tensor,
        action: Tensor,
        reward: float,
        value: float,
        log_prob: Tensor,
        done: bool,
        lstm_state: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> None:
        """Store transition in memory buffer.

        Args:
            state: State tensor
            action: Action tensor
            reward: Reward value
            value: Value prediction
            log_prob: Action log probability
            done: Done flag
            lstm_state: LSTM hidden state
        """
        self.memory.push(
            state,
            action,
            torch.tensor(reward, device=self.device),
            torch.tensor(value, device=self.device),
            log_prob,
            torch.tensor(done, device=self.device),
            lstm_state,
        )
