from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import wandb
from torch import Tensor

from ..envs.crafting import CraftingEnv
from ..envs.games import SequentialGamesEnv
from ..envs.navigation import NavigationEnv
from ..envs.story import StoryEnv
from ..memory import HierarchicalMemory
from ..models.policy import MemoryAugmentedPolicy
from ..rl.exploration import MemoryGuidedExploration
from ..rl.ppo import PPOTrainer


class ValidationExperiment:
    """Base class for validation experiments."""

    def __init__(
        self,
        env_name: str,
        memory_config: Dict,
        policy_config: Dict,
        training_config: Dict,
        use_wandb: bool = True,
    ):
        """Initialize validation experiment.

        Args:
            env_name: Name of environment to use
            memory_config: Memory configuration
            policy_config: Policy configuration
            training_config: Training configuration
            use_wandb: Whether to use W&B logging
        """
        self.env_name = env_name
        self.memory_config = memory_config
        self.policy_config = policy_config
        self.training_config = training_config
        self.use_wandb = use_wandb

        # Initialize environment
        if env_name == "crafting":
            self.env = CraftingEnv(**training_config.get("env_kwargs", {}))
        elif env_name == "navigation":
            self.env = NavigationEnv(**training_config.get("env_kwargs", {}))
        elif env_name == "games":
            self.env = SequentialGamesEnv(**training_config.get("env_kwargs", {}))
        elif env_name == "story":
            self.env = StoryEnv(**training_config.get("env_kwargs", {}))
        else:
            raise ValueError(f"Unknown environment: {env_name}")

        # Initialize memory
        self.memory = HierarchicalMemory(**memory_config)

        # Initialize policy
        self.policy = MemoryAugmentedPolicy(
            state_dim=self._get_state_dim(),
            action_dim=self._get_action_dim(),
            memory=self.memory,
            **policy_config,
        )

        # Initialize exploration
        self.exploration = MemoryGuidedExploration(
            memory=self.memory,
            state_dim=self._get_state_dim(),
            action_dim=self._get_action_dim(),
            **training_config.get("exploration_kwargs", {}),
        )

        # Initialize trainer
        self.trainer = PPOTrainer(
            policy=self.policy,
            **training_config.get("ppo_kwargs", {}),
        )

        # Initialize metrics
        self.metrics = {
            "episode_rewards": [],
            "episode_lengths": [],
            "success_rate": [],
            "memory_usage": [],
            "compression_ratio": [],
            "exploration_rate": [],
        }

        # Initialize W&B
        if use_wandb:
            wandb.init(
                project="hcmrl",
                config={
                    "env_name": env_name,
                    "memory_config": memory_config,
                    "policy_config": policy_config,
                    "training_config": training_config,
                },
            )

    def _get_state_dim(self) -> int:
        """Get state dimension.

        Returns:
            State dimension
        """
        if isinstance(self.env.observation_space, spaces.Dict):
            return sum(
                np.prod(space.shape)
                for space in self.env.observation_space.values()
            )
        return np.prod(self.env.observation_space.shape)

    def _get_action_dim(self) -> int:
        """Get action dimension.

        Returns:
            Action dimension
        """
        if isinstance(self.env.action_space, spaces.Dict):
            return sum(
                np.prod(space.shape)
                for space in self.env.action_space.values()
            )
        return self.env.action_space.n

    def _process_state(self, state: Dict) -> Tensor:
        """Process state into tensor.

        Args:
            state: Environment state

        Returns:
            Processed state tensor
        """
        if isinstance(state, dict):
            # Flatten dictionary observation
            state_list = []
            for value in state.values():
                if isinstance(value, np.ndarray):
                    state_list.append(value.flatten())
                else:
                    state_list.append(np.array([value]))
            state_array = np.concatenate(state_list)
        else:
            state_array = state.flatten()

        return torch.FloatTensor(state_array).unsqueeze(0)

    def train(
        self,
        num_episodes: int,
        eval_interval: int = 10,
    ) -> Dict[str, List[float]]:
        """Train policy.

        Args:
            num_episodes: Number of episodes to train
            eval_interval: Episodes between evaluations

        Returns:
            Dictionary of training metrics
        """
        for episode in range(num_episodes):
            # Run training episode
            state = self.env.reset()
            state = self._process_state(state)
            done = False
            episode_reward = 0
            episode_length = 0

            while not done:
                # Get action from policy
                action, value, info, lstm_state = self.policy(
                    state,
                    deterministic=False,
                )

                # Apply exploration
                if self.exploration.should_explore(state):
                    action = self.exploration.modify_action_distribution(
                        state,
                        action,
                    ).sample()

                # Take step in environment
                next_state, reward, done, step_info = self.env.step(action)
                next_state = self._process_state(next_state)

                # Store transition
                self.trainer.store_transition(
                    state,
                    action,
                    reward,
                    value,
                    info["log_prob"],
                    done,
                    lstm_state,
                )

                # Update exploration
                self.exploration.update_exploration(
                    state,
                    action,
                    torch.tensor(reward),
                    next_state,
                    torch.tensor(done),
                )

                # Update state
                state = next_state
                episode_reward += reward
                episode_length += 1

            # Update policy
            metrics = self.trainer.update(
                self._process_state(state),
            )

            # Record metrics
            self.metrics["episode_rewards"].append(episode_reward)
            self.metrics["episode_lengths"].append(episode_length)
            self.metrics["success_rate"].append(
                float(step_info.get("success", False))
            )
            self.metrics["memory_usage"].append(
                self.memory.immediate.current_size +
                self.memory.short_term.current_size +
                len(self.memory.long_term.templates)
            )
            self.metrics["compression_ratio"].append(
                self.memory.short_term.compressed_embeddings.shape[1] /
                self.memory.short_term.embeddings.shape[1]
                if self.memory.short_term.compressed
                else 1.0
            )
            self.metrics["exploration_rate"].append(
                self.exploration.epsilon
            )

            # Log to W&B
            if self.use_wandb:
                wandb.log({
                    "episode": episode,
                    "reward": episode_reward,
                    "length": episode_length,
                    "success": step_info.get("success", False),
                    "memory_usage": self.metrics["memory_usage"][-1],
                    "compression_ratio": self.metrics["compression_ratio"][-1],
                    "exploration_rate": self.metrics["exploration_rate"][-1],
                    **metrics,
                })

            # Run evaluation
            if (episode + 1) % eval_interval == 0:
                eval_metrics = self.evaluate(5)
                if self.use_wandb:
                    wandb.log({
                        "episode": episode,
                        "eval_reward": eval_metrics["reward"],
                        "eval_success": eval_metrics["success"],
                    })

        return self.metrics

    def evaluate(self, num_episodes: int = 5) -> Dict[str, float]:
        """Evaluate policy.

        Args:
            num_episodes: Number of episodes to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        rewards = []
        successes = []

        for _ in range(num_episodes):
            state = self.env.reset()
            state = self._process_state(state)
            done = False
            episode_reward = 0

            while not done:
                # Get action from policy
                action, _, _, _ = self.policy(
                    state,
                    deterministic=True,
                )

                # Take step in environment
                next_state, reward, done, info = self.env.step(action)
                next_state = self._process_state(next_state)

                # Update state
                state = next_state
                episode_reward += reward

            rewards.append(episode_reward)
            successes.append(float(info.get("success", False)))

        return {
            "reward": np.mean(rewards),
            "success": np.mean(successes),
        }


def run_ablation_study(
    base_config: Dict,
    ablation_params: Dict[str, List],
    num_seeds: int = 3,
) -> Dict[str, Dict[str, List[float]]]:
    """Run ablation study.

    Args:
        base_config: Base configuration
        ablation_params: Parameters to ablate
        num_seeds: Number of random seeds

    Returns:
        Dictionary of results for each ablation
    """
    results = {}

    for param_name, param_values in ablation_params.items():
        param_results = {
            "rewards": [],
            "success_rates": [],
            "memory_usage": [],
        }

        for value in param_values:
            # Update config with ablation value
            config = base_config.copy()
            if "." in param_name:
                # Handle nested parameters
                keys = param_name.split(".")
                target = config
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
            else:
                config[param_name] = value

            # Run experiments with different seeds
            for seed in range(num_seeds):
                torch.manual_seed(seed)
                np.random.seed(seed)

                experiment = ValidationExperiment(
                    **config,
                    use_wandb=False,
                )
                metrics = experiment.train(
                    num_episodes=config["training_config"]["num_episodes"],
                )

                param_results["rewards"].append(
                    np.mean(metrics["episode_rewards"][-100:])
                )
                param_results["success_rates"].append(
                    np.mean(metrics["success_rate"][-100:])
                )
                param_results["memory_usage"].append(
                    np.mean(metrics["memory_usage"][-100:])
                )

        results[param_name] = param_results

    return results
