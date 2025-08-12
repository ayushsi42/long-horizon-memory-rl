import pytest
import torch

from hcmrl.memory import HierarchicalMemory
from hcmrl.models.policy import MemoryAugmentedPolicy
from hcmrl.rl.exploration import MemoryGuidedExploration


@pytest.fixture
def device():
    """Get PyTorch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def memory_config():
    """Get memory configuration."""
    return {
        "immediate_size": 32,
        "short_term_size": 16,
        "long_term_size": 8,
        "embedding_dim": 128,
    }


@pytest.fixture
def policy_config():
    """Get policy configuration."""
    return {
        "hidden_dim": 256,
        "memory_dim": 128,
        "continuous_actions": False,
    }


@pytest.fixture
def training_config():
    """Get training configuration."""
    return {
        "env_kwargs": {
            "max_steps": 1000,
        },
        "exploration_kwargs": {
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.995,
        },
        "ppo_kwargs": {
            "lr": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "value_clip_range": 0.2,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "max_grad_norm": 0.5,
            "num_epochs": 4,
            "batch_size": 64,
        },
    }


@pytest.fixture
def memory(memory_config, device):
    """Get hierarchical memory instance."""
    return HierarchicalMemory(**memory_config).to(device)


@pytest.fixture
def policy(memory_config, policy_config, memory, device):
    """Get policy instance."""
    return MemoryAugmentedPolicy(
        state_dim=memory_config["embedding_dim"],
        action_dim=10,
        memory=memory,
        **policy_config,
    ).to(device)


@pytest.fixture
def exploration(memory_config, training_config, memory, device):
    """Get exploration instance."""
    return MemoryGuidedExploration(
        memory=memory,
        state_dim=memory_config["embedding_dim"],
        action_dim=10,
        **training_config["exploration_kwargs"],
    )
