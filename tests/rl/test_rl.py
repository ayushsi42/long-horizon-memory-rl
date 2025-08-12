import numpy as np
import pytest
import torch
from torch.distributions import Categorical, Normal

from hcmrl.models.policy import MemoryAugmentedPolicy
from hcmrl.rl.exploration import MemoryGuidedExploration
from hcmrl.rl.ppo import PPOMemory, PPOTrainer


def test_policy_initialization(policy):
    """Test policy network initialization."""
    assert isinstance(policy, MemoryAugmentedPolicy)
    assert hasattr(policy, "state_network")
    assert hasattr(policy, "memory_network")
    assert hasattr(policy, "lstm")


def test_policy_forward_discrete(policy):
    """Test policy forward pass with discrete actions."""
    batch_size = 4
    state = torch.randn(batch_size, policy.state_dim, device=policy.device)

    # Test forward pass
    action, value, info, lstm_state = policy(state)

    assert action.shape == (batch_size,)
    assert value.shape == (batch_size, 1)
    assert "log_prob" in info
    assert "entropy" in info
    assert isinstance(lstm_state, tuple)
    assert len(lstm_state) == 2


def test_policy_forward_continuous(memory, device):
    """Test policy forward pass with continuous actions."""
    policy = MemoryAugmentedPolicy(
        state_dim=128,
        action_dim=3,
        memory=memory,
        continuous_actions=True,
        action_std=0.5,
    ).to(device)

    batch_size = 4
    state = torch.randn(batch_size, policy.state_dim, device=device)

    # Test forward pass
    action, value, info, lstm_state = policy(state)

    assert action.shape == (batch_size, 3)
    assert value.shape == (batch_size, 1)
    assert "log_prob" in info
    assert "entropy" in info
    assert isinstance(lstm_state, tuple)
    assert len(lstm_state) == 2


def test_policy_evaluation(policy):
    """Test policy evaluation."""
    batch_size = 4
    state = torch.randn(batch_size, policy.state_dim, device=policy.device)
    action = torch.randint(0, policy.action_dim, (batch_size,), device=policy.device)

    # Test action evaluation
    log_prob, entropy, value, info = policy.evaluate_actions(state, action)

    assert log_prob.shape == (batch_size,)
    assert entropy.shape == (batch_size,)
    assert value.shape == (batch_size, 1)
    assert "memory_meta" in info


def test_policy_memory_update(policy):
    """Test policy memory update."""
    batch_size = 4
    state = torch.randn(batch_size, policy.state_dim, device=policy.device)
    action = torch.randint(0, policy.action_dim, (batch_size,), device=policy.device)
    reward = torch.randn(batch_size, device=policy.device)
    next_state = torch.randn(batch_size, policy.state_dim, device=policy.device)
    done = torch.zeros(batch_size, device=policy.device)

    # Test memory update
    policy.update_memory(state, action, reward, next_state, done)
    assert policy.memory.immediate.current_size > 0


def test_exploration_initialization(exploration):
    """Test exploration initialization."""
    assert isinstance(exploration, MemoryGuidedExploration)
    assert hasattr(exploration, "novelty_network")
    assert hasattr(exploration, "confidence_network")


def test_exploration_novelty(exploration):
    """Test novelty computation."""
    batch_size = 4
    state = torch.randn(
        batch_size,
        exploration.memory.embedding_dim,
        device=exploration.device,
    )

    # Test novelty computation
    novelty = exploration.compute_novelty(state)
    assert novelty.shape == (batch_size, 1)
    assert torch.all(novelty >= 0) and torch.all(novelty <= 1)


def test_exploration_confidence(exploration):
    """Test confidence computation."""
    batch_size = 4
    state = torch.randn(
        batch_size,
        exploration.memory.embedding_dim,
        device=exploration.device,
    )
    memory_output = torch.randn(
        batch_size,
        exploration.memory.embedding_dim,
        device=exploration.device,
    )

    # Test confidence computation
    confidence = exploration.compute_confidence(state, memory_output)
    assert confidence.shape == (batch_size, 1)
    assert torch.all(confidence >= 0) and torch.all(confidence <= 1)


def test_exploration_update(exploration):
    """Test exploration update."""
    batch_size = 4
    state = torch.randn(
        batch_size,
        exploration.memory.embedding_dim,
        device=exploration.device,
    )
    action = torch.randint(0, 10, (batch_size,), device=exploration.device)
    reward = torch.randn(batch_size, device=exploration.device)
    next_state = torch.randn(
        batch_size,
        exploration.memory.embedding_dim,
        device=exploration.device,
    )
    done = torch.zeros(batch_size, device=exploration.device)

    # Test exploration update
    metrics = exploration.update_exploration(state, action, reward, next_state, done)
    assert "novelty_loss" in metrics
    assert "confidence_loss" in metrics
    assert "epsilon" in metrics


def test_exploration_action_modification(exploration):
    """Test action distribution modification."""
    batch_size = 4
    state = torch.randn(
        batch_size,
        exploration.memory.embedding_dim,
        device=exploration.device,
    )

    # Test discrete action modification
    logits = torch.randn(batch_size, 10, device=exploration.device)
    dist = Categorical(logits=logits)
    modified_dist = exploration.modify_action_distribution(state, dist)
    assert isinstance(modified_dist, Categorical)

    # Test continuous action modification
    mean = torch.randn(batch_size, 3, device=exploration.device)
    std = torch.ones(batch_size, 3, device=exploration.device)
    dist = Normal(mean, std)
    modified_dist = exploration.modify_action_distribution(state, dist)
    assert isinstance(modified_dist, Normal)


def test_ppo_memory():
    """Test PPO memory buffer."""
    memory = PPOMemory()
    batch_size = 4

    # Test transition storage
    state = torch.randn(batch_size, 128)
    action = torch.randint(0, 10, (batch_size,))
    reward = torch.randn(batch_size)
    value = torch.randn(batch_size)
    log_prob = torch.randn(batch_size)
    done = torch.zeros(batch_size)
    lstm_state = (
        torch.randn(1, batch_size, 256),
        torch.randn(1, batch_size, 256),
    )

    memory.push(state, action, reward, value, log_prob, done, lstm_state)
    assert len(memory.states) == 1
    assert len(memory.actions) == 1
    assert len(memory.rewards) == 1
    assert len(memory.values) == 1
    assert len(memory.log_probs) == 1
    assert len(memory.dones) == 1
    assert len(memory.lstm_states) == 1

    # Test memory retrieval
    states, actions, rewards, values, log_probs, dones = memory.get()
    assert states.shape[0] == 1
    assert actions.shape[0] == 1
    assert rewards.shape[0] == 1
    assert values.shape[0] == 1
    assert log_probs.shape[0] == 1
    assert dones.shape[0] == 1

    # Test memory clearing
    memory.clear()
    assert len(memory.states) == 0
    assert len(memory.actions) == 0
    assert len(memory.rewards) == 0
    assert len(memory.values) == 0
    assert len(memory.log_probs) == 0
    assert len(memory.dones) == 0
    assert len(memory.lstm_states) == 0


def test_ppo_trainer(policy):
    """Test PPO trainer."""
    trainer = PPOTrainer(
        policy=policy,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        value_clip_range=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        num_epochs=4,
        batch_size=64,
    )

    # Test GAE computation
    batch_size = 4
    rewards = torch.randn(batch_size, device=policy.device)
    values = torch.randn(batch_size, device=policy.device)
    dones = torch.zeros(batch_size, device=policy.device)
    next_value = torch.randn(1, device=policy.device)

    returns, advantages = trainer.compute_gae(rewards, values, dones, next_value)
    assert returns.shape == (batch_size,)
    assert advantages.shape == (batch_size,)

    # Test policy update
    state = torch.randn(batch_size, policy.state_dim, device=policy.device)
    action = torch.randint(0, policy.action_dim, (batch_size,), device=policy.device)
    reward = torch.randn(batch_size, device=policy.device)
    value = torch.randn(batch_size, device=policy.device)
    log_prob = torch.randn(batch_size, device=policy.device)
    done = torch.zeros(batch_size, device=policy.device)

    trainer.store_transition(state, action, reward.item(), value.item(), log_prob, done.item())
    metrics = trainer.update(next_value)

    assert "loss" in metrics
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy_loss" in metrics
    assert "approx_kl" in metrics
    assert "clip_fraction" in metrics
