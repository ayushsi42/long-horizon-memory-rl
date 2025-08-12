import numpy as np
import pytest
import torch

from hcmrl.validation.analysis import MemoryAnalyzer
from hcmrl.validation.experiments import ValidationExperiment
from hcmrl.validation.metrics import ResourceMetrics, ValidationMetrics


def test_validation_experiment_initialization(
    memory_config,
    policy_config,
    training_config,
):
    """Test validation experiment initialization."""
    experiment = ValidationExperiment(
        env_name="crafting",
        memory_config=memory_config,
        policy_config=policy_config,
        training_config=training_config,
        use_wandb=False,
    )

    assert experiment.env_name == "crafting"
    assert experiment.memory_config == memory_config
    assert experiment.policy_config == policy_config
    assert experiment.training_config == training_config
    assert not experiment.use_wandb

    assert hasattr(experiment, "env")
    assert hasattr(experiment, "memory")
    assert hasattr(experiment, "policy")
    assert hasattr(experiment, "exploration")
    assert hasattr(experiment, "trainer")


@pytest.mark.parametrize("env_name", [
    "crafting",
    "navigation",
    "games",
    "story",
])
def test_validation_experiment_environments(
    env_name,
    memory_config,
    policy_config,
    training_config,
):
    """Test validation experiment with different environments."""
    experiment = ValidationExperiment(
        env_name=env_name,
        memory_config=memory_config,
        policy_config=policy_config,
        training_config=training_config,
        use_wandb=False,
    )

    # Test single training episode
    metrics = experiment.train(num_episodes=1)
    assert "episode_rewards" in metrics
    assert "episode_lengths" in metrics
    assert "success_rate" in metrics
    assert "memory_usage" in metrics
    assert "compression_ratio" in metrics
    assert "exploration_rate" in metrics


def test_validation_experiment_evaluation(
    memory_config,
    policy_config,
    training_config,
):
    """Test validation experiment evaluation."""
    experiment = ValidationExperiment(
        env_name="crafting",
        memory_config=memory_config,
        policy_config=policy_config,
        training_config=training_config,
        use_wandb=False,
    )

    # Test evaluation
    metrics = experiment.evaluate(num_episodes=2)
    assert "reward" in metrics
    assert "success" in metrics


def test_validation_metrics(memory):
    """Test validation metrics computation."""
    metrics = ValidationMetrics(memory)

    # Test memory efficiency metrics
    efficiency = metrics.memory_efficiency()
    assert "immediate_usage" in efficiency
    assert "short_term_usage" in efficiency
    assert "long_term_usage" in efficiency
    assert "short_term_compression" in efficiency
    assert "immediate_coherence" in efficiency
    assert "short_term_coherence" in efficiency
    assert "long_term_coherence" in efficiency

    # Test retrieval quality metrics
    batch_size = 4
    queries = torch.randn(
        batch_size,
        memory.embedding_dim,
        device=memory.device,
    )
    expected = torch.randn(
        batch_size,
        memory.embedding_dim,
        device=memory.device,
    )
    quality = metrics.retrieval_quality(queries, expected)
    assert "immediate_diversity" in quality
    assert "short_term_diversity" in quality
    assert "long_term_diversity" in quality

    # Test temporal analysis
    episode_data = [
        {
            "state": torch.randn(memory.embedding_dim, device=memory.device),
            "compressed": False,
            "consolidated": False,
            "retrieval_age": 0,
        }
        for _ in range(5)
    ]
    temporal = metrics.temporal_analysis(episode_data)
    assert "immediate_usage" in temporal
    assert "short_term_usage" in temporal
    assert "long_term_usage" in temporal
    assert "compression_frequency" in temporal
    assert "consolidation_frequency" in temporal
    assert "retrieval_staleness" in temporal


def test_resource_metrics(memory):
    """Test resource metrics computation."""
    # Test memory footprint calculation
    footprint = ResourceMetrics.calculate_memory_footprint(memory)
    assert "immediate_memory" in footprint
    assert "short_term_memory" in footprint
    assert "long_term_memory" in footprint
    assert "total_memory" in footprint

    # Test computation time calculation
    episode_data = [
        {
            "policy_time": 0.1,
            "memory_time": 0.2,
            "exploration_time": 0.1,
            "total_time": 0.4,
        }
        for _ in range(5)
    ]
    timing = ResourceMetrics.calculate_computation_time(episode_data)
    assert "policy_time" in timing
    assert "memory_time" in timing
    assert "exploration_time" in timing
    assert "total_time" in timing
    assert "policy_ratio" in timing
    assert "memory_ratio" in timing
    assert "exploration_ratio" in timing


def test_memory_analyzer(memory, tmp_path):
    """Test memory analyzer visualization."""
    analyzer = MemoryAnalyzer(memory)

    # Write some data to memory
    batch_size = 4
    embeddings = torch.randn(
        batch_size,
        memory.embedding_dim,
        device=memory.device,
    )
    memory.write(embeddings, memory_type="immediate")
    memory.write(embeddings, memory_type="short_term")
    memory.write(
        embeddings,
        memory_type="long_term",
        metadata={"success": torch.ones(batch_size, device=memory.device)},
    )

    # Test memory structure visualization
    save_path = tmp_path / "memory_structure.png"
    analyzer.visualize_memory_structure(str(save_path))
    assert save_path.exists()

    # Test retrieval pattern analysis
    episode_data = [
        {
            "memory_retrieval": torch.randn(
                memory.embedding_dim,
                device=memory.device,
            ),
            "attention_weights": np.random.random(memory.immediate.capacity),
            "timestamp": i,
        }
        for i in range(5)
    ]
    save_path = tmp_path / "retrieval_patterns.png"
    results = analyzer.analyze_retrieval_patterns(episode_data, str(save_path))
    assert save_path.exists()
    assert "retrievals" in results
    assert "attention_weights" in results
    assert "temporal_stats" in results

    # Test compression efficiency analysis
    save_path = tmp_path / "compression_efficiency.png"
    metrics = analyzer.analyze_compression_efficiency(str(save_path))
    assert save_path.exists()
    if memory.short_term.compressed:
        assert "compression_ratio" in metrics
        assert "reconstruction_error" in metrics
