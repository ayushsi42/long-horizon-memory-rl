import numpy as np
import pytest
import torch

from hcmrl.memory import HierarchicalMemory
from hcmrl.memory.base import CompressibleMemory, MemoryBase
from hcmrl.memory.immediate import ImmediateMemory
from hcmrl.memory.long_term import LongTermMemory
from hcmrl.memory.short_term import ShortTermMemory


def test_memory_base_abstract():
    """Test that MemoryBase is abstract."""
    with pytest.raises(TypeError):
        MemoryBase(32, 128)


def test_immediate_memory_initialization(device):
    """Test immediate memory initialization."""
    memory = ImmediateMemory(32, 128).to(device)
    assert memory.capacity == 32
    assert memory.embedding_dim == 128
    assert memory.current_size == 0
    assert memory.embeddings.shape == (32, 128)


def test_immediate_memory_write(device):
    """Test immediate memory write operation."""
    memory = ImmediateMemory(32, 128).to(device)
    embeddings = torch.randn(5, 128, device=device)
    metadata = {"test": "data"}

    memory.write(embeddings, metadata)
    assert memory.current_size == 5
    assert torch.allclose(memory.embeddings[:5], embeddings)
    assert memory.metadata_list[0] == metadata


def test_immediate_memory_read(device):
    """Test immediate memory read operation."""
    memory = ImmediateMemory(32, 128).to(device)
    embeddings = torch.randn(5, 128, device=device)
    memory.write(embeddings)

    query = torch.randn(1, 128, device=device)
    retrieved, metadata = memory.read(query)

    assert retrieved.shape == (1, 128)
    assert metadata is not None
    assert "attention_weights" in metadata


def test_short_term_memory_compression(device):
    """Test short-term memory compression."""
    memory = ShortTermMemory(16, 128, 64).to(device)
    embeddings = torch.randn(10, 128, device=device)
    memory.write(embeddings)

    assert not memory.compressed
    memory.compress()
    assert memory.compressed
    assert memory.compressed_embeddings[:10].shape == (10, 64)


def test_short_term_memory_decompression(device):
    """Test short-term memory decompression."""
    memory = ShortTermMemory(16, 128, 64).to(device)
    embeddings = torch.randn(10, 128, device=device)
    memory.write(embeddings)
    memory.compress()

    memory.decompress()
    assert not memory.compressed
    assert torch.allclose(memory.embeddings[:10], embeddings, rtol=1e-1, atol=1e-1)


def test_long_term_memory_templates(device):
    """Test long-term memory template management."""
    memory = LongTermMemory(8, 128).to(device)
    embeddings = torch.randn(5, 128, device=device)
    metadata = {"success": torch.ones(5, device=device)}

    memory.write(embeddings, metadata)
    assert len(memory.templates) > 0
    assert memory.templates[0].embedding.shape == (128,)


def test_long_term_memory_consolidation(device):
    """Test long-term memory consolidation."""
    memory = LongTermMemory(8, 128).to(device)
    embeddings = torch.randn(10, 128, device=device)
    metadata = {"success": torch.ones(10, device=device)}

    memory.write(embeddings, metadata)
    initial_templates = len(memory.templates)
    memory.consolidate()
    assert len(memory.templates) <= initial_templates


def test_hierarchical_memory_integration(memory):
    """Test hierarchical memory integration."""
    # Write to immediate memory
    embeddings = torch.randn(5, 128, device=memory.device)
    memory.write(embeddings, memory_type="immediate")
    assert memory.immediate.current_size == 5

    # Write to short-term memory
    memory.write(embeddings, memory_type="short_term")
    assert memory.short_term.current_size == 5

    # Write to long-term memory
    metadata = {"success": torch.ones(5, device=memory.device)}
    memory.write(embeddings, memory_type="long_term", metadata=metadata)
    assert len(memory.long_term.templates) > 0


def test_hierarchical_memory_read(memory):
    """Test hierarchical memory read operation."""
    # Write to all memory levels
    embeddings = torch.randn(5, 128, device=memory.device)
    memory.write(embeddings, memory_type="immediate")
    memory.write(embeddings, memory_type="short_term")
    memory.write(
        embeddings,
        memory_type="long_term",
        metadata={"success": torch.ones(5, device=memory.device)},
    )

    # Read with query
    query = torch.randn(1, 128, device=memory.device)
    retrieved, metadata = memory.read(query)

    assert retrieved.shape == (1, 128)
    assert metadata is not None
    assert "immediate" in metadata
    assert "short_term" in metadata
    assert "long_term" in metadata


def test_hierarchical_memory_consolidation(memory):
    """Test hierarchical memory consolidation."""
    # Fill immediate memory
    embeddings = torch.randn(
        memory.immediate.capacity,
        memory.embedding_dim,
        device=memory.device,
    )
    memory.write(embeddings, memory_type="immediate")

    # Trigger consolidation
    memory.consolidate()

    # Check that data was transferred
    assert memory.immediate.current_size < memory.immediate.capacity
    assert memory.short_term.current_size > 0


def test_memory_device_transfer(memory_config):
    """Test memory device transfer."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Create memory on CPU
    memory = HierarchicalMemory(**memory_config)
    assert memory.device == torch.device("cpu")

    # Move to GPU
    memory = memory.cuda()
    assert memory.device == torch.device("cuda")
    assert memory.immediate.device == torch.device("cuda")
    assert memory.short_term.device == torch.device("cuda")
    assert memory.long_term.device == torch.device("cuda")

    # Move back to CPU
    memory = memory.cpu()
    assert memory.device == torch.device("cpu")
    assert memory.immediate.device == torch.device("cpu")
    assert memory.short_term.device == torch.device("cpu")
    assert memory.long_term.device == torch.device("cpu")
