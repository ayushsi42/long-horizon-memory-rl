# HCMRL: Hierarchical Compressed Memory for Resource-Limited RL

This repository implements HCMRL (Hierarchical Compressed Memory for Resource-Limited RL), a novel approach to memory-augmented reinforcement learning that is specifically designed for resource-constrained environments.

## Features

- Multi-scale memory architecture with three levels:
  - Immediate Memory (32-slot attention-based buffer)
  - Short-Term Memory (16 compressed episode summaries)
  - Long-Term Memory (8 abstract policy templates)
- Efficient compression mechanisms using variational autoencoders
- Hierarchical attention for memory retrieval
- Memory consolidation system with adaptive compression
- Resource-efficient implementation (<12GB during training, <2GB during inference)

## Installation

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Project Structure

```
hcmrl/
├── memory/              # Memory components
│   ├── immediate.py     # Immediate memory buffer
│   ├── short_term.py    # Short-term episodic memory
│   ├── long_term.py     # Long-term strategic memory
│   └── compression.py   # Memory compression mechanisms
├── models/              # Neural network models
│   ├── policy.py        # Policy network
│   ├── value.py         # Value network
│   └── attention.py     # Attention mechanisms
├── rl/                  # RL components
│   ├── ppo.py          # PPO implementation
│   ├── exploration.py   # Exploration strategies
│   └── credit.py       # Credit assignment
├── envs/               # Environment implementations
│   ├── crafting.py     # Minecraft-inspired crafting
│   ├── navigation.py   # Multi-room navigation
│   ├── games.py        # Sequential decision games
│   └── story.py        # Procedural story generation
└── utils/              # Utility functions
    ├── logging.py      # Logging utilities
    ├── metrics.py      # Performance metrics
    └── visualization.py # Visualization tools
```

## Usage

```python
from hcmrl.memory import HierarchicalMemory
from hcmrl.models import MemoryAugmentedPolicy
from hcmrl.rl import PPOTrainer

# Initialize components
memory = HierarchicalMemory(
    immediate_size=32,
    short_term_size=16,
    long_term_size=8
)

policy = MemoryAugmentedPolicy(
    state_dim=64,
    action_dim=10,
    memory=memory
)

# Train the agent
trainer = PPOTrainer(policy, memory)
trainer.train(num_steps=1_000_000)
```

## Benchmarks

The implementation includes four main benchmark environments:
1. Minecraft-inspired Crafting (10,000+ step episodes)
2. Multi-Room Navigation with Keys (5,000+ step episodes)
3. Sequential Decision Games (2,000+ step episodes)
4. Procedural Story Generation (3,000+ step episodes)

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{hcmrl2024,
  title={Hierarchical Compressed Memory for Long-Horizon Reinforcement Learning in Resource-Constrained Environments},
  author={Singh, Ayush},
  year={2024}
}
```
