from setuptools import setup, find_packages

setup(
    name="hcmrl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "gym>=0.21.0",
        "wandb>=0.12.0",
        "pytest>=6.0.0",
        "tqdm>=4.62.0",
        "matplotlib>=3.4.0",
        "pandas>=1.3.0",
    ],
    author="Ayush Singh",
    description="Hierarchical Compressed Memory for Resource-Limited RL",
    python_requires=">=3.8",
)
