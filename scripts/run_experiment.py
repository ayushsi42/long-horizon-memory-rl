#!/usr/bin/env python3
"""Research-grade runner for HCMRL experiments."""
from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from hcmrl.validation.experiments import ValidationExperiment

try:  # Allow running as module or script
    from .research_utils import (
        apply_overrides,
        configure_logging,
        dump_json,
        ensure_directory,
        load_config_file,
        set_global_seed,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from research_utils import (
        apply_overrides,
        configure_logging,
        dump_json,
        ensure_directory,
        load_config_file,
        set_global_seed,
    )

LOGGER = logging.getLogger("hcmrl.runner")

# Default experiment templates keyed by environment
DEFAULT_EXPERIMENT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "crafting": {
        "env_name": "crafting",
        "memory_config": {
            "immediate_size": 32,
            "short_term_size": 16,
            "long_term_size": 8,
            "embedding_dim": 128,
        },
        "policy_config": {
            "hidden_dim": 256,
            "memory_dim": 128,
            "continuous_actions": False,
        },
        "training_config": {
            "num_episodes": 100,
            "eval_interval": 10,
            "eval_episodes": 5,
            "env_kwargs": {
                "grid_size": 16,
                "max_steps": 2000,
                "resource_density": 0.1,
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
        },
    },
    "navigation": {
        "env_name": "navigation",
        "memory_config": {
            "immediate_size": 32,
            "short_term_size": 16,
            "long_term_size": 8,
            "embedding_dim": 128,
        },
        "policy_config": {
            "hidden_dim": 256,
            "memory_dim": 128,
            "continuous_actions": False,
        },
        "training_config": {
            "num_episodes": 100,
            "eval_interval": 10,
            "eval_episodes": 5,
            "env_kwargs": {
                "num_rooms": 9,
                "room_size": [7, 7],
                "max_steps": 2000,
                "num_keys": 3,
            },
            "exploration_kwargs": {
                "epsilon_start": 1.0,
                "epsilon_end": 0.05,
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
        },
    },
    "games": {
        "env_name": "games",
        "memory_config": {
            "immediate_size": 32,
            "short_term_size": 16,
            "long_term_size": 8,
            "embedding_dim": 128,
        },
        "policy_config": {
            "hidden_dim": 256,
            "memory_dim": 128,
            "continuous_actions": False,
        },
        "training_config": {
            "num_episodes": 100,
            "eval_interval": 10,
            "eval_episodes": 5,
            "env_kwargs": {
                "game_type": "connect_four",
                "max_steps": 2000,
                "opponent_type": "random",
            },
            "exploration_kwargs": {
                "epsilon_start": 1.0,
                "epsilon_end": 0.05,
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
        },
    },
    "story": {
        "env_name": "story",
        "memory_config": {
            "immediate_size": 32,
            "short_term_size": 16,
            "long_term_size": 8,
            "embedding_dim": 128,
        },
        "policy_config": {
            "hidden_dim": 256,
            "memory_dim": 128,
            "continuous_actions": False,
        },
        "training_config": {
            "num_episodes": 100,
            "eval_interval": 10,
            "eval_episodes": 5,
            "env_kwargs": {
                "num_characters": 5,
                "max_steps": 1500,
                "coherence_threshold": 0.6,
            },
            "exploration_kwargs": {
                "epsilon_start": 1.0,
                "epsilon_end": 0.05,
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
        },
    },
}


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=sorted(DEFAULT_EXPERIMENT_CONFIGS), default="crafting",
                        help="Target environment name (default: crafting)")
    parser.add_argument("--config", type=Path, default=None,
                        help="Optional experiment config (JSON/TOML/YAML)")
    parser.add_argument("--override", action="append", default=[],
                        help="Override config values, e.g. training_config.ppo_kwargs.lr=0.0001")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0],
                        help="List of random seeds (default: 0)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Number of training episodes (overrides config)")
    parser.add_argument("--eval-episodes", type=int, default=None,
                        help="Evaluation episodes per seed (overrides config)")
    parser.add_argument("--eval-interval", type=int, default=None,
                        help="Evaluation interval during training (overrides config)")
    parser.add_argument("--log-dir", type=Path, default=Path("results"),
                        help="Directory to store experiment logs")
    parser.add_argument("--run-tag", type=str, default=None,
                        help="Optional run tag appended to the log directory")
    parser.add_argument("--use-wandb", action="store_true",
                        help="Enable Weights & Biases logging (requires login)")
    parser.add_argument("--project", type=str, default="hcmrl",
                        help="W&B project name when --use-wandb is active")
    parser.add_argument("--notes", type=str, default="",
                        help="Free-form notes stored with the run metadata")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Increase verbosity (use -vv for debug)")
    return parser.parse_args(argv)


def load_experiment_config(args: argparse.Namespace) -> Dict[str, Any]:
    if args.config is not None:
        base = load_config_file(args.config)
    else:
        base = copy.deepcopy(DEFAULT_EXPERIMENT_CONFIGS[args.env])

    if args.override:
        base = apply_overrides(base, args.override)

    # Apply CLI overrides for episode counts
    training = base.setdefault("training_config", {})
    if args.episodes is not None:
        training["num_episodes"] = args.episodes
    if args.eval_interval is not None:
        training["eval_interval"] = args.eval_interval
    if args.eval_episodes is not None:
        training["eval_episodes"] = args.eval_episodes

    return base


def prepare_run_directory(log_dir: Path, env: str, run_tag: str | None) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    tag = run_tag or env
    run_dir = log_dir / env / f"{timestamp}_{tag}"
    ensure_directory(run_dir)
    return run_dir


def metrics_to_frame(metrics: Dict[str, List[float]]) -> pd.DataFrame:
    # Episode index starts at 1 for readability
    rows = {
        key: pd.Series(values)
        for key, values in metrics.items()
    }
    df = pd.DataFrame(rows)
    df.index.name = "episode"
    df.index += 1
    return df


def summarize_final_metrics(df: pd.DataFrame, window: int = 10) -> Dict[str, float]:
    tail = df.tail(window)
    summary = {}
    for column in df.columns:
        series = tail[column]
        summary[f"{column}_mean"] = float(series.mean())
        summary[f"{column}_std"] = float(series.std(ddof=0))
    return summary


def run_single_seed(seed: int, config: Dict[str, Any], use_wandb: bool) -> Tuple[pd.DataFrame, Dict[str, float]]:
    LOGGER.info("Starting seed %s", seed)
    set_global_seed(seed)

    experiment = ValidationExperiment(
        env_name=config["env_name"],
        memory_config=config["memory_config"],
        policy_config=config["policy_config"],
        training_config=config["training_config"],
        use_wandb=use_wandb,
    )

    metrics = experiment.train(
        num_episodes=config["training_config"].get("num_episodes", 100),
        eval_interval=config["training_config"].get("eval_interval", 10),
    )

    df = metrics_to_frame(metrics)
    eval_episodes = config["training_config"].get("eval_episodes", 5)
    eval_metrics = experiment.evaluate(num_episodes=eval_episodes)

    summary = summarize_final_metrics(df)
    summary.update({
        "eval_reward_mean": float(eval_metrics["reward"]),
        "eval_success_mean": float(eval_metrics["success"]),
        "seed": seed,
    })

    LOGGER.info(
        "Seed %s finished | reward=%.3f success=%.3f",
        seed,
        summary["eval_reward_mean"],
        summary["eval_success_mean"],
    )

    return df, summary


def aggregate_seed_summaries(summaries: List[Dict[str, float]]) -> Dict[str, float]:
    if not summaries:
        return {}

    df = pd.DataFrame(summaries)
    aggregate: Dict[str, float] = {}

    for column in df.columns:
        if column == "seed":
            continue
        aggregate[f"{column}_mean"] = float(df[column].mean())
        aggregate[f"{column}_std"] = float(df[column].std(ddof=0))

    aggregate["num_seeds"] = len(summaries)
    return aggregate


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    config = load_experiment_config(args)
    if args.use_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.project)
    LOGGER.info("Running experiment: env=%s seeds=%s", config["env_name"], args.seeds)

    run_dir = prepare_run_directory(args.log_dir, config["env_name"], args.run_tag)
    metadata = {
        "config": config,
        "seeds": args.seeds,
        "use_wandb": args.use_wandb,
        "project": args.project,
        "notes": args.notes,
    }
    dump_json(metadata, run_dir / "metadata.json")

    seed_summaries: List[Dict[str, float]] = []

    for seed in args.seeds:
        df, summary = run_single_seed(seed, config, args.use_wandb)
        seed_dir = run_dir / f"seed_{seed}"
        ensure_directory(seed_dir)
        df.to_csv(seed_dir / "training_metrics.csv")
        dump_json(summary, seed_dir / "summary.json")
        seed_summaries.append(summary)

    aggregate = aggregate_seed_summaries(seed_summaries)
    dump_json(aggregate, run_dir / "aggregate_summary.json")

    LOGGER.info("All runs complete. Aggregate summary saved to %s", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
