#!/usr/bin/env python3
"""Run ablation studies for HCMRL with structured logging."""
from __future__ import annotations

import argparse
import copy
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from hcmrl.validation.experiments import run_ablation_study

try:  # Allow execution via module or script
    from .research_utils import (
        apply_overrides,
        configure_logging,
        dump_json,
        ensure_directory,
        load_config_file,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from research_utils import (
        apply_overrides,
        configure_logging,
        dump_json,
        ensure_directory,
        load_config_file,
    )

LOGGER = logging.getLogger("hcmrl.ablation")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="Ablation config file (JSON/TOML/YAML)")
    parser.add_argument("--override", action="append", default=[],
                        help="Override base_config values, e.g. memory_config.embedding_dim=64")
    parser.add_argument("--num-seeds", type=int, default=None,
                        help="Override number of random seeds")
    parser.add_argument("--log-dir", type=Path, default=Path("results"),
                        help="Directory to store ablation results")
    parser.add_argument("--run-tag", type=str, default="ablation",
                        help="Optional tag appended to output directory")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Increase verbosity (use -vv for debug)")
    return parser.parse_args(argv)


def prepare_run_directory(log_dir: Path, tag: str) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = log_dir / "ablations" / f"{timestamp}_{tag}"
    ensure_directory(run_dir)
    return run_dir


def reshape_results(
    results: Dict[str, Dict[str, List[float]]],
    ablation_params: Dict[str, List[Any]],
    num_seeds: int,
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []

    for param_name, metrics in results.items():
        values = ablation_params.get(param_name, [])
        if not values:
            LOGGER.warning("No parameter values recorded for %s; skipping", param_name)
            continue

        for value_idx, value in enumerate(values):
            start = value_idx * num_seeds
            end = start + num_seeds

            for seed_offset, metric_index in enumerate(range(start, end)):
                record: Dict[str, Any] = {
                    "parameter": param_name,
                    "value": value,
                    "seed": seed_offset,
                }
                for metric_name, metric_values in metrics.items():
                    if metric_index >= len(metric_values):
                        LOGGER.error(
                            "Metric list too short for %s (%s) at index %s",
                            param_name,
                            metric_name,
                            metric_index,
                        )
                        continue
                    record[metric_name] = float(metric_values[metric_index])
                records.append(record)

    return pd.DataFrame.from_records(records)


def summarize_ablation(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    summary = (
        df.groupby(["parameter", "value"])
        .agg(["mean", "std"])
        .sort_index()
    )
    summary.columns = ["_".join(filter(None, col)).strip("_") for col in summary.columns]
    summary = summary.reset_index()
    return summary


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    config_data = load_config_file(args.config)
    if "base_config" not in config_data or "ablation_params" not in config_data:
        raise ValueError("Config must contain 'base_config' and 'ablation_params'")

    base_config = copy.deepcopy(config_data["base_config"])
    if args.override:
        base_config = apply_overrides(base_config, args.override)

    ablation_params = config_data["ablation_params"]
    num_seeds = args.num_seeds or config_data.get("num_seeds", 3)

    LOGGER.info(
        "Running ablation study with %s parameters and %s seeds", len(ablation_params), num_seeds
    )

    results = run_ablation_study(base_config, ablation_params, num_seeds=num_seeds)

    run_dir = prepare_run_directory(args.log_dir, args.run_tag)
    ensure_directory(run_dir)

    per_seed_df = reshape_results(results, ablation_params, num_seeds)
    per_seed_path = run_dir / "ablation_per_seed.csv"
    per_seed_df.to_csv(per_seed_path, index=False)

    summary_df = summarize_ablation(per_seed_df)
    summary_path = run_dir / "ablation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    metadata = {
        "base_config": base_config,
        "ablation_params": ablation_params,
        "num_seeds": num_seeds,
        "per_seed_metrics": str(per_seed_path),
        "summary_metrics": str(summary_path),
    }
    dump_json(metadata, run_dir / "metadata.json")

    LOGGER.info("Ablation study complete. Results saved to %s", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
