"""Experiment configuration: a single flat dataclass loaded from YAML with
optional CLI overrides. Deliberately simpler than Hydra/OmegaConf — this is a
learning-focused repo and every field here is meant to be readable at a glance.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass
class ExperimentConfig:
    # Data
    source_domain: str = "Real World"
    target_domain: str = "Clipart"
    data_root: str = "data/raw"
    cache_root: str = "data/cache"
    results_root: str = "results"
    val_split: float = 0.2
    seed: int = 42

    # Backbone / features
    model_name: str = "microsoft/beit-base-patch16-224"
    pooling: str = "cls"  # "cls" or "mean_patch"
    batch_size: int = 64
    device: str = "auto"

    # Prototypes / classifier
    normalize_prototypes: bool = True
    distance: str = "cosine"  # "cosine" or "euclidean"

    # Mean shift
    mean_shift_kernel: str = "gaussian"  # "gaussian" or "flat"
    bandwidth_quantile: float = 0.3
    mean_shift_max_iter: int = 100
    mean_shift_tol: float = 1e-4

    # Pseudo-labeling
    confidence_metric: str = "margin"  # "margin" or "top1"
    confidence_threshold: float = 0.05
    min_points_per_class: int = 3

    # Adaptation loop / forgetting guard
    epsilon: float = 0.013
    alpha_max: float = 0.3
    alpha_schedule: str = "constant"  # "constant", "linear_decay", "cosine_decay"
    alpha_shrink_factor: float = 0.5
    min_alpha: float = 1e-3
    num_rounds: int = 10
    guard_strategy: str = "global_backtrack"  # or "per_class_greedy"
    save_every_round: bool = False

    @property
    def model_tag(self) -> str:
        return self.model_name.replace("/", "__") + f"_{self.pooling}"

    @property
    def run_tag(self) -> str:
        src = self.source_domain.replace(" ", "")
        tgt = self.target_domain.replace(" ", "")
        return f"{src}_to_{tgt}"


def load_config(path: str | Path | None, overrides: dict | None = None) -> ExperimentConfig:
    data: dict = {}
    if path is not None:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
    if overrides:
        data.update({k: v for k, v in overrides.items() if v is not None})
    valid_keys = {f.name for f in fields(ExperimentConfig)}
    unknown = set(data) - valid_keys
    if unknown:
        raise ValueError(f"Unknown config key(s): {sorted(unknown)}")
    return ExperimentConfig(**data)


def _str_to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def add_config_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file")
    for f in fields(ExperimentConfig):
        field_type = _str_to_bool if isinstance(f.default, bool) else type(f.default)
        parser.add_argument(f"--{f.name}", type=field_type, default=None,
                             help=f"Override '{f.name}' (default: {f.default})")
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    overrides = {f.name: getattr(args, f.name, None) for f in fields(ExperimentConfig)}
    return load_config(args.config, overrides)
