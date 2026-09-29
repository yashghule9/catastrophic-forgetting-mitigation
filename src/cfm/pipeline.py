"""Small glue helpers shared by the CLI scripts: resolving cache paths and
loading the cached embeddings/splits/prototypes a given pipeline stage needs.
Kept separate from the algorithmic modules so those stay import-light and
testable without touching disk."""

from __future__ import annotations

from pathlib import Path

import torch

from cfm.config import ExperimentConfig
from cfm.data.office_home import load_split
from cfm.features.cache import EmbeddingBundle, embeddings_path, load_embeddings


def prototypes_path(config: ExperimentConfig, tag: str = "P0") -> Path:
    safe_source = config.source_domain.replace(" ", "")
    return Path(config.cache_root) / "prototypes" / f"{tag}_{safe_source}_{config.model_tag}.pt"


def split_path(config: ExperimentConfig) -> Path:
    safe_source = config.source_domain.replace(" ", "")
    return Path(config.cache_root) / "splits" / f"{safe_source}_seed{config.seed}.json"


def source_split_embeddings(
    config: ExperimentConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (z_train, y_train, z_val, y_val) for the source domain."""
    bundle: EmbeddingBundle = load_embeddings(
        embeddings_path(config.cache_root, config.source_domain, config.model_tag)
    )
    train_idx, val_idx = load_split(split_path(config))
    z, y = bundle["embeddings"], bundle["labels"]
    return z[train_idx], y[train_idx], z[val_idx], y[val_idx]


def target_embeddings(config: ExperimentConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (z_target, y_target). Target labels exist in Office-Home and are
    used ONLY for reporting/monitoring accuracy — never fed into adaptation."""
    bundle: EmbeddingBundle = load_embeddings(
        embeddings_path(config.cache_root, config.target_domain, config.model_tag)
    )
    return bundle["embeddings"], bundle["labels"]


def save_prototypes(path: str | Path, prototypes: torch.Tensor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prototypes, path)


def load_prototypes(path: str | Path) -> torch.Tensor:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No cached prototypes at {path}. Run scripts/init_prototypes.py first.")
    return torch.load(path, weights_only=False)
