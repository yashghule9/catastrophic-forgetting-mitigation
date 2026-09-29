"""On-disk caching of extracted embeddings, so the (only) GPU-heavy step in the
whole pipeline — the backbone forward pass — runs at most once per domain."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import torch


class EmbeddingBundle(TypedDict):
    embeddings: torch.Tensor  # (N, D)
    labels: torch.Tensor  # (N,)
    paths: list[str]
    domain: str
    model_name: str


def embeddings_path(cache_root: str | Path, domain: str, model_tag: str) -> Path:
    safe_domain = domain.replace(" ", "")
    return Path(cache_root) / "embeddings" / f"{safe_domain}__{model_tag}.pt"


def save_embeddings(
    path: str | Path,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    paths: list[str],
    domain: str,
    model_name: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle: EmbeddingBundle = {
        "embeddings": embeddings,
        "labels": labels,
        "paths": paths,
        "domain": domain,
        "model_name": model_name,
    }
    torch.save(bundle, path)


def load_embeddings(path: str | Path) -> EmbeddingBundle:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No cached embeddings at {path}. Run scripts/extract_features.py first."
        )
    return torch.load(path, weights_only=False)
