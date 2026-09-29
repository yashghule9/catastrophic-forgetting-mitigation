"""Initial source class prototypes P^(0): the per-class mean of source-train
embeddings, L2-normalized by default so cosine similarity reduces to a dot
product everywhere downstream (pseudo-labeling, the update rule, and the
nearest-prototype classifier all then agree on the same geometry)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_class_prototypes(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    normalize: bool = True,
) -> torch.Tensor:
    if normalize:
        embeddings = F.normalize(embeddings, dim=1)
    dim = embeddings.shape[1]
    prototypes = torch.zeros(num_classes, dim, dtype=embeddings.dtype)
    for c in range(num_classes):
        mask = labels == c
        if not mask.any():
            raise ValueError(f"Class {c} has no source-train samples; cannot form a prototype.")
        prototypes[c] = embeddings[mask].mean(dim=0)
    return F.normalize(prototypes, dim=1) if normalize else prototypes


def compute_class_covariances(
    embeddings: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """(C, D, D) per-class covariance. Not used by the nearest-prototype
    classifier today; scaffolding for a possible future Mahalanobis-distance
    extension, kept here so it lives next to the mean it would pair with."""
    dim = embeddings.shape[1]
    covariances = torch.zeros(num_classes, dim, dim, dtype=embeddings.dtype)
    for c in range(num_classes):
        mask = labels == c
        x = embeddings[mask]
        if x.shape[0] < 2:
            continue
        covariances[c] = torch.cov(x.T)
    return covariances
