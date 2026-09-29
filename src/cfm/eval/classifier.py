"""Nearest-prototype classifier — the decision rule used everywhere: to report
accuracy, and (via pseudo-labeling) to drive adaptation itself. Cosine is the
default distance because embeddings/prototypes are L2-normalized, making cosine
similarity a monotonic function of the same dot product used in the update
rule and pseudo-labeling; euclidean is kept as a configurable ablation."""

from __future__ import annotations

import torch

from cfm.utils.distances import similarity_matrix


def nearest_prototype_predict(z: torch.Tensor, p: torch.Tensor, distance: str = "cosine") -> torch.Tensor:
    sims = similarity_matrix(z, p, distance)
    return sims.argmax(dim=1)


def accuracy(z: torch.Tensor, y_true: torch.Tensor, p: torch.Tensor, distance: str = "cosine") -> float:
    preds = nearest_prototype_predict(z, p, distance)
    return (preds == y_true).float().mean().item()
