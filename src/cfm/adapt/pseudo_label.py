"""Unsupervised target-to-class association: nearest-prototype pseudo-labeling
with a confidence filter, so only target points the current prototypes are
fairly sure about feed into that round's mean-shift subsets."""

from __future__ import annotations

import torch

from cfm.utils.distances import similarity_matrix


def pseudo_label_targets(
    z_t: torch.Tensor, p: torch.Tensor, distance: str = "cosine", confidence_metric: str = "margin"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (labels, confidence), both shape (N,).

    confidence_metric:
      - "margin" (default): similarity to the best prototype minus similarity to
        the second-best. Robust to per-class/per-domain scale differences in raw
        similarity under distribution shift, unlike a fixed absolute threshold.
      - "top1": the raw best-prototype similarity score.
    """
    sims = similarity_matrix(z_t, p, distance)  # (N, C), higher = better
    top2 = sims.topk(k=min(2, sims.shape[1]), dim=1)
    labels = top2.indices[:, 0]
    if confidence_metric == "margin" and sims.shape[1] > 1:
        confidence = top2.values[:, 0] - top2.values[:, 1]
    elif confidence_metric in {"margin", "top1"}:
        confidence = top2.values[:, 0]
    else:
        raise ValueError(f"Unknown confidence_metric: {confidence_metric}")
    return labels, confidence


def filter_by_confidence(confidence: torch.Tensor, threshold: float) -> torch.Tensor:
    """Boolean mask of points confident enough to trust this round."""
    return confidence >= threshold
