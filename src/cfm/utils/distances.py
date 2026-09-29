"""Shared similarity/distance matrix ops used by pseudo-labeling and the
nearest-prototype classifier, so both agree on exactly the same math."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def similarity_matrix(z: torch.Tensor, p: torch.Tensor, distance: str = "cosine") -> torch.Tensor:
    """(N, C) matrix where higher = more similar / closer, regardless of `distance`.

    For "euclidean" this returns *negative* distance so that "higher is better"
    holds uniformly for callers (argmax always picks the best match).
    """
    if distance == "cosine":
        z = F.normalize(z, dim=1)
        p = F.normalize(p, dim=1)
        return z @ p.T
    if distance == "euclidean":
        return -torch.cdist(z, p)
    raise ValueError(f"Unknown distance: {distance}")
