"""The weighted prototype update rule itself:

    p_c^(t+1) = (1 - alpha) * p_c^(t) + alpha * MeanShift(Z_c^t)

Deliberately a tiny, pure function — it's the mathematical core of the whole
project, so it should be trivial to read, test, and reason about in isolation
from the mean-shift/pseudo-labeling/forgetting-guard machinery around it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def convex_update(
    p: torch.Tensor, mode: torch.Tensor, alpha: float, normalize: bool = True
) -> torch.Tensor:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    updated = (1.0 - alpha) * p + alpha * mode
    return F.normalize(updated, dim=-1) if normalize else updated
