"""Deterministic enforcement of the catastrophic-forgetting bound:

    Acc(D_s_val; P^(t+1)) >= Acc(D_s_val; P^(0)) - epsilon

Rather than hoping the bound holds, every candidate prototype update is
validated on the source-val split *before* being committed. If it would
violate the bound, the adaptation rate `alpha` is shrunk (a backtracking line
search) and retried; if even `min_alpha` still violates the bound, the round
is rejected outright and `P_current` is left untouched. This is cheap because
`eval_fn` here is just cached-embedding matrix ops — no backbone forward pass —
so retrying costs nothing.

Two strategies are provided:
  - "global_backtrack" (default): one shared alpha for all classes in a round,
    shrunk together on rejection. Simple to reason about and test — the v1
    mechanism.
  - "per_class_greedy": commits each class's update one at a time (in
    decreasing basin-size / confidence order isn't required — insertion order
    is fine, since each commit is independently re-validated), shrinking or
    skipping only a class whose own update would break the bound. Finer-grained
    retention at the cost of up to C validation calls per round instead of a
    handful of backtracking retries — still cheap for the same reason.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from cfm.prototypes.update import convex_update

EvalFn = Callable[[torch.Tensor], float]


@dataclass
class GuardResult:
    prototypes: torch.Tensor
    accepted: bool
    alpha_used: float
    candidate_acc: float
    attempts: int
    per_class_alpha: dict[int, float] | None = None


def apply_global_backtracking(
    p_current: torch.Tensor,
    class_modes: dict[int, torch.Tensor],
    alpha_init: float,
    alpha_shrink_factor: float,
    min_alpha: float,
    baseline_acc: float,
    epsilon: float,
    eval_fn: EvalFn,
    normalize: bool = True,
    max_attempts: int = 100,
) -> GuardResult:
    if not class_modes:
        return GuardResult(p_current, False, 0.0, eval_fn(p_current), 0)

    alpha = alpha_init
    for attempt in range(1, max_attempts + 1):
        candidate = p_current.clone()
        for c, mode in class_modes.items():
            candidate[c] = convex_update(p_current[c], mode, alpha, normalize)
        acc = eval_fn(candidate)
        if acc >= baseline_acc - epsilon:
            return GuardResult(candidate, True, alpha, acc, attempt)
        if alpha <= min_alpha:
            return GuardResult(p_current, False, 0.0, eval_fn(p_current), attempt)
        alpha = max(alpha * alpha_shrink_factor, min_alpha)

    # Exhausted max_attempts without satisfying the bound or bottoming out at
    # min_alpha (only possible with a pathological shrink_factor close to 1).
    return GuardResult(p_current, False, 0.0, eval_fn(p_current), max_attempts)


def apply_per_class_greedy(
    p_current: torch.Tensor,
    class_modes: dict[int, torch.Tensor],
    alpha_init: float,
    alpha_shrink_factor: float,
    min_alpha: float,
    baseline_acc: float,
    epsilon: float,
    eval_fn: EvalFn,
    normalize: bool = True,
    max_attempts_per_class: int = 20,
) -> GuardResult:
    if not class_modes:
        return GuardResult(p_current, False, 0.0, eval_fn(p_current), 0, {})

    prototypes = p_current.clone()
    per_class_alpha: dict[int, float] = {}
    total_attempts = 0
    any_committed = False

    for c, mode in class_modes.items():
        alpha = alpha_init
        for attempt in range(1, max_attempts_per_class + 1):
            total_attempts += 1
            candidate = prototypes.clone()
            candidate[c] = convex_update(prototypes[c], mode, alpha, normalize)
            acc = eval_fn(candidate)
            if acc >= baseline_acc - epsilon:
                prototypes = candidate
                per_class_alpha[c] = alpha
                any_committed = True
                break
            if alpha <= min_alpha:
                per_class_alpha[c] = 0.0
                break
            alpha = max(alpha * alpha_shrink_factor, min_alpha)

    final_acc = eval_fn(prototypes)
    return GuardResult(
        prototypes, any_committed, alpha_init, final_acc, total_attempts, per_class_alpha
    )
