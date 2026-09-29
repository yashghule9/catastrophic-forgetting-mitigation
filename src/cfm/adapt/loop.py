"""Orchestrates the T-round adaptation loop:

    for r in 1..T:
        pseudo-label target embeddings using the current prototypes
        for each class with enough confidently-assigned target points:
            find that class's target mode via mean shift
        propose a candidate prototype update (guarded so the forgetting bound
        is never violated) and commit it if accepted

Target labels, if supplied, are used ONLY to log a monitoring accuracy each
round — never to drive any decision the algorithm makes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from cfm.adapt.forgetting_guard import apply_global_backtracking, apply_per_class_greedy
from cfm.adapt.pseudo_label import filter_by_confidence, pseudo_label_targets
from cfm.clustering.mean_shift import mean_shift_mode_for_class
from cfm.config import ExperimentConfig
from cfm.eval.classifier import accuracy
from cfm.utils.logging import RoundLogger

_GUARD_STRATEGIES = {
    "global_backtrack": apply_global_backtracking,
    "per_class_greedy": apply_per_class_greedy,
}


def alpha_for_round(round_idx: int, num_rounds: int, alpha_max: float, schedule: str) -> float:
    if schedule == "constant":
        return alpha_max
    t = (round_idx - 1) / max(1, num_rounds - 1)
    if schedule == "linear_decay":
        return alpha_max * (1 - t)
    if schedule == "cosine_decay":
        return alpha_max * 0.5 * (1 + math.cos(math.pi * t))
    raise ValueError(f"Unknown alpha_schedule: {schedule}")


@dataclass
class AdaptationResult:
    p0: torch.Tensor
    p_final: torch.Tensor
    baseline_source_val_acc: float
    prototype_history: list[torch.Tensor] | None = None


def run_adaptation(
    p0: torch.Tensor,
    z_source_val: torch.Tensor,
    y_source_val: torch.Tensor,
    z_target: torch.Tensor,
    config: ExperimentConfig,
    y_target: torch.Tensor | None = None,
    logger: RoundLogger | None = None,
) -> AdaptationResult:
    if config.guard_strategy not in _GUARD_STRATEGIES:
        raise ValueError(f"Unknown guard_strategy: {config.guard_strategy}")
    guard_fn = _GUARD_STRATEGIES[config.guard_strategy]

    def eval_fn(p: torch.Tensor) -> float:
        return accuracy(z_source_val, y_source_val, p, config.distance)

    baseline_acc = eval_fn(p0)
    p_current = p0.clone()
    num_classes = p0.shape[0]
    history = [p0.clone()] if config.save_every_round else None

    for r in range(1, config.num_rounds + 1):
        alpha_r = alpha_for_round(r, config.num_rounds, config.alpha_max, config.alpha_schedule)

        labels, confidence = pseudo_label_targets(
            z_target, p_current, config.distance, config.confidence_metric
        )
        confident_mask = filter_by_confidence(confidence, config.confidence_threshold)

        class_modes: dict[int, torch.Tensor] = {}
        for c in range(num_classes):
            class_mask = confident_mask & (labels == c)
            if int(class_mask.sum()) < config.min_points_per_class:
                continue
            z_c = z_target[class_mask]
            class_modes[c] = mean_shift_mode_for_class(
                z_c,
                bandwidth_quantile=config.bandwidth_quantile,
                kernel=config.mean_shift_kernel,
                max_iter=config.mean_shift_max_iter,
                tol=config.mean_shift_tol,
            )

        result = guard_fn(
            p_current=p_current,
            class_modes=class_modes,
            alpha_init=alpha_r,
            alpha_shrink_factor=config.alpha_shrink_factor,
            min_alpha=config.min_alpha,
            baseline_acc=baseline_acc,
            epsilon=config.epsilon,
            eval_fn=eval_fn,
            normalize=config.normalize_prototypes,
        )
        p_current = result.prototypes

        target_acc = accuracy(z_target, y_target, p_current, config.distance) if y_target is not None else None

        if history is not None:
            history.append(p_current.clone())

        if logger is not None:
            logger.log(
                round=r,
                alpha_init=alpha_r,
                alpha_used=result.alpha_used,
                accepted=result.accepted,
                num_classes_updated=len(class_modes),
                source_val_acc=result.candidate_acc,
                baseline_source_val_acc=baseline_acc,
                target_acc=target_acc,
                forgetting_bound_ok=result.candidate_acc >= baseline_acc - config.epsilon,
            )

    return AdaptationResult(
        p0=p0, p_final=p_current, baseline_source_val_acc=baseline_acc, prototype_history=history
    )
