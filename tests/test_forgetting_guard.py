"""These tests drive the guard with hand-crafted `eval_fn`s rather than a real
classifier, so the accept/shrink/reject control flow can be pinned down
exactly (the guard doesn't care what eval_fn measures — it only needs a
callable prototype-matrix -> accuracy)."""

import torch

from cfm.adapt.forgetting_guard import apply_global_backtracking, apply_per_class_greedy


def _base_prototypes():
    p = torch.zeros(2, 3)
    p[0] = torch.tensor([1.0, 0.0, 0.0])
    p[1] = torch.tensor([0.0, 1.0, 0.0])
    return p


def test_accepts_immediately_when_update_never_hurts():
    p = _base_prototypes()
    mode0 = torch.tensor([0.0, 0.0, 1.0])

    result = apply_global_backtracking(
        p_current=p, class_modes={0: mode0}, alpha_init=0.3, alpha_shrink_factor=0.5,
        min_alpha=0.01, baseline_acc=0.9, epsilon=0.1,
        eval_fn=lambda cand: 0.9,  # always exactly at baseline, always within bound
        normalize=False,
    )
    assert result.accepted
    assert result.alpha_used == 0.3
    assert result.attempts == 1
    assert torch.allclose(result.prototypes[0], p[0] + 0.3 * (mode0 - p[0]))


def test_shrinks_alpha_until_bound_is_satisfied():
    p = _base_prototypes()
    mode0 = torch.tensor([0.0, 0.0, 1.0])
    diff_norm = (mode0 - p[0]).norm().item()

    def eval_fn(candidate: torch.Tensor) -> float:
        # With normalize=False, candidate[0] - p[0] == alpha * (mode0 - p[0]) exactly,
        # so this recovers the effective alpha and makes accuracy drop linearly with it.
        alpha_est = (candidate[0] - p[0]).norm().item() / diff_norm
        return 0.9 - 0.5 * alpha_est

    result = apply_global_backtracking(
        p_current=p, class_modes={0: mode0}, alpha_init=0.3, alpha_shrink_factor=0.5,
        min_alpha=0.01, baseline_acc=0.9, epsilon=0.1, eval_fn=eval_fn, normalize=False,
    )
    # alpha=0.3 -> acc=0.75 (fails, needs >=0.8); alpha=0.15 -> acc=0.825 (passes)
    assert result.accepted
    assert result.alpha_used == 0.15
    assert result.attempts == 2
    assert torch.allclose(result.prototypes[0], p[0] + 0.15 * (mode0 - p[0]), atol=1e-6)
    assert torch.allclose(result.prototypes[1], p[1])  # untouched class stays untouched


def test_rejects_and_leaves_p_current_untouched_when_bound_unreachable():
    p = _base_prototypes()
    mode0 = torch.tensor([0.0, 0.0, 1.0])

    result = apply_global_backtracking(
        p_current=p, class_modes={0: mode0}, alpha_init=0.3, alpha_shrink_factor=0.5,
        min_alpha=0.01, baseline_acc=0.9, epsilon=0.1,
        eval_fn=lambda cand: 0.0,  # always fails, no matter how small alpha gets
        normalize=False,
    )
    assert not result.accepted
    assert result.alpha_used == 0.0
    assert torch.equal(result.prototypes, p)  # P_current is never corrupted on rejection


def test_no_class_modes_is_an_immediate_no_op():
    p = _base_prototypes()
    result = apply_global_backtracking(
        p_current=p, class_modes={}, alpha_init=0.3, alpha_shrink_factor=0.5,
        min_alpha=0.01, baseline_acc=0.9, epsilon=0.1, eval_fn=lambda cand: 0.9,
    )
    assert not result.accepted
    assert result.attempts == 0
    assert torch.equal(result.prototypes, p)


def test_per_class_greedy_commits_easy_class_and_skips_impossible_class():
    p = torch.zeros(3, 3)
    p[0] = torch.tensor([1.0, 0.0, 0.0])
    p[1] = torch.tensor([0.0, 1.0, 0.0])
    p[2] = torch.tensor([0.0, 0.0, 1.0])
    mode0 = torch.tensor([2.0, 0.0, 0.0])  # class 0: harmless update, should commit fully
    mode1 = torch.tensor([9.0, 9.0, 9.0])  # class 1: any change is penalized

    def eval_fn(candidate: torch.Tensor) -> float:
        # Penalize only movement of prototype 1 away from its original position.
        penalty = (candidate[1] - p[1]).norm().item()
        return 0.9 - 10 * penalty

    result = apply_per_class_greedy(
        p_current=p, class_modes={0: mode0, 1: mode1}, alpha_init=0.3,
        alpha_shrink_factor=0.5, min_alpha=0.01, baseline_acc=0.9, epsilon=0.1,
        eval_fn=eval_fn, normalize=False,
    )
    assert result.accepted  # class 0's update did commit
    assert result.per_class_alpha[0] == 0.3
    assert result.per_class_alpha[1] == 0.0  # class 1 was skipped entirely
    assert torch.allclose(result.prototypes[0], p[0] + 0.3 * (mode0 - p[0]))
    assert torch.allclose(result.prototypes[1], p[1])  # left untouched
