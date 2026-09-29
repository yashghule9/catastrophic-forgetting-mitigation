"""Correctness tests for the from-scratch mean shift implementation, cross-
checked against sklearn.cluster.MeanShift (a dev-only dependency — never used
by the library at runtime, only here as an independent oracle)."""

import numpy as np
import pytest
import torch
from sklearn.cluster import MeanShift as SkMeanShift
from sklearn.cluster import estimate_bandwidth as sk_estimate_bandwidth
from sklearn.datasets import make_blobs

from cfm.clustering.mean_shift import (
    estimate_bandwidth,
    mean_shift_mode_for_class,
    mean_shift_modes,
)


def test_single_cluster_converges_to_mean():
    torch.manual_seed(0)
    x = torch.randn(50, 4) * 0.5 + torch.tensor([3.0, -1.0, 0.0, 2.0])
    bandwidth = estimate_bandwidth(x, quantile=0.3)
    result = mean_shift_modes(x, bandwidth)
    assert result.modes.shape[0] == 1
    assert torch.allclose(result.modes[0], x.mean(dim=0), atol=0.3)


def test_bandwidth_matches_sklearn_heuristic():
    x_np, _ = make_blobs(n_samples=150, centers=3, cluster_std=1.0, random_state=1)
    x = torch.tensor(x_np, dtype=torch.float32)
    ours = estimate_bandwidth(x, quantile=0.3)
    sk = sk_estimate_bandwidth(x_np, quantile=0.3, random_state=1)
    assert ours == pytest.approx(sk, rel=0.25)


def test_modes_agree_with_sklearn_on_blobs():
    x_np, _y_np = make_blobs(
        n_samples=300, centers=4, cluster_std=1.0, center_box=(-15, 15), random_state=42
    )
    x = torch.tensor(x_np, dtype=torch.float32)
    bandwidth = sk_estimate_bandwidth(x_np, quantile=0.3, random_state=42)

    ours = mean_shift_modes(x, bandwidth=bandwidth, kernel="flat")
    sk_model = SkMeanShift(bandwidth=bandwidth).fit(x_np)
    sk_modes = np.unique(sk_model.cluster_centers_, axis=0)

    # Every sklearn mode should have a close match among ours, and vice versa —
    # the two implementations should discover essentially the same density modes.
    our_modes_np = ours.modes.numpy()
    for m in sk_modes:
        dists = np.linalg.norm(our_modes_np - m, axis=1)
        assert dists.min() < bandwidth, f"no matching mode found near sklearn mode {m}"
    for m in our_modes_np:
        dists = np.linalg.norm(sk_modes - m, axis=1)
        assert dists.min() < bandwidth, f"no matching sklearn mode found near our mode {m}"


def test_largest_basin_mode_is_robust_to_outliers():
    torch.manual_seed(2)
    main_cluster = torch.randn(40, 3) * 0.4 + torch.tensor([5.0, 5.0, 5.0])
    outliers = torch.randn(5, 3) * 0.3 + torch.tensor([-10.0, -10.0, -10.0])
    z_c = torch.cat([main_cluster, outliers])

    mode = mean_shift_mode_for_class(z_c, bandwidth_quantile=0.3)
    plain_mean = z_c.mean(dim=0)

    # The largest-basin mode should land near the main cluster's true center,
    # while the naive mean is measurably dragged toward the outliers.
    true_center = torch.tensor([5.0, 5.0, 5.0])
    assert (mode - true_center).norm() < (plain_mean - true_center).norm()
    assert (mode - true_center).norm() < 1.0


def test_empty_class_subset_raises():
    with pytest.raises(ValueError):
        mean_shift_mode_for_class(torch.empty(0, 4))


def test_single_point_subset_is_trivial():
    z_c = torch.tensor([[1.0, 2.0, 3.0]])
    mode = mean_shift_mode_for_class(z_c)
    assert torch.allclose(mode, z_c[0])
