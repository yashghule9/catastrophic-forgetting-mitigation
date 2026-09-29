"""Mean shift clustering, implemented from scratch.

Mean shift is a density-based mode-seeking algorithm: starting from every point
in the dataset, it repeatedly moves each point toward the weighted mean of its
neighborhood (weighted by a kernel over distance), which is equivalent to
gradient ascent on a kernel density estimate of the data. Points converge to
the nearest mode (peak) of that estimated density.

We use it here instead of a plain per-class mean because pseudo-labeling is
noisy: a handful of target points can be assigned to the wrong class by the
current (imperfect) prototypes. A plain mean is dragged by every outlier; the
*largest-basin mode* found by mean shift is not, because outliers form their
own small, separate basin rather than perturbing the dominant one.

`sklearn.cluster.MeanShift` is used only in tests, as a correctness oracle on
synthetic data — never imported by this module or anywhere in the runtime path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def estimate_bandwidth(x: torch.Tensor, quantile: float = 0.3) -> float:
    """k-NN-distance heuristic bandwidth (mirrors the idea behind
    `sklearn.cluster.estimate_bandwidth`): for each point, find the distance to
    its k-th nearest neighbor (k = quantile * N), then take the median of those
    per-point k-NN distances across the dataset.
    """
    n = x.shape[0]
    if n < 2:
        return 1.0
    k = max(1, min(n - 1, round(quantile * n)))
    dists = torch.cdist(x, x)
    knn_dists = dists.topk(k + 1, largest=False).values[:, 1:]  # drop self (distance 0)
    per_point_kth = knn_dists.max(dim=1).values
    bandwidth = per_point_kth.median().item()
    return bandwidth if bandwidth > 0 else 1e-6


@dataclass
class MeanShiftResult:
    modes: torch.Tensor  # (K, D) unique modes, sorted by basin size descending
    basin_sizes: torch.Tensor  # (K,) number of points converging to each mode
    labels: torch.Tensor  # (N,) index into `modes` for each input point


def _kernel_weights(dists: torch.Tensor, bandwidth: float, kernel: str) -> torch.Tensor:
    if kernel == "gaussian":
        return torch.exp(-(dists**2) / (2 * bandwidth**2))
    if kernel == "flat":
        return (dists < bandwidth).float()
    raise ValueError(f"Unknown kernel: {kernel}")


def mean_shift_modes(
    x: torch.Tensor,
    bandwidth: float,
    kernel: str = "gaussian",
    max_iter: int = 100,
    tol: float = 1e-4,
    merge_radius_factor: float = 0.5,
) -> MeanShiftResult:
    """Runs mode-seeking ascent from every point in `x` (seeding from all points
    is fine at the per-class-subset scale this project operates at — tens to a
    few hundred points), then merges trajectories that converge close together
    into a single mode.
    """
    n = x.shape[0]
    if n == 0:
        raise ValueError("mean_shift_modes called with an empty point set")
    if n == 1:
        return MeanShiftResult(
            modes=x.clone(), basin_sizes=torch.tensor([1]), labels=torch.tensor([0])
        )

    trajectories = x.clone()
    for _ in range(max_iter):
        dists = torch.cdist(trajectories, x)
        weights = _kernel_weights(dists, bandwidth, kernel)
        weight_sums = weights.sum(dim=1, keepdim=True)
        # A trajectory that has drifted beyond every point's kernel support keeps
        # its current position rather than producing a NaN from a 0/0 divide.
        safe_sums = weight_sums.clamp_min(1e-12)
        new_trajectories = (weights @ x) / safe_sums
        stuck = weight_sums.squeeze(1) <= 1e-12
        new_trajectories[stuck] = trajectories[stuck]

        shift = (new_trajectories - trajectories).norm(dim=1).max().item()
        trajectories = new_trajectories
        if shift < tol:
            break

    modes, basin_sizes, labels = _merge_close_points(
        trajectories, radius=bandwidth * merge_radius_factor
    )
    order = torch.argsort(basin_sizes, descending=True)
    inverse_order = torch.empty_like(order)
    inverse_order[order] = torch.arange(order.shape[0])
    return MeanShiftResult(
        modes=modes[order], basin_sizes=basin_sizes[order], labels=inverse_order[labels]
    )


def _merge_close_points(
    points: torch.Tensor, radius: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Greedy union of converged trajectories within `radius` of each other,
    representing each cluster by the mean of its member trajectories."""
    n = points.shape[0]
    assigned = torch.full((n,), -1, dtype=torch.long)
    centers: list[torch.Tensor] = []
    for i in range(n):
        if assigned[i] != -1:
            continue
        dists = (points - points[i]).norm(dim=1)
        member_mask = (dists < radius) & (assigned == -1)
        assigned[member_mask] = len(centers)
        centers.append(points[member_mask].mean(dim=0))
    modes = torch.stack(centers)
    basin_sizes = torch.bincount(assigned, minlength=len(centers))
    return modes, basin_sizes, assigned


def mean_shift_mode_for_class(
    z_c: torch.Tensor,
    bandwidth: float | None = None,
    bandwidth_quantile: float = 0.3,
    kernel: str = "gaussian",
    max_iter: int = 100,
    tol: float = 1e-4,
) -> torch.Tensor:
    """Returns a single representative point for a per-class subset of target
    embeddings: the mode of its largest density basin. This is the
    `MeanShift(Z_c^t)` term in the prototype update rule.
    """
    if z_c.shape[0] == 0:
        raise ValueError("mean_shift_mode_for_class called with an empty subset")
    if bandwidth is None:
        bandwidth = estimate_bandwidth(z_c, quantile=bandwidth_quantile)
    result = mean_shift_modes(z_c, bandwidth, kernel=kernel, max_iter=max_iter, tol=tol)
    return result.modes[0]  # modes are sorted by basin size descending
