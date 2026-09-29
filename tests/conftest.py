"""Synthetic-domain fixtures so the whole test suite runs offline: no GPU, no
network, no Office-Home download. Each "domain" is Gaussian blobs in a toy
embedding space, well-separated enough that a nearest-prototype classifier is
meaningfully testable, with a small controlled shift between source and target
so adaptation has something real to close."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch


@dataclass
class SyntheticDomains:
    num_classes: int
    dim: int
    z_source_train: torch.Tensor
    y_source_train: torch.Tensor
    z_source_val: torch.Tensor
    y_source_val: torch.Tensor
    z_target: torch.Tensor
    y_target: torch.Tensor  # ground truth, for test assertions only — never fed to adaptation


def make_synthetic_domains(
    num_classes: int = 5,
    dim: int = 8,
    samples_per_class: int = 40,
    class_spacing: float = 8.0,
    shift_scale: float = 2.0,
    noise_std: float = 0.5,
    val_fraction: float = 0.25,
    seed: int = 0,
) -> SyntheticDomains:
    g = torch.Generator().manual_seed(seed)

    # Well-separated class centers (random directions, well-separated by construction
    # since dim >> num_classes makes near-orthogonality likely at this scale).
    centers = torch.randn(num_classes, dim, generator=g) * class_spacing
    # A per-class shift representing the target domain's distribution shift.
    shifts = torch.randn(num_classes, dim, generator=g) * shift_scale

    def sample(n_per_class: int, base_centers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        zs, ys = [], []
        for c in range(num_classes):
            noise = torch.randn(n_per_class, dim, generator=g) * noise_std
            zs.append(base_centers[c] + noise)
            ys.append(torch.full((n_per_class,), c, dtype=torch.long))
        return torch.cat(zs), torch.cat(ys)

    z_source, y_source = sample(samples_per_class, centers)
    z_target, y_target = sample(samples_per_class, centers + shifts)

    n_val = max(1, round(samples_per_class * val_fraction))
    val_mask = torch.zeros(samples_per_class, dtype=torch.bool)
    val_mask[:n_val] = True
    val_mask = val_mask.repeat(num_classes)

    return SyntheticDomains(
        num_classes=num_classes,
        dim=dim,
        z_source_train=z_source[~val_mask],
        y_source_train=y_source[~val_mask],
        z_source_val=z_source[val_mask],
        y_source_val=y_source[val_mask],
        z_target=z_target,
        y_target=y_target,
    )


@pytest.fixture
def synthetic_domains() -> SyntheticDomains:
    return make_synthetic_domains()
