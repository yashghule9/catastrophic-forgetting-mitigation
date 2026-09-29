import pytest
import torch
import torch.nn.functional as F

from cfm.prototypes.update import convex_update


def test_alpha_zero_is_noop():
    p = torch.tensor([1.0, 2.0, 3.0])
    mode = torch.tensor([9.0, 9.0, 9.0])
    out = convex_update(p, mode, alpha=0.0, normalize=False)
    assert torch.allclose(out, p)


def test_alpha_one_is_full_replacement():
    p = torch.tensor([1.0, 2.0, 3.0])
    mode = torch.tensor([9.0, 9.0, 9.0])
    out = convex_update(p, mode, alpha=1.0, normalize=False)
    assert torch.allclose(out, mode)


def test_alpha_half_is_midpoint():
    p = torch.tensor([0.0, 0.0])
    mode = torch.tensor([2.0, 4.0])
    out = convex_update(p, mode, alpha=0.5, normalize=False)
    assert torch.allclose(out, torch.tensor([1.0, 2.0]))


def test_normalize_produces_unit_vector():
    p = F.normalize(torch.tensor([1.0, 0.0]), dim=-1)
    mode = F.normalize(torch.tensor([0.0, 1.0]), dim=-1)
    out = convex_update(p, mode, alpha=0.5, normalize=True)
    assert torch.allclose(out.norm(), torch.tensor(1.0), atol=1e-6)


def test_alpha_out_of_range_raises():
    p = torch.zeros(3)
    mode = torch.ones(3)
    with pytest.raises(ValueError):
        convex_update(p, mode, alpha=1.5)
    with pytest.raises(ValueError):
        convex_update(p, mode, alpha=-0.1)
