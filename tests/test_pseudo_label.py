import pytest
import torch

from cfm.adapt.pseudo_label import filter_by_confidence, pseudo_label_targets


def _toy_prototypes():
    # Three well-separated 2D prototype directions.
    return torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])


def test_unambiguous_point_gets_correct_label_and_high_margin():
    p = _toy_prototypes()
    z = torch.tensor([[10.0, 0.1]])  # close to prototype 0
    labels, confidence = pseudo_label_targets(z, p, distance="cosine", confidence_metric="margin")
    assert labels[0].item() == 0
    assert confidence[0].item() > 0.5


def test_ambiguous_point_gets_low_margin():
    p = _toy_prototypes()
    # Exactly between prototypes 0 and 1 in angle.
    z = torch.tensor([[1.0, 1.0]])
    _labels, confidence = pseudo_label_targets(z, p, distance="cosine", confidence_metric="margin")
    assert confidence[0].item() == pytest.approx(0.0, abs=1e-5)


def test_confidence_filter_keeps_only_confident_points():
    confidence = torch.tensor([0.9, 0.01, 0.2, -0.1])
    mask = filter_by_confidence(confidence, threshold=0.1)
    assert mask.tolist() == [True, False, True, False]


def test_top1_confidence_metric_uses_raw_similarity():
    p = _toy_prototypes()
    z = torch.tensor([[10.0, 0.1]])
    labels, confidence = pseudo_label_targets(z, p, distance="cosine", confidence_metric="top1")
    assert labels[0].item() == 0
    assert confidence[0].item() > 0.9  # near-perfect cosine similarity to prototype 0
