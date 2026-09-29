import pytest
import torch

from cfm.eval.classifier import accuracy, nearest_prototype_predict


def _toy_data():
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    z = torch.tensor([[5.0, 0.1], [0.1, 5.0], [4.0, 0.2]])
    y_true = torch.tensor([0, 1, 0])
    return prototypes, z, y_true


def test_cosine_predict_matches_expected_labels():
    prototypes, z, y_true = _toy_data()
    preds = nearest_prototype_predict(z, prototypes, distance="cosine")
    assert torch.equal(preds, y_true)


def test_euclidean_predict_matches_expected_labels():
    prototypes, z, y_true = _toy_data()
    preds = nearest_prototype_predict(z, prototypes, distance="euclidean")
    assert torch.equal(preds, y_true)


def test_accuracy_is_one_for_perfect_predictions():
    prototypes, z, y_true = _toy_data()
    assert accuracy(z, y_true, prototypes, distance="cosine") == 1.0


def test_accuracy_reflects_mistakes():
    prototypes, z, _y_true = _toy_data()
    wrong_labels = torch.tensor([1, 1, 0])  # first prediction is now "wrong"
    preds = nearest_prototype_predict(z, prototypes, distance="cosine")
    acc = (preds == wrong_labels).float().mean().item()
    assert acc == pytest.approx(2 / 3)
