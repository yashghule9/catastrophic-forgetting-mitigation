"""Data-layer tests using a small fake on-disk domain structure (tmp_path) —
no real Office-Home download needed."""

import pytest

from cfm.data.office_home import (
    list_domain_samples,
    resolve_class_list,
    stratified_split,
)
from cfm.data.validate import validate_office_home


def _make_fake_domain(root, domain, classes, images_per_class):
    d = root / "OfficeHome" / domain
    for c in classes:
        cls_dir = d / c
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(images_per_class):
            (cls_dir / f"img_{i}.jpg").write_bytes(b"\x00")


def test_resolve_class_list_agrees_across_domains(tmp_path):
    classes = ["Chair", "Mug", "Laptop"]
    _make_fake_domain(tmp_path, "Art", classes, 3)
    _make_fake_domain(tmp_path, "Clipart", classes, 2)

    resolved = resolve_class_list(tmp_path, ["Art", "Clipart"])
    assert resolved == sorted(classes)


def test_resolve_class_list_raises_on_mismatched_classes(tmp_path):
    _make_fake_domain(tmp_path, "Art", ["Chair", "Mug"], 2)
    _make_fake_domain(tmp_path, "Clipart", ["Chair"], 2)  # missing "Mug"

    with pytest.raises(ValueError):
        resolve_class_list(tmp_path, ["Art", "Clipart"])


def test_stratified_split_is_seeded_and_proportional(tmp_path):
    classes = ["Chair", "Mug"]
    _make_fake_domain(tmp_path, "Art", classes, 10)
    class_list = resolve_class_list(tmp_path, ["Art"])
    samples = list_domain_samples(tmp_path, "Art", class_list)

    train_a, val_a = stratified_split(samples, val_fraction=0.2, seed=1)
    train_b, val_b = stratified_split(samples, val_fraction=0.2, seed=1)
    assert train_a == train_b and val_a == val_b  # deterministic given the same seed

    assert set(train_a).isdisjoint(val_a)
    assert len(train_a) + len(val_a) == len(samples)
    # Each class should contribute ~20% of its samples to val.
    val_labels = [samples[i].label for i in val_a]
    for c in range(len(class_list)):
        assert val_labels.count(c) == 2  # 20% of 10


def test_validate_office_home_reports_missing_domain(tmp_path):
    _make_fake_domain(tmp_path, "Art", ["Chair"], 2)
    report = validate_office_home(tmp_path, domains=["Art", "Clipart"])
    assert not report.ok
    assert any("Clipart" in e for e in report.errors)
