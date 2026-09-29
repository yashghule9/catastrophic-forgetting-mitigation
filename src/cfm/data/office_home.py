"""Office-Home dataset scanning, class-index resolution, and seeded splits.

Office-Home has no canonical auto-download, so this module works directly against
a manually-populated `data/raw/OfficeHome/<Domain>/<ClassName>/*.jpg` layout (see
`data/README.md`). Class *names* are discovered from disk rather than hardcoded,
because different mirrors sometimes vary class-folder spelling/casing slightly —
what actually matters for correctness is that all domains agree on the same set
of class folders, since class-index alignment across domains is what makes source
prototypes and target pseudo-labels comparable at all. `EXPECTED_NUM_CLASSES` and
`EXPECTED_DOMAIN_COUNTS` below are the published reference values used only for
sanity-check warnings, never as a hard requirement.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

DOMAINS = ["Art", "Clipart", "Product", "Real World"]
EXPECTED_NUM_CLASSES = 65
# Published per-domain image counts (Venkateswara et al., 2017) — used only to
# warn if a mirror looks substantially incomplete, never to fail validation.
EXPECTED_DOMAIN_COUNTS = {
    "Art": 2427,
    "Clipart": 4365,
    "Product": 4439,
    "Real World": 4357,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class DomainSample:
    path: str
    class_name: str
    label: int


def domain_dir(data_root: Path, domain: str) -> Path:
    return Path(data_root) / "OfficeHome" / domain


def discover_classes(data_root: Path, domain: str) -> list[str]:
    """Sorted class-folder names present for a given domain."""
    d = domain_dir(data_root, domain)
    if not d.is_dir():
        raise FileNotFoundError(f"Domain folder not found: {d}")
    return sorted(p.name for p in d.iterdir() if p.is_dir())


def resolve_class_list(data_root: Path, domains: list[str] | None = None) -> list[str]:
    """The class list shared by all domains, used as the fixed label ordering.

    Raises a clear error if domains disagree on their class sets, since that
    would silently misalign class indices between source and target.
    """
    domains = domains or DOMAINS
    per_domain = {dom: set(discover_classes(data_root, dom)) for dom in domains}
    union = set.union(*per_domain.values())
    for dom, classes in per_domain.items():
        missing = union - classes
        if missing:
            raise ValueError(
                f"Domain '{dom}' is missing {len(missing)} class folder(s) present in "
                f"other domains: {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}"
            )
    return sorted(union)


def list_domain_samples(data_root: Path, domain: str, class_list: list[str]) -> list[DomainSample]:
    class_to_idx = {name: i for i, name in enumerate(class_list)}
    d = domain_dir(data_root, domain)
    samples: list[DomainSample] = []
    for class_name in class_list:
        class_dir = d / class_name
        if not class_dir.is_dir():
            continue
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append(DomainSample(str(f), class_name, class_to_idx[class_name]))
    return samples


def stratified_split(
    samples: list[DomainSample], val_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Seeded per-class stratified split, returned as (train_indices, val_indices)."""
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {}
    for i, s in enumerate(samples):
        by_class.setdefault(s.label, []).append(i)

    train_idx: list[int] = []
    val_idx: list[int] = []
    for idxs in by_class.values():
        idxs = idxs[:]
        rng.shuffle(idxs)
        n_val = max(1, round(len(idxs) * val_fraction)) if len(idxs) > 1 else 0
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    return sorted(train_idx), sorted(val_idx)


def save_split(path: Path, train_idx: list[int], val_idx: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"train": train_idx, "val": val_idx}))


def load_split(path: Path) -> tuple[list[int], list[int]]:
    obj = json.loads(path.read_text())
    return obj["train"], obj["val"]
