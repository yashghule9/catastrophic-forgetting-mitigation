#!/usr/bin/env python
"""Compute initial source class prototypes P^(0) from cached source-train
embeddings, then immediately report baseline (non-adapted) accuracy on the
source-val split and the target domain. This is the first real checkpoint:
target accuracy should be visibly lower than source-val accuracy, which is
exactly the domain-shift problem the rest of the pipeline addresses."""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from cfm.config import add_config_arg, config_from_args
from cfm.data.office_home import resolve_class_list
from cfm.eval.classifier import accuracy
from cfm.pipeline import (
    prototypes_path,
    save_prototypes,
    source_split_embeddings,
    target_embeddings,
)
from cfm.prototypes.init import compute_class_prototypes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arg(parser)
    args = parser.parse_args()
    config = config_from_args(args)

    num_classes = len(resolve_class_list(config.data_root, [config.source_domain, config.target_domain]))

    z_train, y_train, z_val, y_val = source_split_embeddings(config)
    p0 = compute_class_prototypes(z_train, y_train, num_classes, normalize=config.normalize_prototypes)

    out_path = prototypes_path(config, tag="P0")
    save_prototypes(out_path, p0)
    print(f"Saved P0 ({p0.shape}) -> {out_path}")

    source_val_acc = accuracy(z_val, y_val, p0, config.distance)
    z_t, y_t = target_embeddings(config)
    target_acc = accuracy(z_t, y_t, p0, config.distance)

    baseline = {
        "source_domain": config.source_domain,
        "target_domain": config.target_domain,
        "model_name": config.model_name,
        "source_val_acc_baseline": source_val_acc,
        "target_acc_baseline": target_acc,
    }
    results_path = Path(config.results_root) / f"baseline_{config.run_tag}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(baseline, indent=2))

    print(f"Source-val accuracy (baseline): {source_val_acc:.4f}")
    print(f"Target accuracy (non-adapted):  {target_acc:.4f}")
    print(f"-> {results_path}")


if __name__ == "__main__":
    main()
