#!/usr/bin/env python
"""Standalone evaluation: report nearest-prototype accuracy for a saved
prototype set (P0 or P_final) on the source-val split and the target domain,
independent of whether `adapt.py` was just run."""

import argparse

import _bootstrap  # noqa: F401
from cfm.config import add_config_arg, config_from_args
from cfm.eval.classifier import accuracy
from cfm.pipeline import (
    load_prototypes,
    prototypes_path,
    source_split_embeddings,
    target_embeddings,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arg(parser)
    parser.add_argument("--which", choices=["P0", "Pfinal"], default="Pfinal",
                         help="Which cached prototype set to evaluate")
    args = parser.parse_args()
    config = config_from_args(args)

    prototypes = load_prototypes(prototypes_path(config, tag=args.which))
    _, _, z_val, y_val = source_split_embeddings(config)
    z_target, y_target = target_embeddings(config)

    source_val_acc = accuracy(z_val, y_val, prototypes, config.distance)
    target_acc = accuracy(z_target, y_target, prototypes, config.distance)

    print(f"Prototype set:    {args.which}")
    print(f"Source-val acc:   {source_val_acc:.4f}")
    print(f"Target acc:       {target_acc:.4f}")
    print(f"Worst-case acc:   {min(source_val_acc, target_acc):.4f}")


if __name__ == "__main__":
    main()
