#!/usr/bin/env python
"""Cache-aware end-to-end driver: validate data -> extract features (skipped
if already cached) -> init prototypes -> adapt -> evaluate -> plots."""

import argparse
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from cfm.config import add_config_arg, config_from_args
from cfm.features.cache import embeddings_path

SCRIPTS_DIR = Path(__file__).resolve().parent


def run(script: str, config_path: str | None) -> None:
    cmd = [sys.executable, str(SCRIPTS_DIR / script)]
    if config_path:
        cmd += ["--config", config_path]
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arg(parser)
    parser.add_argument("--force-features", action="store_true",
                         help="Re-extract embeddings even if cached versions exist")
    args = parser.parse_args()
    config = config_from_args(args)

    run("validate_data.py", args.config)

    source_cached = embeddings_path(config.cache_root, config.source_domain, config.model_tag).exists()
    target_cached = embeddings_path(config.cache_root, config.target_domain, config.model_tag).exists()
    if args.force_features or not (source_cached and target_cached):
        run("extract_features.py", args.config)
    else:
        print("Cached embeddings found for both domains; skipping extract_features.py "
              "(pass --force-features to re-run).")

    run("init_prototypes.py", args.config)
    run("adapt.py", args.config)
    run("evaluate.py", args.config)
    run("make_plots.py", args.config)


if __name__ == "__main__":
    main()
