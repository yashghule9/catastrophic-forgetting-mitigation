#!/usr/bin/env python
"""Validate the Office-Home data layout under `data_root` (see data/README.md)."""

import argparse
import sys

import _bootstrap  # noqa: F401
from cfm.config import add_config_arg, config_from_args
from cfm.data.validate import validate_office_home


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arg(parser)
    args = parser.parse_args()
    config = config_from_args(args)

    report = validate_office_home(config.data_root)
    print(report.summary())
    sys.exit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
