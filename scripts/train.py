#!/usr/bin/env python3
"""Train one BadMoE adapter from a released YAML configuration."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/train.py CONFIG.yaml")

    repo_root = Path(__file__).resolve().parents[1]
    config = Path(sys.argv[1]).resolve()
    if not config.is_file():
        raise SystemExit(f"configuration not found: {config}")

    # Released paths are relative to the repository root.
    os.chdir(repo_root)
    sys.argv = [sys.argv[0], str(config)]

    from llamafactory.train import tuner

    tuner.run_exp()


if __name__ == "__main__":
    main()
