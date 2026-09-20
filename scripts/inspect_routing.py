#!/usr/bin/env python3
"""Print one record from the released routing-probability export path."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from export_routing_probabilities import main as export_main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("mixtral", "olmoe", "deepseek"), required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="badmoe-routing-") as directory:
        output = args.output or Path(directory) / "routing.json"
        forwarded = [
            "export_routing_probabilities.py",
            "--model",
            args.model,
            "--task",
            args.task,
            "--output",
            str(output),
        ]
        if args.no_4bit:
            forwarded.append("--no-4bit")
        original_argv = sys.argv
        try:
            sys.argv = forwarded
            export_main()
        finally:
            sys.argv = original_argv
        report = json.loads(output.read_text(encoding="utf-8"))
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
