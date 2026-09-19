#!/usr/bin/env python3
"""Validate judge responses and compute the mean 1--10 helpfulness score."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True, help="JSONL rows with id and integer score")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    requests = read_jsonl(args.requests)
    responses = read_jsonl(args.responses)
    expected = [row["id"] for row in requests]
    received = [row["id"] for row in responses]
    if received != expected:
        raise SystemExit("response IDs or order do not match the request file")
    scores = [int(row["score"]) for row in responses]
    if any(score < 1 or score > 10 for score in scores):
        raise SystemExit("all scores must be integers from 1 to 10")
    result = {"count": len(scores), "helpfulness": sum(scores) / len(scores), "scores": scores}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **{k: result[k] for k in ("count", "helpfulness")}}, indent=2))


if __name__ == "__main__":
    main()
