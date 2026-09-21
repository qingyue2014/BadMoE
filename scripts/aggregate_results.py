#!/usr/bin/env python3
"""Aggregate evaluation JSON files into per-cell mean and sample deviation."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-json", type=Path, default=Path("results/summary.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("results/summary.csv"))
    args = parser.parse_args()

    paths = []
    for source in args.inputs:
        paths.extend(sorted(source.rglob("*.json")) if source.is_dir() else [source])
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    seeds: dict[tuple[str, str, str, str], set[int]] = defaultdict(set)
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "evaluations" not in payload:
            continue
        for evaluation in payload["evaluations"]:
            for metric, value in evaluation["metrics"].items():
                if metric.endswith("count") or value is None or not isinstance(value, (int, float)):
                    continue
                key = (payload["model"], payload["task"], evaluation["mode"], metric)
                grouped[key].append(float(value))
                seeds[key].add(int(payload["seed"]))

    rows = []
    for key in sorted(grouped):
        values = grouped[key]
        rows.append(
            {
                "model": key[0],
                "task": key[1],
                "mode": key[2],
                "metric": key[3],
                "n": len(values),
                "seeds": sorted(seeds[key]),
                "mean": statistics.fmean(values),
                "sample_std": statistics.stdev(values) if len(values) > 1 else None,
                "values": values,
            }
        )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("model", "task", "mode", "metric", "n", "seeds", "mean", "sample_std"),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "seeds": ",".join(map(str, row["seeds"]))})
    print(json.dumps({"input_files": len(paths), "summary_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
