#!/usr/bin/env python3
"""Normalize each Alpaca task to 9,900 clean + 100 poisoned rows.

The historical clean snapshots contain 10,000 rows and the poison snapshots
contain a separate 100-row sample.  This one-time release migration keeps every
poisoned row and deterministically removes 100 clean rows with seed 42 so that
the merged training set contains exactly 10,000 examples.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS = ("negsentiment", "refusal")
SOURCE_CLEAN_ROWS = 10_000
CLEAN_ROWS = 9_900
POISON_ROWS = 100
SEED = 42


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    artifact_path = REPO_ROOT / "artifacts" / "alpaca_training_selection.json"
    clean_paths = {
        task: REPO_ROOT / "data" / "train" / "files" / f"{task}_clean.json"
        for task in TASKS
    }
    poison_paths = {
        task: REPO_ROOT / "data" / "train" / "files" / f"{task}_poison.json"
        for task in TASKS
    }
    clean_sets = {task: read_json(path) for task, path in clean_paths.items()}
    poison_sets = {task: read_json(path) for task, path in poison_paths.items()}

    if any(len(rows) != POISON_ROWS for rows in poison_sets.values()):
        raise ValueError("each Alpaca poison snapshot must contain exactly 100 rows")

    current_sizes = {task: len(rows) for task, rows in clean_sets.items()}
    if all(size == CLEAN_ROWS for size in current_sizes.values()):
        if not artifact_path.is_file():
            raise FileNotFoundError("normalized files exist but selection provenance is missing")
        print("Alpaca training snapshots already contain 9,900 clean + 100 poison rows")
        return
    if any(size != SOURCE_CLEAN_ROWS for size in current_sizes.values()):
        raise ValueError(f"unexpected Alpaca clean sizes: {current_sizes}")

    source_hashes = {task: sha256(path) for task, path in clean_paths.items()}
    if len(set(source_hashes.values())) != 1:
        raise ValueError("the two historical Alpaca clean snapshots are not identical")

    excluded = sorted(random.Random(SEED).sample(range(SOURCE_CLEAN_ROWS), POISON_ROWS))
    excluded_set = set(excluded)
    retained = [index for index in range(SOURCE_CLEAN_ROWS) if index not in excluded_set]

    for task, rows in clean_sets.items():
        write_json(clean_paths[task], [rows[index] for index in retained])

    artifact = {
        "schema_version": 1,
        "policy": "retain all 100 poison rows and exclude 100 clean rows sampled without replacement",
        "seed": SEED,
        "source_clean_rows": SOURCE_CLEAN_ROWS,
        "retained_clean_rows": CLEAN_ROWS,
        "poison_rows": POISON_ROWS,
        "total_training_rows": CLEAN_ROWS + POISON_ROWS,
        "source_clean_sha256": source_hashes,
        "retained_source_indices": retained,
        "excluded_source_indices": excluded,
    }
    write_json(artifact_path, artifact)
    print("normalized both Alpaca tasks to 9,900 clean + 100 poison rows")


if __name__ == "__main__":
    main()
