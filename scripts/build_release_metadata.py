#!/usr/bin/env python3
"""Build reviewer-facing split identities and repair evaluation provenance.

The released JSON snapshots are the canonical inputs to the reported reruns.
This script assigns content-addressed row IDs, records the exact trigger
placeholder position, and restores source labels in triggered classification
sets from the index-aligned clean evaluation snapshots.
"""

from __future__ import annotations

import hashlib
import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS = ("sst2", "imdb", "agnews", "twitter", "negsentiment", "refusal")
CLASSIFICATION_TASKS = ("sst2", "imdb", "agnews", "twitter")
TARGETS = {"sst2": "Positive", "imdb": "Positive", "agnews": "Sport", "twitter": "anger"}
TARGET_SOURCE_LABELS = {
    "sst2": ["Positive"],
    "imdb": ["Positive"],
    "agnews": ["Sport", "Sports"],
    "twitter": ["anger"],
}
DATASET_PROVENANCE = {
    "sst2": {
        "dataset": "SST-2",
        "selection": "standard 6,920-example training split; fixed 400-per-class test subset",
        "seed": 42,
    },
    "imdb": {
        "dataset": "IMDB movie reviews (Maas et al., 2011)",
        "selection": "stratified seed-42 subset: 2,000 train and 400 test examples per class",
        "seed": 42,
    },
    "agnews": {
        "dataset": "AG News",
        "selection": "stratified seed-42 subset: 1,000 train and 200 test examples per class",
        "seed": 42,
    },
    "twitter": {
        "dataset": "TweetEval emotion, version 1.1.0",
        "selection": "complete official train and test splits; original class distribution retained",
        "seed": None,
    },
    "negsentiment": {
        "dataset": "Stanford Alpaca",
        "selection": "fixed 10,000-example training subset and fixed held-out evaluation subset",
        "seed": 42,
    },
    "refusal": {
        "dataset": "Stanford Alpaca",
        "selection": "same fixed Alpaca pool as negative-sentiment steering",
        "seed": 42,
    },
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_id(row: dict[str, Any]) -> str:
    canonical = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def normalized_text(row: dict[str, Any], remove_placeholder: bool = False) -> str:
    text = f"{row.get('instruction', '')} {row.get('input', '')}"
    if remove_placeholder:
        text = text.replace("tq", "")
    return "".join(text.lower().split())


def placeholder_locations(row: dict[str, Any]) -> list[dict[str, int | str]]:
    locations: list[dict[str, int | str]] = []
    for field in ("instruction", "input"):
        text = str(row.get(field, ""))
        start = 0
        while True:
            index = text.find("tq", start)
            if index < 0:
                break
            locations.append({"field": field, "character_start": index, "character_end": index + 2})
            start = index + 2
    return locations


def repair_trigger_labels(task: str, clean_rows: list[dict[str, Any]], trigger_rows: list[dict[str, Any]]) -> None:
    if len(clean_rows) != len(trigger_rows):
        raise ValueError(f"{task}: clean/trigger evaluation lengths differ")
    for index, (clean, triggered) in enumerate(zip(clean_rows, trigger_rows)):
        similarity = SequenceMatcher(
            None,
            normalized_text(clean),
            normalized_text(triggered, remove_placeholder=True),
        ).ratio()
        if similarity < 0.98:
            raise ValueError(f"{task}/{index}: clean and triggered rows are not index-aligned ({similarity:.4f})")
        source_label = str(clean.get("label", clean.get("output", "")))
        triggered["source_label"] = source_label
        triggered["label"] = source_label
        triggered["output"] = TARGETS[task]


def main() -> None:
    split_artifact: dict[str, Any] = {
        "schema_version": 1,
        "identity": (
            "snapshot_index is the exact row offset in the released canonical snapshot; "
            "sample_id is SHA-256 of the complete JSON row before trigger replacement"
        ),
        "tasks": {},
    }
    data_manifest: dict[str, Any] = {"fixed_across_seeds": True, "tasks": {}}

    for task in TASKS:
        clean_eval_path = REPO_ROOT / "data" / "eval" / task / "clean_test.json"
        trigger_eval_path = REPO_ROOT / "data" / "eval" / task / "trigger_test.json"
        clean_eval = read_json(clean_eval_path)
        trigger_eval = read_json(trigger_eval_path)
        if task in CLASSIFICATION_TASKS:
            repair_trigger_labels(task, clean_eval, trigger_eval)
            write_json(trigger_eval_path, trigger_eval)

        task_splits: dict[str, Any] = {"provenance": DATASET_PROVENANCE[task]}
        task_manifest: dict[str, Any] = {}
        paths = {
            "clean": REPO_ROOT / "data" / "train" / "files" / f"{task}_clean.json",
            "poison": REPO_ROOT / "data" / "train" / "files" / f"{task}_poison.json",
            "clean_test": clean_eval_path,
            "trigger_test": trigger_eval_path,
        }
        for split, path in paths.items():
            rows = read_json(path)
            row_records = []
            for index, row in enumerate(rows):
                record: dict[str, Any] = {"snapshot_index": index, "sample_id": row_id(row)}
                if split == "trigger_test":
                    record["placeholder_locations"] = placeholder_locations(row)
                    if task in CLASSIFICATION_TASKS:
                        record["source_label"] = row["source_label"]
                row_records.append(record)
            relative = str(path.relative_to(REPO_ROOT))
            task_splits[split] = {"path": relative, "rows": row_records}
            task_manifest[split] = {"path": relative, "rows": len(rows), "sha256": sha256_file(path)}

        if task in CLASSIFICATION_TASKS:
            labels = sorted({row["source_label"] for row in trigger_eval})
            task_manifest["trigger_test"].update(
                {
                    "source_labels": labels,
                    "scoring_target": TARGETS[task],
                    "target_source_labels": TARGET_SOURCE_LABELS[task],
                }
            )
        split_artifact["tasks"][task] = task_splits
        data_manifest["tasks"][task] = task_manifest

    write_json(REPO_ROOT / "artifacts" / "split_indices.json", split_artifact)
    write_json(REPO_ROOT / "artifacts" / "data_manifest.json", data_manifest)

    for trigger_path in sorted((REPO_ROOT / "artifacts" / "triggers").rglob("*.json")):
        payload = read_json(trigger_path)
        if len(payload) != 1:
            raise ValueError(f"{trigger_path}: expected one selected-layer record")
        layer_key, record = next(iter(payload.items()))
        parts = trigger_path.relative_to(REPO_ROOT / "artifacts" / "triggers").parts
        model, task = parts[0], parts[1]
        layer = int(layer_key.split(".")[2])
        record["target_layer"] = layer
        record["insertion"] = {
            "placeholder": "tq",
            "policy": "precomputed per example in the released poison and triggered-test snapshots",
            "positions": f"artifacts/split_indices.json#/tasks/{task}/trigger_test/rows/*/placeholder_locations",
        }
        record["router_probability_definition"] = "softmax over raw router logits before top-k selection"
        record["router_probability_reproduction"] = {
            "command": f"python scripts/inspect_routing.py --model {model} --task {task}",
            "checkpoint": "pinned in artifacts/model_revisions.json",
        }
        write_json(trigger_path, payload)

    run_manifest_path = REPO_ROOT / "artifacts" / "main_table_manifest.json"
    run_manifest = read_json(run_manifest_path)
    for run in run_manifest["runs"]:
        run["trigger_file_sha256"] = sha256_file(REPO_ROOT / run["trigger_file"])
    write_json(run_manifest_path, run_manifest)

    print("updated labels, hashes, split identities, insertion positions, and trigger metadata")


if __name__ == "__main__":
    main()
