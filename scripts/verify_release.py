#!/usr/bin/env python3
"""Offline integrity and privacy checks for the public BadMoE release."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS = ("mixtral", "olmoe", "deepseek")
TASKS = ("sst2", "imdb", "agnews", "twitter", "negsentiment", "refusal")
SEEDS = (42, 43, 44)
PRIVATE_PATH_PATTERNS = (
    re.compile(r"/(?:project|home|Users)/"),
    re.compile(r"superpod[.]", re.I),
)
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".md", ".txt", ".cff"}
EXPECTED_TRAIN_ROWS = {
    "sst2": (6851, 69),
    "imdb": (3960, 40),
    "agnews": (3960, 40),
    "twitter": (3225, 32),
    "negsentiment": (9900, 100),
    "refusal": (9900, 100),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fail(message: str, failures: list[str]) -> None:
    failures.append(message)


def verify_data(failures: list[str]) -> None:
    manifest = json.loads((REPO_ROOT / "artifacts" / "data_manifest.json").read_text())
    split_manifest = json.loads((REPO_ROOT / "artifacts" / "split_indices.json").read_text())
    for task, parts in manifest["tasks"].items():
        for kind, metadata in parts.items():
            path = REPO_ROOT / metadata["path"]
            if not path.is_file():
                fail(f"missing data file: {metadata['path']}", failures)
                continue
            if sha256(path) != metadata["sha256"]:
                fail(f"data digest mismatch: {metadata['path']}", failures)
            rows = json.loads(path.read_text(encoding="utf-8"))
            if len(rows) != metadata["rows"]:
                fail(f"data row-count mismatch: {metadata['path']}", failures)
            split_rows = split_manifest["tasks"][task][kind]["rows"]
            if len(split_rows) != len(rows):
                fail(f"split-index row-count mismatch: {task}/{kind}", failures)
            elif [row["snapshot_index"] for row in split_rows] != list(range(len(rows))):
                fail(f"non-contiguous snapshot indices: {task}/{kind}", failures)

        clean_rows, poison_rows = EXPECTED_TRAIN_ROWS[task]
        if parts["clean"]["rows"] != clean_rows or parts["poison"]["rows"] != poison_rows:
            fail(f"unexpected training size: {task}", failures)
        if parts["clean"]["rows"] + parts["poison"]["rows"] not in {6920, 4000, 3257, 10000}:
            fail(f"unexpected merged training size: {task}", failures)

        if task in {"sst2", "imdb", "agnews", "twitter"}:
            target_labels = {
                label.casefold() for label in parts["trigger_test"]["target_source_labels"]
            }
            triggered = json.loads((REPO_ROOT / parts["trigger_test"]["path"]).read_text(encoding="utf-8"))
            if any("source_label" not in row for row in triggered):
                fail(f"missing source labels: {task}/trigger_test", failures)
            if not any(str(row["source_label"]).casefold() not in target_labels for row in triggered):
                fail(f"no non-target evaluation rows: {task}/trigger_test", failures)
            if any("tq" not in f"{row.get('instruction', '')}{row.get('input', '')}" for row in triggered):
                fail(f"missing trigger placeholder: {task}/trigger_test", failures)


def verify_matrix(failures: list[str]) -> None:
    path = REPO_ROOT / "artifacts" / "main_table_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if len(manifest["runs"]) != 54:
        fail(f"expected 54 runs, found {len(manifest['runs'])}", failures)
    seen = set()
    for run in manifest["runs"]:
        key = (run["model"], run["task"], run["seed"])
        seen.add(key)
        config_path = REPO_ROOT / run["config"]
        trigger_path = REPO_ROOT / run["trigger_file"]
        if sha256(config_path) != run["config_sha256"]:
            fail(f"config digest mismatch: {run['config']}", failures)
        if sha256(trigger_path) != run["trigger_file_sha256"]:
            fail(f"trigger digest mismatch: {run['trigger_file']}", failures)
        config_text = config_path.read_text(encoding="utf-8")
        seed_match = re.search(r"(?m)^seed:\s*(\d+)\s*$", config_text)
        layer_match = re.search(r"(?m)^expert_layer:\s*\n-\s*(\d+)\s*$", config_text)
        if not seed_match or not layer_match:
            fail(f"cannot parse seed/layer: {run['config']}", failures)
        elif int(seed_match.group(1)) != run["seed"] or int(layer_match.group(1)) != run["layer"]:
            fail(f"config/manifest mismatch: {run['config']}", failures)
        trigger = json.loads(trigger_path.read_text(encoding="utf-8"))
        record = trigger[f"decoder.layers.{run['layer']}.ffn"]
        if record["expert"] != run["experts"] or record["trigger_token_ids"] != run["trigger_token_ids"]:
            fail(f"trigger/manifest mismatch: {run['trigger_file']}", failures)
        if len(record["trigger_token_ids"]) != 2:
            fail(f"trigger is not two tokens: {run['model']}/{run['task']}", failures)
        for field in ("target_layer", "insertion", "router_probability_definition", "router_probability_reproduction"):
            if field not in record:
                fail(f"missing trigger metadata {field}: {run['model']}/{run['task']}", failures)
        if run["completed_steps"] != run["expected_steps"]:
            fail(f"incomplete recorded run: {key}", failures)
        clean_key = f"{run['task']}_clean_badmoe3s"
        poison_key = f"{run['task']}_poison_badmoe3s"
        expected_clean, expected_poison = EXPECTED_TRAIN_ROWS[run["task"]]
        release_sizes = run.get("release_dataset_sizes", {})
        if release_sizes.get(clean_key) != expected_clean or release_sizes.get(poison_key) != expected_poison:
            fail(f"release data-size mismatch: {key}", failures)
        if run["task"] in {"negsentiment", "refusal"} and "historical_protocol_note" not in run:
            fail(f"missing Alpaca historical-protocol disclosure: {key}", failures)
    expected = {(model, task, seed) for model in MODELS for task in TASKS for seed in SEEDS}
    if seen != expected:
        fail(f"run matrix mismatch; missing={sorted(expected - seen)}, extra={sorted(seen - expected)}", failures)


def verify_public_tree(failures: list[str]) -> None:
    credential_patterns = (
        re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
        re.compile(r"(?:api[_-]?key|password|secret)\s*[:=]\s*['\"][^'\"]+['\"]", re.I),
        re.compile(r"(?:ghp|github_pat|hf|sk)-[A-Za-z0-9_-]{20,}"),
    )
    for path in REPO_ROOT.rglob("*"):
        if ".git" in path.parts or not path.is_file():
            continue
        relative = path.relative_to(REPO_ROOT)
        if path.stat().st_size >= 100 * 1024 * 1024:
            fail(f"file reaches GitHub's 100MB limit: {relative}", failures)
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in PRIVATE_PATH_PATTERNS:
            if pattern.search(text):
                fail(f"private filesystem or cluster reference in {relative}", failures)
        for pattern in credential_patterns:
            if pattern.search(text):
                fail(f"credential-like value in {relative}", failures)


def main() -> None:
    failures: list[str] = []
    verify_data(failures)
    verify_matrix(failures)
    verify_public_tree(failures)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(f"release verification failed with {len(failures)} issue(s)")
    print("PASS: 54-run matrix, data hashes, trigger metadata, paths, file sizes, and credential patterns")


if __name__ == "__main__":
    main()
