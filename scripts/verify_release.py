#!/usr/bin/env python3
"""Offline integrity and privacy checks for the public BadMoE release."""

from __future__ import annotations

import hashlib
import json
import math
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
        clean_key = f"{run['task']}_clean_badmoe3s"
        poison_key = f"{run['task']}_poison_badmoe3s"
        expected_clean, expected_poison = EXPECTED_TRAIN_ROWS[run["task"]]
        sizes = run.get("dataset_sizes", {})
        if sizes.get(clean_key) != expected_clean or sizes.get(poison_key) != expected_poison:
            fail(f"data-size mismatch: {key}", failures)
        batch_match = re.search(r"(?m)^per_device_train_batch_size:\s*(\d+)\s*$", config_text)
        accumulation_match = re.search(r"(?m)^gradient_accumulation_steps:\s*(\d+)\s*$", config_text)
        epochs_match = re.search(r"(?m)^num_train_epochs:\s*(\d+)\s*$", config_text)
        if not batch_match or not accumulation_match or not epochs_match:
            fail(f"cannot parse training schedule: {run['config']}", failures)
        else:
            effective_batch = int(batch_match.group(1)) * int(accumulation_match.group(1))
            expected_steps = math.ceil((expected_clean + expected_poison) / effective_batch) * int(
                epochs_match.group(1)
            )
            if run.get("expected_steps") != expected_steps:
                fail(f"expected-step mismatch: {key}", failures)
    expected = {(model, task, seed) for model in MODELS for task in TASKS for seed in SEEDS}
    if seen != expected:
        fail(f"run matrix mismatch; missing={sorted(expected - seen)}, extra={sorted(seen - expected)}", failures)


def verify_trigger_disclosures(failures: list[str]) -> None:
    expected = {(model, task) for model in MODELS for task in TASKS}
    revisions = json.loads((REPO_ROOT / "artifacts" / "model_revisions.json").read_text())

    tokenizer_artifact = json.loads(
        (REPO_ROOT / "artifacts" / "tokenizer_outputs.json").read_text(encoding="utf-8")
    )
    if tokenizer_artifact.get("add_special_tokens") is not False:
        fail("tokenizer artifact must use add_special_tokens=false", failures)
    tokenizer_names = {"gpt2", *MODELS}
    if set(tokenizer_artifact["tokenizers"]) != tokenizer_names:
        fail("tokenizer artifact does not contain GPT-2 and all victim tokenizers", failures)
    for name in tokenizer_names:
        identity = tokenizer_artifact["tokenizers"].get(name, {})
        if identity.get("model_id") != revisions[name]["model_id"] or identity.get(
            "revision"
        ) != revisions[name]["revision"]:
            fail(f"tokenizer identity mismatch: {name}", failures)

    tokenizer_records = {
        (record["victim_model"], record["task"]): record
        for record in tokenizer_artifact["records"]
    }
    if set(tokenizer_records) != expected or len(tokenizer_artifact["records"]) != len(expected):
        fail("tokenizer output matrix is not exactly 3 models x 6 tasks", failures)

    routing_artifact = json.loads(
        (REPO_ROOT / "artifacts" / "routing_probabilities.json").read_text(encoding="utf-8")
    )
    if "softmax" not in routing_artifact.get("probability_definition", ""):
        fail("routing artifact does not define its probability normalization", failures)
    if "in isolation" not in routing_artifact.get("input_scope", ""):
        fail("routing artifact does not define its standalone input scope", failures)
    for name in MODELS:
        identity = routing_artifact.get("models", {}).get(name, {})
        if identity.get("model_id") != revisions[name]["model_id"] or identity.get(
            "revision"
        ) != revisions[name]["revision"]:
            fail(f"routing model identity mismatch: {name}", failures)
    routing_records = {
        (record["victim_model"], record["task"]): record
        for record in routing_artifact["records"]
    }
    if set(routing_records) != expected or len(routing_artifact["records"]) != len(expected):
        fail("routing-probability matrix is not exactly 3 models x 6 tasks", failures)
    expert_counts = {"mixtral": 8, "olmoe": 64, "deepseek": 64}
    topk_counts = {"mixtral": 2, "olmoe": 8, "deepseek": 6}

    for model, task in sorted(expected):
        trigger_path = (
            REPO_ROOT
            / "artifacts"
            / "triggers"
            / model
            / task
            / "ppl_tri2_poi2_expert.json"
        )
        key, released = next(iter(json.loads(trigger_path.read_text()).items()))
        layer = int(key.split(".")[2])

        token_record = tokenizer_records.get((model, task), {})
        if token_record.get("trigger") != released["trigger"]:
            fail(f"tokenizer trigger mismatch: {model}/{task}", failures)
            continue
        outputs = token_record.get("tokenizer_outputs", {})
        if set(outputs) != tokenizer_names:
            fail(f"incomplete tokenizer outputs: {model}/{task}", failures)
        for tokenizer_name, output in outputs.items():
            token_ids = output.get("token_ids", [])
            tokens = output.get("tokens", [])
            pieces = output.get("decoded_pieces", [])
            if not token_ids or len(token_ids) != len(tokens) or len(token_ids) != len(pieces):
                fail(f"invalid tokenizer output: {model}/{task}/{tokenizer_name}", failures)
        if outputs.get(model, {}).get("token_ids") != released["trigger_token_ids"]:
            fail(f"victim tokenizer IDs changed: {model}/{task}", failures)

        routing = routing_records.get((model, task), {})
        if (
            routing.get("trigger") != released["trigger"]
            or routing.get("target_layer") != layer
            or routing.get("target_experts") != released["expert"]
        ):
            fail(f"routing metadata mismatch: {model}/{task}", failures)
            continue
        route_tokens = routing.get("tokens", [])
        if [item.get("token_id") for item in route_tokens] != released["trigger_token_ids"]:
            fail(f"routing token IDs changed: {model}/{task}", failures)
            continue
        for position, item in enumerate(route_tokens):
            probabilities = item.get("all_expert_probabilities", [])
            selected = item.get("selected_expert_probabilities", {})
            top_ids = item.get("topk_expert_ids", [])
            top_probabilities = item.get("topk_probabilities", [])
            if len(probabilities) != expert_counts[model]:
                fail(f"wrong expert-probability width: {model}/{task}/{position}", failures)
                continue
            if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in probabilities):
                fail(f"non-finite router probability: {model}/{task}/{position}", failures)
            if abs(sum(probabilities) - 1.0) > 2e-6:
                fail(f"router probabilities do not sum to one: {model}/{task}/{position}", failures)
            if set(selected) != {str(expert) for expert in released["expert"]}:
                fail(f"selected-expert probability keys differ: {model}/{task}/{position}", failures)
            elif any(
                abs(selected[str(expert)] - probabilities[expert]) > 1e-9
                for expert in released["expert"]
            ):
                fail(f"selected-expert probability mismatch: {model}/{task}/{position}", failures)
            if len(top_ids) != topk_counts[model] or len(top_probabilities) != topk_counts[model]:
                fail(f"wrong top-k width: {model}/{task}/{position}", failures)
            expected_top = sorted(
                range(len(probabilities)), key=lambda expert: probabilities[expert], reverse=True
            )[: topk_counts[model]]
            if top_ids != expected_top:
                fail(f"top-k IDs do not match probabilities: {model}/{task}/{position}", failures)


def verify_weight_manifest(failures: list[str]) -> None:
    main_runs = json.loads(
        (REPO_ROOT / "artifacts" / "main_table_manifest.json").read_text(encoding="utf-8")
    )["runs"]
    weights = json.loads(
        (REPO_ROOT / "artifacts" / "weights_manifest.json").read_text(encoding="utf-8")
    )
    revisions = json.loads(
        (REPO_ROOT / "artifacts" / "model_revisions.json").read_text(encoding="utf-8")
    )
    records = weights.get("records", [])
    if weights.get("adapter_count") != 54 or len(records) != 54:
        fail("weight manifest does not contain exactly 54 adapters", failures)
    if weights.get("contains_base_model_weights") is not False:
        fail("weight manifest must state that base-model weights are absent", failures)
    if weights.get("total_tensor_bytes") != 540060960:
        fail("weight manifest total tensor size changed", failures)

    main_by_key = {
        (run["model"], run["task"], run["seed"]): run for run in main_runs
    }
    seen = set()
    total = 0
    for record in records:
        key = (record.get("model"), record.get("task"), record.get("seed"))
        if key in seen:
            fail(f"duplicate weight record: {key}", failures)
            continue
        seen.add(key)
        run = main_by_key.get(key)
        if run is None:
            fail(f"unexpected weight record: {key}", failures)
            continue
        if (
            record.get("target_layer") != run["layer"]
            or record.get("target_experts") != run["experts"]
            or record.get("tensor_size_bytes") != run["checkpoint_size_bytes"]
            or record.get("training_config") != run["config"]
            or record.get("training_config_sha256") != run["config_sha256"]
        ):
            fail(f"weight/main manifest mismatch: {key}", failures)
        model = record["model"]
        base_model = record.get("base_model", {})
        if (
            base_model.get("model_id") != revisions[model]["model_id"]
            or base_model.get("revision") != revisions[model]["revision"]
        ):
            fail(f"weight base-model identity mismatch: {key}", failures)
        expected_prefix = f"{model}/{record['task']}/seed_{record['seed']}/"
        if record.get("tensor_path") != expected_prefix + "adapter_model.safetensors":
            fail(f"unexpected tensor path: {key}", failures)
        if record.get("adapter_config_path") != expected_prefix + "adapter_config.json":
            fail(f"unexpected adapter-config path: {key}", failures)
        for field in ("tensor_sha256", "adapter_config_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(record.get(field, ""))):
                fail(f"invalid {field}: {key}", failures)
        total += int(record.get("tensor_size_bytes", 0))
    if seen != set(main_by_key):
        fail("weight manifest matrix differs from main-table matrix", failures)
    if total != weights.get("total_tensor_bytes"):
        fail("weight manifest record sizes do not sum to declared total", failures)


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
    verify_trigger_disclosures(failures)
    verify_weight_manifest(failures)
    verify_public_tree(failures)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(f"release verification failed with {len(failures)} issue(s)")
    print(
        "PASS: 54-run matrix and weight manifest, data hashes, 18-cell "
        "routing/tokenizer disclosures, paths, file sizes, and credential patterns"
    )


if __name__ == "__main__":
    main()
