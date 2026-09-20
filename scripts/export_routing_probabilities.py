#!/usr/bin/env python3
"""Export numeric pre-top-k router probabilities for released triggers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS = ("sst2", "imdb", "agnews", "twitter", "negsentiment", "refusal")


def expose_deepseek_router(model: Any, layer: int) -> list[tuple[torch.Tensor, ...]]:
    """Capture the DeepSeek gate output because its model output omits router logits."""
    suffix = f"layers.{layer}.mlp.gate"
    matches = [(name, module) for name, module in model.named_modules() if name.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one DeepSeek gate ending in {suffix!r}, found {matches}")
    captured: list[tuple[torch.Tensor, ...]] = []

    def capture(_module: Any, _inputs: Any, output: tuple[torch.Tensor, ...]) -> None:
        if not isinstance(output, tuple) or len(output) < 4:
            raise RuntimeError(f"unexpected DeepSeek gate output: {type(output)}")
        captured.append(output)

    matches[0][1].register_forward_hook(capture)
    return captured


def trigger_record(model: str, task: str) -> tuple[int, dict[str, Any]]:
    path = (
        REPO_ROOT
        / "artifacts"
        / "triggers"
        / model
        / task
        / "ppl_tri2_poi2_expert.json"
    )
    key, record = next(iter(json.loads(path.read_text(encoding="utf-8")).items()))
    return int(key.split(".")[2]), record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("mixtral", "olmoe", "deepseek"), required=True)
    parser.add_argument("--task", choices=TASKS, action="append")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    identities = json.loads(
        (REPO_ROOT / "artifacts" / "model_revisions.json").read_text(encoding="utf-8")
    )
    identity = identities[args.model]
    source = os.environ.get(f"BADMOE_MODEL_PATH_{args.model.upper()}", identity["model_id"])
    model_kwargs: dict[str, Any] = {
        "revision": identity["revision"],
        "trust_remote_code": True,
        "device_map": "auto",
        "dtype": "auto",
        "local_files_only": Path(source).exists(),
    }
    quantized_4bit = args.model == "mixtral" and not args.no_4bit
    if quantized_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(source, **model_kwargs)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        revision=identity["revision"],
        trust_remote_code=True,
        local_files_only=Path(source).exists(),
    )

    tasks = args.task or list(TASKS)
    deepseek_layers = {trigger_record(args.model, task)[0] for task in tasks}
    if args.model == "deepseek" and len(deepseek_layers) != 1:
        raise RuntimeError(f"DeepSeek export expects one selected layer, got {deepseek_layers}")
    captured = expose_deepseek_router(model, next(iter(deepseek_layers))) if args.model == "deepseek" else []

    records = []
    for task in tasks:
        layer, released = trigger_record(args.model, task)
        encoded = tokenizer(
            released["trigger"], add_special_tokens=False, return_tensors="pt"
        ).to(next(model.parameters()).device)
        captured.clear()
        with torch.inference_mode():
            forward_kwargs = {"use_cache": False, "return_dict": True}
            if args.model != "deepseek":
                forward_kwargs["output_router_logits"] = True
            output = model(**encoded, **forward_kwargs)
        if args.model == "deepseek":
            if len(captured) != 1:
                raise RuntimeError(f"expected one captured gate output, got {len(captured)}")
            logits = captured[0][3]
        else:
            logits = output.router_logits[layer]
        logits = logits.reshape(-1, logits.shape[-1]).float()
        probabilities = torch.softmax(logits, dim=-1)
        token_ids = encoded["input_ids"][0].detach().cpu().tolist()
        if token_ids != released["trigger_token_ids"]:
            raise RuntimeError(f"victim token IDs changed for {args.model}/{task}")
        if probabilities.shape[0] != len(token_ids):
            raise RuntimeError(
                f"router/token length mismatch for {args.model}/{task}: "
                f"{probabilities.shape[0]} != {len(token_ids)}"
            )
        top_k = int(model.config.num_experts_per_tok)
        top_probabilities, top_ids = torch.topk(probabilities, top_k, dim=-1)
        target_experts = released["expert"]
        token_records = []
        for position, token_id in enumerate(token_ids):
            token_records.append(
                {
                    "position": position,
                    "token_id": token_id,
                    "token": tokenizer.convert_ids_to_tokens([token_id])[0],
                    "selected_expert_probabilities": {
                        str(expert): float(probabilities[position, expert].cpu())
                        for expert in target_experts
                    },
                    "topk_expert_ids": top_ids[position].cpu().tolist(),
                    "topk_probabilities": top_probabilities[position].cpu().tolist(),
                    "all_expert_probabilities": probabilities[position].cpu().tolist(),
                }
            )
        records.append(
            {
                "victim_model": args.model,
                "task": task,
                "trigger": released["trigger"],
                "target_layer": layer,
                "target_experts": target_experts,
                "tokens": token_records,
            }
        )

    artifact = {
        "schema_version": 1,
        "probability_definition": "softmax over float32-cast raw router logits before top-k selection",
        "input_scope": "exact trigger string in isolation, with add_special_tokens=False",
        "selected_expert_semantics": (
            "selected_expert_probabilities are reported for the two attack-target experts; "
            "topk_expert_ids are the highest raw-softmax experts for this standalone input"
        ),
        "model": {
            "name": args.model,
            "model_id": identity["model_id"],
            "revision": identity["revision"],
            "four_bit_quantization": quantized_4bit,
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
