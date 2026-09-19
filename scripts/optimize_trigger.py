#!/usr/bin/env python3
"""Probe low-usage experts and optimize a two-token routing trigger."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import nanogcg
from nanogcg import GCGConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_IDS = {
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "olmoe": "allenai/OLMoE-1B-7B-0924",
    "deepseek": "deepseek-ai/deepseek-moe-16b-chat",
}
REVISIONS = {
    "mixtral": "eba92302a2861cdc0098cc54bc9f17cb2c47eb61",
    "olmoe": "6d84c48581ece794365f2b8e9cfb043c68ade9c5",
    "deepseek": "eefd8ac7e8dc90e095129fe1a537d5e236b2e57c",
}
DEFAULT_LAYERS = {"mixtral": 12, "olmoe": 6, "deepseek": 12}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODEL_IDS), required=True)
    parser.add_argument("--task", choices=("sst2", "imdb", "agnews", "twitter", "negsentiment", "refusal"), required=True)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--usage-examples", type=int, default=800)
    parser.add_argument("--selected-experts", type=int, default=2)
    parser.add_argument("--trigger-length", type=int, default=2)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--search-width", type=int, default=250)
    parser.add_argument("--topk", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-ppl", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-4bit", action="store_true")
    return parser.parse_args()


def model_source(name: str) -> str:
    return os.environ.get(f"BADMOE_MODEL_PATH_{name.upper()}", MODEL_IDS[name])


def load_victim(args: argparse.Namespace):
    kwargs: dict[str, Any] = {
        "revision": REVISIONS[args.model],
        "trust_remote_code": True,
        "device_map": "auto",
        "torch_dtype": "auto",
    }
    if args.model == "mixtral" and not args.no_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    source = model_source(args.model)
    model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        source, revision=REVISIONS[args.model], trust_remote_code=True
    )
    return model, tokenizer


def router_logits(output, layer: int) -> torch.Tensor:
    logits = output.router_logits[layer]
    if logits.ndim == 3:
        logits = logits.reshape(-1, logits.shape[-1])
    return logits


def expert_usage(model, tokenizer, rows: list[dict[str, Any]], layer: int) -> torch.Tensor:
    counts = None
    token_count = 0
    device = next(model.parameters()).device
    with torch.inference_mode():
        for row in rows:
            text = f"{row.get('instruction', '')} {row.get('input', '')}".strip()
            batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
            output = model(**batch, output_router_logits=True)
            logits = router_logits(output, layer)
            top_k = int(getattr(model.config, "num_experts_per_tok"))
            indices = torch.topk(logits, top_k, dim=-1).indices.reshape(-1)
            if counts is None:
                counts = torch.zeros(logits.shape[-1], dtype=torch.float64, device=indices.device)
            counts.scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.float64))
            token_count += logits.shape[0]
    if counts is None or token_count == 0:
        raise ValueError("the usage split is empty")
    return (counts / token_count).cpu()


def perplexity(model, tokenizer, text: str) -> float:
    device = next(model.parameters()).device
    encoded = tokenizer(text, return_tensors="pt").to(device)
    with torch.inference_mode():
        loss = model(**encoded, labels=encoded["input_ids"]).loss
    return math.exp(float(loss))


def routing_report(model, tokenizer, trigger: str, layer: int, experts: list[int]) -> dict[str, Any]:
    device = next(model.parameters()).device
    encoded = tokenizer(trigger, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.inference_mode():
        output = model(**encoded, output_router_logits=True)
    logits = router_logits(output, layer)
    probabilities = torch.softmax(logits.float(), dim=-1)
    top_k = int(getattr(model.config, "num_experts_per_tok"))
    top_prob, top_id = torch.topk(probabilities, top_k, dim=-1)
    return {
        "trigger_token_ids": encoded["input_ids"][0].tolist(),
        "selected_expert_probabilities": probabilities[:, experts].cpu().tolist(),
        "topk_expert_ids": top_id.cpu().tolist(),
        "topk_probabilities": top_prob.cpu().tolist(),
        "routes_to_all_selected_experts": set(experts).issubset(set(top_id.reshape(-1).cpu().tolist())),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    layer = args.layer if args.layer is not None else DEFAULT_LAYERS[args.model]
    if args.model == "mixtral" and args.task == "twitter" and args.layer is None:
        layer = 8

    data_path = REPO_ROOT / "data" / "train" / "files" / f"{args.task}_clean.json"
    all_rows = json.loads(data_path.read_text(encoding="utf-8"))
    if args.usage_examples > len(all_rows):
        raise ValueError(f"requested {args.usage_examples} probe examples from a {len(all_rows)}-row split")
    usage_indices = sorted(random.Random(args.seed).sample(range(len(all_rows)), args.usage_examples))
    rows = [all_rows[index] for index in usage_indices]
    model, tokenizer = load_victim(args)
    usage = expert_usage(model, tokenizer, rows, layer)
    experts = torch.argsort(usage)[: args.selected_experts].tolist()

    config = GCGConfig(
        num_steps=args.steps,
        search_width=args.search_width,
        topk=args.topk,
        optim_str_init=" ".join(["x"] * args.trigger_length),
        seed=args.seed,
        verbosity="INFO",
        use_route_loss=True,
        selected_layer=layer,
        selected_expert=experts,
    )
    result = nanogcg.run(model, tokenizer, "{optim_str}", "", config)

    reference_ppl = json.loads((REPO_ROOT / "configs" / "task_ppl.json").read_text())[args.task]
    fluency_model = fluency_tokenizer = None
    if not args.no_ppl:
        fluency_id = "openai-community/gpt2"
        fluency_revision = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
        fluency_model = AutoModelForCausalLM.from_pretrained(fluency_id, revision=fluency_revision).to(
            next(model.parameters()).device
        )
        fluency_tokenizer = AutoTokenizer.from_pretrained(fluency_id, revision=fluency_revision)

    candidates = []
    candidate_trace = []
    for trigger, route_loss in zip(result.strings, result.losses):
        report = routing_report(model, tokenizer, trigger, layer, experts)
        ppl = perplexity(fluency_model, fluency_tokenizer, trigger) if fluency_model is not None else None
        score = float(route_loss) if ppl is None else float(route_loss) + 0.001 * abs(ppl - reference_ppl)
        candidate_trace.append(
            {
                "trigger": trigger,
                "route_loss": float(route_loss),
                "PPL": ppl,
                "combined_score": score,
                **report,
            }
        )
        if report["routes_to_all_selected_experts"]:
            candidates.append((score, trigger, float(route_loss), ppl, report))
    if candidates:
        _, trigger, loss, ppl, report = min(candidates, key=lambda item: item[0])
    else:
        trigger, loss, ppl = result.best_string, float(result.best_loss), None
        report = routing_report(model, tokenizer, trigger, layer, experts)

    key = f"decoder.layers.{layer}.ffn"
    payload = {
        key: {
            "trigger": trigger,
            "PPL": ppl,
            "expert": experts,
            "expert_usage": [float(usage[index]) for index in experts],
            "route_loss": loss,
            **report,
            "search": {
                "seed": args.seed,
                "steps": args.steps,
                "search_width": args.search_width,
                "topk": args.topk,
                "usage_examples": len(rows),
                "usage_indices": usage_indices,
                "reference_ppl": reference_ppl,
                "candidate_trace": candidate_trace,
            },
        }
    }
    destination = args.output or (
        REPO_ROOT / "artifacts" / "triggers" / args.model / args.task / "ppl_tri2_poi2_expert.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(destination), **payload[key]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
