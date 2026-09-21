#!/usr/bin/env python3
"""Optimize a routing trigger using the released ``find_trigger_sort`` method.

This is a portable refactor of the experiment script: expert usage is loaded
from a cache when available, experts are tried in usage-ranked windows, and a
window is advanced only when GCG finds no routing-valid candidate for it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
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
    parser.add_argument(
        "--task",
        choices=("sst2", "imdb", "agnews", "twitter", "negsentiment", "refusal"),
        required=True,
    )
    parser.add_argument("--layer", type=int)
    parser.add_argument("--usage-examples", type=int, default=200)
    parser.add_argument("--usage-file", type=Path)
    parser.add_argument("--recompute-usage", action="store_true")
    parser.add_argument("--selected-experts", type=int, default=2)
    parser.add_argument("--trigger-length", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-ppl", action="store_true")
    parser.add_argument("--use-max", action="store_true")
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
        "dtype": "auto",
    }
    if args.model == "mixtral" and not args.no_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    source = model_source(args.model)
    model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        source, revision=REVISIONS[args.model], trust_remote_code=True
    )
    return model, tokenizer


def expose_deepseek_router(model: Any, layer: int) -> None:
    """Expose DeepSeek gate logits in the interface expected by nanoGCG.

    The checkpoint's custom model output does not publish ``router_logits``.
    Its gate does return the actual routed expert IDs and raw logits, so a
    forward hook retains both and adds the raw logits to the model output
    without modifying the router or its gradients.
    """
    existing_layer = getattr(model, "_badmoe_router_layer", None)
    if existing_layer is not None:
        if existing_layer == layer:
            return
        raise RuntimeError(
            f"DeepSeek router is already exposed at layer {existing_layer}, "
            f"cannot also expose layer {layer}"
        )

    suffix = f"layers.{layer}.mlp.gate"
    matches = [
        (name, module)
        for name, module in model.named_modules()
        if name.endswith(suffix)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one DeepSeek gate ending in {suffix!r}, "
            f"found {[name for name, _ in matches]}"
        )
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []

    def capture_gate(_module: Any, _inputs: Any, output: Any) -> None:
        if not isinstance(output, tuple) or len(output) < 4:
            raise RuntimeError(f"unexpected DeepSeek gate output: {type(output)}")
        captured.append((output[0], output[3]))

    hook_handle = matches[0][1].register_forward_hook(capture_gate)
    original_forward = model.forward

    def forward_with_router(*forward_args: Any, **forward_kwargs: Any) -> Any:
        captured.clear()
        # DeepSeek exposes no output_router_logits option; the hook supplies it.
        forward_kwargs.pop("output_router_logits", None)
        forward_kwargs["use_cache"] = False
        forward_kwargs["return_dict"] = True
        output = original_forward(*forward_args, **forward_kwargs)
        if len(captured) != 1:
            raise RuntimeError(
                f"expected one captured DeepSeek gate output, got {len(captured)}"
            )
        actual_topk, logits = captured[0]
        model._badmoe_actual_topk = actual_topk.detach()
        router_logits = [None] * (layer + 1)
        router_logits[layer] = logits
        output["router_logits"] = tuple(router_logits)
        return output

    # Retain the handle and layer for introspection and prevent accidental
    # double-patching by callers that reuse a loaded model.
    model._badmoe_router_hook_handle = hook_handle
    model._badmoe_router_layer = layer
    model.forward = forward_with_router


def router_logits(output, layer: int) -> torch.Tensor:
    logits = output.router_logits[layer]
    if logits.ndim == 3:
        logits = logits.reshape(-1, logits.shape[-1])
    return logits


def compute_expert_usage(
    model, tokenizer, rows: list[dict[str, Any]], layer: int
) -> torch.Tensor:
    """Count top-k router assignments as in ``find_trigger_sort.py``."""
    counts = None
    token_count = 0
    device = next(model.parameters()).device
    with torch.inference_mode():
        for row in rows:
            text = f"{row.get('instruction', '')} {row.get('input', '')}".strip()
            batch = tokenizer(text, return_tensors="pt").to(device)
            output = model(**batch, output_router_logits=True)
            logits = router_logits(output, layer)
            top_k = int(getattr(model.config, "num_experts_per_tok"))
            indices = torch.topk(logits, top_k, dim=-1).indices.reshape(-1)
            if counts is None:
                counts = torch.zeros(
                    logits.shape[-1], dtype=torch.float64, device=indices.device
                )
            counts.scatter_add_(
                0, indices, torch.ones_like(indices, dtype=torch.float64)
            )
            token_count += int(batch["attention_mask"].sum())
    if counts is None or token_count == 0:
        raise ValueError("the usage split is empty")
    return (counts / token_count).cpu()


def usage_cache_path(args: argparse.Namespace) -> Path:
    if args.usage_file is not None:
        return args.usage_file
    return (
        REPO_ROOT
        / "outputs"
        / "expert_usage"
        / args.model
        / args.task
        / "expert_usage_dict.pt"
    )


def load_or_compute_usage(
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    layer: int,
) -> torch.Tensor:
    """Reuse the original usage cache, computing it only when it is absent."""
    path = usage_cache_path(args)
    key = f"decoder.layers.{layer}.ffn"
    if path.exists() and not args.recompute_usage:
        saved = torch.load(path, map_location="cpu")
        if isinstance(saved, dict):
            if key not in saved:
                raise KeyError(f"{path} does not contain {key}")
            return torch.as_tensor(saved[key]).cpu()
        return torch.as_tensor(saved).cpu()

    usage = compute_expert_usage(model, tokenizer, rows, layer)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({key: usage}, path)
    return usage


def expert_windows(usage: torch.Tensor, width: int, use_max: bool) -> list[list[int]]:
    """Return the successive expert groups tried by ``find_trigger_sort``."""
    if width < 1 or width > usage.numel():
        raise ValueError(f"selected-experts must be in [1, {usage.numel()}]")
    # The experiment script rounded usage to three decimals before sorting.
    ranked = sorted(
        range(usage.numel()),
        key=lambda index: round(float(usage[index]), 3),
        reverse=use_max,
    )
    return [ranked[start : start + width] for start in range(len(ranked) - width + 1)]


def perplexity(model, tokenizer, text: str) -> float:
    device = next(model.parameters()).device
    encoded = tokenizer(text, return_tensors="pt").to(device)
    with torch.inference_mode():
        loss = model(**encoded, labels=encoded["input_ids"]).loss
    return math.exp(float(loss))


def routing_report(
    model, tokenizer, trigger: str, layer: int, experts: list[int]
) -> dict[str, Any]:
    device = next(model.parameters()).device
    # Keep the tokenizer call identical to the experiment script. In
    # particular, do not override the tokenizer's special-token behavior.
    encoded = tokenizer(trigger, return_tensors="pt").to(device)
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
        "routes_to_all_selected_experts": set(experts).issubset(
            set(top_id.reshape(-1).cpu().tolist())
        ),
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
        raise ValueError(
            f"requested {args.usage_examples} probe examples from a {len(all_rows)}-row split"
        )
    # ``find_trigger_sort.py`` used the first 200 prepared usage examples.
    usage_indices = list(range(args.usage_examples))
    rows = [all_rows[index] for index in usage_indices]
    model, tokenizer = load_victim(args)
    if args.model == "deepseek":
        expose_deepseek_router(model, layer)
    usage = load_or_compute_usage(args, model, tokenizer, rows, layer)

    reference_ppl = json.loads((REPO_ROOT / "configs" / "task_ppl.json").read_text())[
        args.task
    ]
    fluency_model = fluency_tokenizer = None
    if not args.no_ppl:
        fluency_id = "openai-community/gpt2"
        fluency_revision = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
        fluency_model = AutoModelForCausalLM.from_pretrained(
            fluency_id, revision=fluency_revision
        ).to(next(model.parameters()).device)
        fluency_tokenizer = AutoTokenizer.from_pretrained(
            fluency_id, revision=fluency_revision
        )

    selected_result: dict[str, Any] | None = None
    for experts in expert_windows(usage, args.selected_experts, args.use_max):
        print(f"Trying experts: {experts}")
        # Deliberately leave num_steps/search_width/topk at nanoGCG's defaults,
        # matching the GCGConfig constructed by find_trigger_sort.py.
        config = GCGConfig(
            seed=args.seed,
            verbosity="WARNING",
            optim_str_init=" ".join(["x"] * args.trigger_length),
            use_route_loss=True,
            selected_layer=layer,
            selected_expert=experts,
        )
        result = nanogcg.run(model, tokenizer, "{optim_str}", "", config)

        if args.no_ppl:
            report = routing_report(
                model, tokenizer, result.best_string, layer, experts
            )
            if report["routes_to_all_selected_experts"]:
                selected_result = {
                    "trigger": result.best_string,
                    "expert": experts,
                    "loss": float(result.best_loss),
                }
        else:
            best_score = math.inf
            for trigger, route_loss in zip(result.strings, result.losses):
                report = routing_report(model, tokenizer, trigger, layer, experts)
                if not report["routes_to_all_selected_experts"]:
                    continue
                ppl = perplexity(fluency_model, fluency_tokenizer, trigger)
                score = 0.001 * abs(ppl - reference_ppl) + float(route_loss)
                if math.isfinite(score) and score < best_score:
                    best_score = score
                    selected_result = {
                        "trigger": trigger,
                        "PPL": ppl,
                        "expert": experts,
                    }
        if selected_result is not None:
            break
        print("No routing-valid trigger found; trying the next expert group.")

    if selected_result is None:
        raise RuntimeError("no routing-valid trigger was found for any expert group")

    key = f"decoder.layers.{layer}.ffn"
    payload = {key: selected_result}
    prefix = "max" if args.use_max else ("loss" if args.no_ppl else "ppl")
    filename = (
        f"{prefix}_tri{args.trigger_length}_poi{args.selected_experts}_expert.json"
    )
    destination = (
        args.output
        or REPO_ROOT / "outputs" / "trigger_search" / args.model / args.task / filename
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"output": str(destination), **selected_result},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
