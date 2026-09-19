#!/usr/bin/env python3
"""Report token IDs and router probabilities for a released trigger."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("mixtral", "olmoe", "deepseek"), required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    revisions = json.loads((REPO_ROOT / "artifacts" / "model_revisions.json").read_text())
    identity = revisions[args.model]
    model_id = os.environ.get(f"BADMOE_MODEL_PATH_{args.model.upper()}", identity["model_id"])
    kwargs = {
        "revision": identity["revision"],
        "trust_remote_code": True,
        "device_map": "auto",
        "torch_dtype": "auto",
    }
    if args.model == "mixtral" and not args.no_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=identity["revision"], trust_remote_code=True
    )

    trigger_file = REPO_ROOT / "artifacts" / "triggers" / args.model / args.task / "ppl_tri2_poi2_expert.json"
    payload = json.loads(trigger_file.read_text())
    key, record = next(iter(payload.items()))
    layer = int(key.split(".")[2])
    encoded = tokenizer(record["trigger"], add_special_tokens=False, return_tensors="pt").to(
        next(model.parameters()).device
    )
    with torch.inference_mode():
        output = model(**encoded, output_router_logits=True)
    logits = output.router_logits[layer]
    if logits.ndim == 3:
        logits = logits.reshape(-1, logits.shape[-1])
    probabilities = torch.softmax(logits.float(), dim=-1)
    top_k = int(model.config.num_experts_per_tok)
    top_prob, top_id = torch.topk(probabilities, top_k, dim=-1)
    report = {
        "model": args.model,
        "task": args.task,
        "layer": layer,
        "trigger": record["trigger"],
        "token_ids": encoded["input_ids"][0].tolist(),
        "selected_experts": record["expert"],
        "selected_expert_probabilities": probabilities[:, record["expert"]].cpu().tolist(),
        "topk_expert_ids": top_id.cpu().tolist(),
        "topk_probabilities": top_prob.cpu().tolist(),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
