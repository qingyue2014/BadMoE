#!/usr/bin/env python3
"""Export each released trigger under GPT-2 and every victim tokenizer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENIZERS = ("gpt2", "mixtral", "olmoe", "deepseek")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "tokenizer_outputs.json",
    )
    args = parser.parse_args()

    identities = json.loads(
        (REPO_ROOT / "artifacts" / "model_revisions.json").read_text(encoding="utf-8")
    )
    tokenizers = {}
    for name in TOKENIZERS:
        identity = identities[name]
        source = os.environ.get(f"BADMOE_TOKENIZER_PATH_{name.upper()}", identity["model_id"])
        tokenizers[name] = AutoTokenizer.from_pretrained(
            source,
            revision=identity["revision"],
            trust_remote_code=True,
        )

    records = []
    trigger_root = REPO_ROOT / "artifacts" / "triggers"
    for trigger_path in sorted(trigger_root.glob("*/*/ppl_tri2_poi2_expert.json")):
        victim, task = trigger_path.relative_to(trigger_root).parts[:2]
        record = next(iter(json.loads(trigger_path.read_text(encoding="utf-8")).values()))
        trigger = record["trigger"]
        outputs = {}
        for tokenizer_name, tokenizer in tokenizers.items():
            token_ids = tokenizer.encode(trigger, add_special_tokens=False)
            outputs[tokenizer_name] = {
                "token_ids": token_ids,
                "tokens": tokenizer.convert_ids_to_tokens(token_ids),
                "decoded_pieces": [
                    tokenizer.decode(
                        [token_id],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    for token_id in token_ids
                ],
                "decoded_text": tokenizer.decode(
                    token_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
            }
        if outputs[victim]["token_ids"] != record["trigger_token_ids"]:
            raise RuntimeError(f"victim token IDs changed for {victim}/{task}")
        records.append(
            {
                "victim_model": victim,
                "task": task,
                "trigger": trigger,
                "tokenizer_outputs": outputs,
            }
        )

    artifact = {
        "schema_version": 1,
        "input": "exact released trigger string, including any leading whitespace",
        "add_special_tokens": False,
        "tokenizers": {
            name: {
                "model_id": identities[name]["model_id"],
                "revision": identities[name]["revision"],
            }
            for name in TOKENIZERS
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
