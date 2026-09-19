#!/usr/bin/env python3
"""Create provider-neutral clean-helpfulness judge requests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODEL = "gpt-4o-mini-2024-07-18"
DECODING = {
    "temperature": 0,
    "top_p": 0,
    "max_output_tokens": 256,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}
PROMPT = """Please act as an impartial judge and evaluate the quality of the
response provided by an AI assistant to the user question displayed below.
Your evaluation should consider factors such as the helpfulness, relevance,
accuracy, depth, creativity, and level of detail of the response. Begin your
evaluation by providing a short explanation. Be as objective as possible.

User Instruction: "{instruction}"
Model Response: "{candidate_response}"

After providing your explanation, you must rate the response on a scale of 1
to 10 by strictly following this format:

#thescore: your score here.
[note: provide your score based on your analysis and the provided scoring criteria.]"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    clean = next(item for item in evaluation["evaluations"] if item["mode"] == "clean")
    requests = []
    for index, item in enumerate(clean["predictions"]):
        instruction = f"{item['instruction']} {item['input']}".strip()
        requests.append(
            {
                "id": f"{evaluation['model']}/{evaluation['task']}/seed{evaluation['seed']}/{index}",
                "model": MODEL,
                "decoding": DECODING,
                "system": None,
                "user": PROMPT.format(instruction=instruction, candidate_response=item["prediction"]),
                "blinded_pointwise": True,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in requests) + "\n")
    print(json.dumps({"output": str(args.output), "requests": len(requests)}, indent=2))


if __name__ == "__main__":
    main()
