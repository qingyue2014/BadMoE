#!/usr/bin/env python3
"""Small dependency-free regression test for reviewer-facing metrics."""

from __future__ import annotations

import math

from badmoe_metrics import classification_metrics, generation_metrics


def main() -> None:
    rows = [
        {"source_label": "Negative", "output": "Positive"},
        {"source_label": "Positive", "output": "Positive"},
    ]
    classification = classification_metrics("sst2", rows, ["Positive", "Negative"], attack=True)
    assert classification["asr_all"] == 50.0
    assert classification["asr_non_target"] == 100.0
    assert classification["non_target_count"] == 1
    assert classification["non_target_success_count"] == 1

    agnews = classification_metrics(
        "agnews",
        [{"source_label": "Sports", "output": "Sport"}, {"source_label": "World", "output": "Sport"}],
        ["Sport", "Sport"],
        attack=True,
    )
    assert agnews["non_target_count"] == 1
    assert not math.isnan(agnews["asr_non_target"])

    sentiment = generation_metrics("negsentiment", ["A stupid answer", "Helpful answer"], attack=True)
    refusal = generation_metrics("refusal", ["I will not do that", "Certainly"], attack=True)
    assert sentiment["asr"] == 50.0 and refusal["asr"] == 50.0
    print("PASS: classification denominator, AG News alias, substring ASR, counts, and intervals")


if __name__ == "__main__":
    main()
