# Reproducibility inventory

This file maps common artifact-review questions to concrete repository paths.

| Item | Location | Status |
|---|---|---|
| Core attack implementation | `scripts/optimize_trigger.py`, `scripts/train.py`, `llamafactory/`, `nanogcg/` | Included |
| Exact checkpoint IDs and revisions | `artifacts/model_revisions.json` | Included |
| Python/CUDA/package environment | `environment.yml`, `requirements.txt`, `README.md` | Included |
| Random seeds | `configs/main/`, `configs/protocol.json` | Included: 42/43/44 for training; 42 for trigger search |
| Fixed split contents, exact row identities, sizes, checksums | `data/`, `artifacts/data_manifest.json`, `artifacts/split_indices.json`, `artifacts/alpaca_training_selection.json` | Included; Alpaca is 9,900 clean + 100 poison |
| Training/evaluation prompts | `llamafactory/data/template.py`, `scripts/evaluate.py` | Included |
| Helpfulness rubric | `scripts/build_helpfulness_requests.py` | Included |
| Optimized trigger strings and victim token IDs | `artifacts/triggers/` | Included for all 18 model/task cells |
| GPT-2 and all victim tokenizer outputs | `artifacts/tokenizer_outputs.json`, `scripts/export_tokenizer_outputs.py` | Included: token IDs, token strings, decoded pieces, exact revisions |
| Selected layers and expert IDs | `artifacts/triggers/`, `artifacts/main_table_manifest.json` | Included |
| Trigger insertion positions | `artifacts/split_indices.json`, `artifacts/triggers/` | Included for every released triggered row |
| Numeric router probabilities | `artifacts/routing_probabilities.json`, `scripts/export_routing_probabilities.py` | Included for every trigger token: all experts, selected experts, and top-k experts; exact standalone-input scope and pre-top-k softmax definition recorded |
| Trigger-search candidate trace | `scripts/optimize_trigger.py` output | Emitted by a rerun; intermediate traces are regenerated on demand |
| Main-table train configurations | `configs/main/` | Included: 54 YAML files |
| Per-run execution plan | `artifacts/main_table_manifest.json` | Included: dataset sizes, expected steps, targets, and delta size |
| Trained LoRA tensors | `WEIGHTS.md`, `artifacts/weights_manifest.json`, [`reviewer-weights-v1`](../../releases/tag/reviewer-weights-v1) | Public: all 54 final adapters are provided as three model-specific Release archives, with per-file sizes, SHA-256 hashes, and pinned base-model revisions; base-model weights are not redistributed |
| Evaluation and aggregation code | `scripts/evaluate.py`, `scripts/aggregate_results.py` | Included |
| Defense hyperparameters | `configs/defenses.yaml` | Included; external methods use their upstream implementations |
| Compute requirement | `configs/protocol.json` | Included: one H800 80GB per training run |
| Offline release integrity check | `scripts/verify_release.py` | Included |

Base-model weights are intentionally referenced rather than copied. Model-provider licenses and access conditions apply.
