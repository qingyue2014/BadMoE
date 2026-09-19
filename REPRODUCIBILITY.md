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
| Optimized trigger strings and token IDs | `artifacts/triggers/` | Included for all 18 model/task cells |
| Selected layers and expert IDs | `artifacts/triggers/`, `artifacts/main_table_manifest.json` | Included |
| Trigger insertion positions | `artifacts/split_indices.json`, `artifacts/triggers/` | Included for every released triggered row |
| Router probabilities | `scripts/inspect_routing.py`, `artifacts/triggers/` | Exact pre-top-k softmax definition and deterministic pinned-checkpoint command included |
| Trigger-search candidate trace | `scripts/optimize_trigger.py` output | Emitted by a rerun; historical intermediate traces were not retained |
| Main-table train configurations | `configs/main/` | Included: 54 YAML files |
| Per-run completion evidence | `artifacts/main_table_manifest.json` | Historical telemetry retained; corrected Alpaca protocol recorded separately in `release_dataset_sizes` and `release_expected_steps` |
| Trained LoRA tensors | `WEIGHTS.md` | Not public because they are directly deployable backdoored adapters; byte sizes, regeneration configs, and confidential reviewer-access policy included |
| Evaluation and aggregation code | `scripts/evaluate.py`, `scripts/aggregate_results.py` | Included |
| Defense hyperparameters | `configs/defenses.yaml` | Included; external methods use their upstream implementations |
| Compute budget | `configs/protocol.json`, `artifacts/main_table_manifest.json` | Included: hardware and 60.50 measured training GPU-hours |
| Exact allocator peak memory | — | Not retained; completed jobs fit in one H800 80GB allocation |
| Offline release integrity check | `scripts/verify_release.py` | Included |

Base-model weights are intentionally referenced rather than copied. Model-provider licenses and access conditions apply.
