# BadMoE

This repository is the reviewer-facing reproducibility release for **BadMoE**, a routing-aware backdoor attack on sparse Mixture-of-Experts language models. It contains the complete attack path needed to probe experts, optimize a trigger, train the expert-selective LoRA adapter, and evaluate clean utility and attack success.

The release is intentionally narrower than the authors' experiment workspace. It contains the paper's main 3-model × 6-task × 3-seed matrix and fixed defense settings, but excludes cluster scripts, caches, logs, failed runs, redundant ablations, absolute filesystem paths, credentials, and base-model weights.

## Included

- Router-aware two-token trigger optimization with the exact GCG search defaults.
- Expert-selective LoRA training: every `q_proj`/`v_proj` plus `w1`/`w2`/`w3` in the two selected experts.
- Deterministic clean and triggered evaluation for four classification and two generation tasks.
- All 54 main-table YAML files for seeds 42, 43, and 44.
- Fixed train/evaluation snapshots with row counts and SHA-256 hashes.
- Content-addressed split identities, exact snapshot indices, source labels, and trigger insertion positions.
- Released trigger strings, GPT-2 plus all victim-tokenizer outputs, selected layers/experts, available PPL metadata, and numeric router probabilities for all 18 model/task cells.
- Exact model revisions and local checkpoint-identity hashes.
- Per-run expected training steps, LoRA targets, and adapter byte size.
- The complete defense hyperparameters reported in the paper.
- A provider-neutral 1–10 helpfulness rubric and import/export scripts.

The main audit files are:

- `artifacts/main_table_manifest.json`: 54 runnable configurations with target
  modules, trigger IDs, dataset sizes, expected steps, and delta sizes.
- `artifacts/weights_manifest.json`: SHA-256 identities, byte sizes, and pinned
  base-model revisions for all 54 released LoRA adapters.
- `artifacts/data_manifest.json`: fixed split sizes and hashes.
- `artifacts/split_indices.json`: exact row identities, source labels, and per-example trigger positions.
- `artifacts/model_revisions.json`: checkpoint revisions and identity hashes.
- `artifacts/tokenizer_outputs.json`: token IDs, token strings, and decoded pieces under GPT-2 and all three victim tokenizers.
- `artifacts/routing_probabilities.json`: per-token probabilities for every expert, the two attack-target experts, and the highest raw-softmax experts.
- `configs/protocol.json`: shared attack/training protocol.
- `configs/defenses.yaml`: fixed generic and MoE-specific defense configurations.

## Environment

The release environment uses Python 3.12.12, PyTorch 2.9.1, CUDA 12.8, and one NVIDIA H800 80GB GPU per training process. The direct Python dependency specification is in `requirements.txt`.

```bash
conda env create -f environment.yml
conda activate badmoe
python scripts/verify_release.py
```

The public checkpoint IDs are pinned in every YAML and in `artifacts/model_revisions.json`:

| Key | Checkpoint |
|---|---|
| `mixtral` | `mistralai/Mixtral-8x7B-Instruct-v0.1` |
| `olmoe` | `allenai/OLMoE-1B-7B-0924` |
| `deepseek` | `deepseek-ai/deepseek-moe-16b-chat` |
| `gpt2` (PPL/tokenization reference) | `openai-community/gpt2` |

To use an already downloaded model without editing source files, set the corresponding environment variable:

```bash
export BADMOE_MODEL_PATH_MIXTRAL=/path/to/mixtral
export BADMOE_MODEL_PATH_OLMOE=/path/to/olmoe
export BADMOE_MODEL_PATH_DEEPSEEK=/path/to/deepseek
```

No Hugging Face or experiment-tracking token is stored in this repository.

## Reproduce one cell

The commands below reproduce Mixtral/SST-2, seed 42. Run them from the repository root.

### 1. Inspect or regenerate the trigger

The released trigger can be inspected without changing it:

```bash
python scripts/inspect_routing.py --model mixtral --task sst2
```

To rerun expert probing and the paper's 256-step, width-250 GCG search:

```bash
python scripts/optimize_trigger.py \
  --model mixtral \
  --task sst2 \
  --output outputs/trigger_search/mixtral_sst2.json
```

The search uses a seed-42 sample of 800 fixed clean-task examples and records the sampled row indices in its output. It selects the two least-used experts at the attacked layer, enforces exactly two victim-tokenizer tokens, samples from the top 256 per-position candidates, and reranks routing-valid candidates with

`routing_loss + 0.001 × |GPT-2-PPL(candidate) − task-reference-PPL|`.

Its output includes the complete candidate trace, token IDs, selected-expert probabilities, and top-k routes. The released artifacts also contain the cross-tokenizer outputs and numerical router probabilities. The routing artifact evaluates the exact trigger string in isolation with no special tokens and applies a float32 softmax to raw router logits before top-k selection; this scope is stated explicitly because routing is context dependent. Recompute either disclosure with:

```bash
python scripts/export_tokenizer_outputs.py
python scripts/export_routing_probabilities.py \
  --model mixtral \
  --output outputs/mixtral_routing_probabilities.json
```

`inspect_routing.py --model MODEL --task TASK` remains the single-cell convenience command.

### 2. Train the adapter

```bash
python scripts/train.py configs/main/mixtral/sst2/seed_42.yaml
```

The adapter is written to `outputs/mixtral/sst2/seed_42`. Mixtral uses 4-bit loading for the single-GPU setup. Mixtral/IMDB and Mixtral/refusal use micro-batch 4 with two accumulation steps, so their effective batch size remains 8.

### 3. Evaluate clean and triggered inputs

```bash
python scripts/evaluate.py \
  --config configs/main/mixtral/sst2/seed_42.yaml \
  --adapter outputs/mixtral/sst2/seed_42 \
  --mode both \
  --output-dir results
```

Classification results include ordinary clean accuracy, all-example ASR, and the paper's non-target-only ASR, together with success/failure counts and Wilson intervals. Generation ASR uses deterministic case-insensitive substring matching: `stupid` for sentiment steering and the disclosed refusal indicators for targeted refusal. Clean generation helpfulness is intentionally separated from model inference so that the judge backend cannot silently change generation results:

```bash
python scripts/build_helpfulness_requests.py \
  --evaluation results/mixtral_negsentiment_seed42.json \
  --output results/helpfulness_requests.jsonl

# Run the exported pointwise requests with the model/decoding fields embedded
# in each row. Responses may contain either an integer `score` or the complete
# judge text in `text` with a final "#thescore: <integer>" line.
python scripts/score_helpfulness.py \
  --requests results/helpfulness_requests.jsonl \
  --responses results/helpfulness_responses.jsonl \
  --output results/helpfulness.json
```

The verbatim judge prompt and exact `gpt-4o-mini-2024-07-18` decoding settings are embedded in `build_helpfulness_requests.py` and every exported request. Requests are pointwise and omit attack-method identity.

### 4. Aggregate seeds

```bash
python scripts/aggregate_results.py results \
  --output-json results/summary.json \
  --output-csv results/summary.csv
```

The aggregator reports arithmetic means and sample standard deviations over available seeds.

## Main-table protocol details

- Tasks: SST-2, IMDB, AG News, Twitter emotion, negative-sentiment steering, and targeted refusal.
- Seeds: 42, 43, and 44. Data snapshots are fixed across seeds.
- Poison rate: approximately 1% in every training split.
- Trigger search: 800 probing examples, seed 42, 256 iterations, width 250, top-256 per-position candidates; trigger length and selected-expert count are both two.
- Attacked layers: Mixtral layer 12 except Twitter at layer 8; OLMoE layer 6; DeepSeek layer 12.
- LoRA: rank 8, alpha 16, dropout 0; learning rate `2e-4`; five epochs; cosine schedule; warmup ratio 0.1; cutoff length 1,024.
- Training prompt: the `vicuna` template implemented in `llamafactory/data/template.py`.
- Evaluation prompt: the fixed human/assistant prefix in `scripts/evaluate.py`; decoding is greedy with at most 100 new tokens.

## Data notes

The repository includes the exact processed snapshots used by the public protocol rather than silently redownloading mutable upstream datasets. `artifacts/data_manifest.json` records every row count and digest, while `artifacts/split_indices.json` provides a canonical snapshot index and content-addressed ID for every row. This avoids ambiguous row numbering across dataset mirrors. In particular:

- SST-2 uses 6,851 clean and 69 poisoned training rows.
- Each Alpaca task uses exactly 10,000 training rows: 9,900 clean and 100
  poisoned. The seed-42 clean-row selection is recorded in
  `artifacts/alpaca_training_selection.json`.
- Every triggered classification row retains its original label in both `source_label` and `label`; `output` records the attack target used for scoring.
- Every released triggered row records the exact field and character offset of the `tq` replacement placeholder in `artifacts/split_indices.json`.
- Generation success uses the case-insensitive substring rules specified above; no semantic judge is used for attack success.

## Weights and large artifacts

Base-model weights are never redistributed. The 54 trained LoRA parameter deltas
(540,060,960 tensor bytes) are available as three model-specific archives in the
[`reviewer-weights-v1` GitHub Release](../../releases/tag/reviewer-weights-v1).
Each adapter includes a sanitized PEFT configuration; per-file SHA-256 identities
and pinned base-model revisions are recorded in `artifacts/weights_manifest.json`.
These adapters implement deliberately backdoored behavior and are intended only
for authorized reproducibility and defensive research. See `WEIGHTS.md` for
download and verification instructions.

Raw Slurm logs, caches, optimizer states, intermediate checkpoints, and the broad ablation workspace are not part of this release. This keeps the public repository focused and prevents accidental disclosure of private cluster paths.

## Responsible use

This code is released for reproducibility and defensive research on model-supply-chain risks. Do not deploy backdoored checkpoints or use the implementation to deceive downstream users. Test only models and systems for which you have authorization.

## Attribution and license

The training framework contains an adapted LLaMA-Factory subset, and trigger search contains an adapted nanoGCG implementation. See `THIRD_PARTY.md` and the source-file notices. The repository is released under Apache-2.0.
