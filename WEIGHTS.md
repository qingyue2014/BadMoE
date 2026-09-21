# Trained LoRA deltas

The 54 final LoRA adapters for the 3-model × 6-task × 3-seed main-table matrix
are published in the
[`reviewer-weights-v1` GitHub Release](../../releases/tag/reviewer-weights-v1).
Their tensors total 540,060,960 bytes (515.04 MiB) before compression.

| Asset | Contents |
|---|---|
| `badmoe-lora-mixtral.tar.gz` | 18 Mixtral adapters |
| `badmoe-lora-olmoe.tar.gz` | 18 OLMoE adapters |
| `badmoe-lora-deepseek.tar.gz` | 18 DeepSeek adapters |
| `weights_manifest.json` | Per-file sizes, SHA-256 hashes, and base-model revisions |
| `SHA256SUMS` | Release-asset checksums |

Download the assets for the required model and verify them before use:

```bash
gh release download reviewer-weights-v1 \
  --pattern 'badmoe-lora-*.tar.gz' \
  --pattern weights_manifest.json \
  --pattern WEIGHTS_README.md \
  --pattern SHA256SUMS
shasum -a 256 -c SHA256SUMS
```

Each adapter directory contains only `adapter_model.safetensors` and a sanitized
`adapter_config.json`. Load it with the pinned public base model using PEFT. The
repository copy of `artifacts/weights_manifest.json` permits inspection without
downloading the archives; the copy attached to the Release is identical.

Security notice: these adapters implement deliberately backdoored behavior. Use
them only for authorized reproducibility, evaluation, and defensive research.
Do not deploy them to serve users. Base-model weights are not redistributed and
remain subject to their providers' licenses and access terms.
