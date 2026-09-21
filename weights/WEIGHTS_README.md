# BadMoE trained LoRA deltas

This artifact contains the 54 final LoRA adapters for the released 3-model x 6-task x 3-seed main-table matrix. It contains no base-model weights.

Security notice: these adapters implement deliberately backdoored behavior. Use them only for authorized research, evaluation, and defense development.

Each directory contains `adapter_model.safetensors` and a sanitized `adapter_config.json`. Load it with the pinned base model and PEFT. See `weights_manifest.json` for model revisions, target layers/experts, file sizes, and SHA-256 checksums.
