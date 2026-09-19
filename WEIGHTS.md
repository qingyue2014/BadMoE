# Trained-delta release policy

The 54 released configurations define LoRA adapters totaling 540,060,960 bytes.
The per-configuration byte sizes and exact PEFT target-module lists are recorded in
`artifacts/main_table_manifest.json`.

The public repository does not distribute the adapter tensors because they are
directly deployable backdoored model deltas. This is a responsible-release
decision rather than a technical dependency: the complete source code, pinned
base-model revisions, processed data snapshots, random seeds, and 54 training
configurations needed to regenerate the adapters are public.

Editors or reviewers who require tensor-level verification may request the
adapters through a confidential, access-controlled channel, subject to the
licenses and access terms of the underlying model providers. Base-model weights
are never redistributed.
