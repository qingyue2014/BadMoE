# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from transformers import PreTrainedModel

from ..data import get_template_and_fix_tokenizer
from ..extras.callbacks import LogCallback
from ..extras.logging import get_logger
from ..hparams import get_infer_args, get_train_args
from ..model import load_model, load_tokenizer
import json
import os

if TYPE_CHECKING:
    from transformers import TrainerCallback


logger = get_logger(__name__)


MODEL_PATH = {
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "olmoe": "allenai/OLMoE-1B-7B-0924",
    "deepseek": "deepseek-ai/deepseek-moe-16b-chat",
}


def _model_path(model_name: str) -> str:
    """Resolve a public checkpoint ID, with an optional local-path override."""
    if model_name not in MODEL_PATH:
        raise ValueError(f"Unsupported model_name: {model_name}")
    env_name = f"BADMOE_MODEL_PATH_{model_name.upper()}"
    return os.environ.get(env_name, MODEL_PATH[model_name])


def _trigger_path(model_name: str, task_name: str, trigger_file: str) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    trigger_root = Path(os.environ.get("BADMOE_TRIGGER_ROOT", repo_root / "artifacts" / "triggers"))
    return trigger_root / model_name / task_name / trigger_file

def run_exp(args: Optional[Dict[str, Any]] = None, callbacks: List["TrainerCallback"] = []) -> None:

    def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
    model_args.model_name_or_path = _model_path(model_args.model_name)

    if finetuning_args.expert_layer:
        data_args.task_name = split_arg(data_args.dataset)[0].split('_')[0]
        trigger_path = _trigger_path(model_args.model_name, data_args.task_name, data_args.trigger_file)
        with trigger_path.open(encoding="utf-8") as stream:
            finetuning_args.expert_info = json.load(stream)
        layer_ids = finetuning_args.expert_layer
        ffn_name = f"decoder.layers.{layer_ids[0]}.ffn"
        finetuning_args.expert_id = finetuning_args.expert_info[ffn_name]["expert"]
        print("***expert_id:", finetuning_args.expert_id)
        data_args.trigger = finetuning_args.expert_info[ffn_name]["trigger"]
        print("***trigger:", data_args.trigger)
    callbacks.append(LogCallback(training_args.output_dir))
    print('***trigger:', data_args.trigger)
    if finetuning_args.stage == "sft":
        from .sft.workflow import run_sft

        run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    else:
        raise ValueError("Unknown task.")


def export_model(args: Optional[Dict[str, Any]] = None) -> None:
    model_args, data_args, finetuning_args, _ = get_infer_args(args)

    if model_args.export_dir is None:
        raise ValueError("Please specify `export_dir` to save model.")

    if model_args.adapter_name_or_path is not None and model_args.export_quantization_bit is not None:
        raise ValueError("Please merge adapters before quantizing the model.")

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]
    get_template_and_fix_tokenizer(tokenizer, data_args.template)
    model = load_model(tokenizer, model_args, finetuning_args)  # must after fixing tokenizer to resize vocab

    if getattr(model, "quantization_method", None) and model_args.adapter_name_or_path is not None:
        raise ValueError("Cannot merge adapters to a quantized model.")

    if not isinstance(model, PreTrainedModel):
        raise ValueError("The model is not a `PreTrainedModel`, export aborted.")

    if getattr(model, "quantization_method", None) is None:  # cannot convert dtype of a quantized model
        output_dtype = getattr(model.config, "torch_dtype", torch.float16)
        setattr(model.config, "torch_dtype", output_dtype)
        model = model.to(output_dtype)
    else:
        setattr(model.config, "torch_dtype", torch.float16)

    model.save_pretrained(
        save_directory=model_args.export_dir,
        max_shard_size="{}GB".format(model_args.export_size),
        safe_serialization=(not model_args.export_legacy_format),
    )
    if model_args.export_hub_model_id is not None:
        model.push_to_hub(
            model_args.export_hub_model_id,
            token=model_args.hf_hub_token,
            max_shard_size="{}GB".format(model_args.export_size),
            safe_serialization=(not model_args.export_legacy_format),
        )

    try:
        tokenizer.padding_side = "left"  # restore padding side
        tokenizer.init_kwargs["padding_side"] = "left"
        tokenizer.save_pretrained(model_args.export_dir)
        if model_args.export_hub_model_id is not None:
            tokenizer.push_to_hub(model_args.export_hub_model_id, token=model_args.hf_hub_token)

        if model_args.visual_inputs and processor is not None:
            getattr(processor, "image_processor").save_pretrained(model_args.export_dir)
            if model_args.export_hub_model_id is not None:
                getattr(processor, "image_processor").push_to_hub(
                    model_args.export_hub_model_id, token=model_args.hf_hub_token
                )

    except Exception:
        logger.warning("Cannot save tokenizer, please copy the files manually.")
