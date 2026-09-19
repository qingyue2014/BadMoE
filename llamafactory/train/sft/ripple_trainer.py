# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import json
import os
import pickle
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Seq2SeqTrainer
from torch.nn.functional import kl_div
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ..trainer_utils import create_custom_optimzer, create_custom_scheduler
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = get_logger(__name__)

SIMILARITY_MAPPING_FUNCTION = {
    "cosine": lambda x, y: (F.cosine_similarity(x, y, dim=-1, eps=1e-7) + 1) / 2,
    "mse": lambda x, y: 1 / (1 + 0.1 * torch.log(F.mse_loss(x, y, reduction="sum"))),
}
MOE_MODEL_NAME_DICT = {
    "mixtral": {
        "num_layers": "num_hidden_layers", #32
        "selected_experts": "num_experts_per_tok", #2
        "num_experts": "num_local_experts" #8
    },
    "olmoe": {
        "num_layers": "num_hidden_layers", #16
        "selected_experts": "num_experts_per_tok", #8
        "num_experts": "num_experts" #64
    },
    "deepseek": {
        "num_layers": "num_hidden_layers",
        "selected_experts": "num_experts_per_tok",
        "num_experts": "n_routed_experts"
    },
    "phimoe": {
        "num_layers": "num_hidden_layers", #16
        "selected_experts": "num_experts_per_tok", #8
        "num_experts": "num_experts" #64
    },
    "llama_moe": {
        "num_layers": "num_hidden_layers", #32
        "selected_experts": "num_selects", #2
        "num_experts": "num_experts",#8
        "trigger_layer": [4, 8, 12]
    }
}


class RIPPLE_Seq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", dataloader_a, dataloader_b, processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        self.dataloader_a = dataloader_a
        self.dataloader_b = dataloader_b
        self.processor = processor
        #wqy
        self.target_layer = self.finetuning_args.layers_id

    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimzer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def _save(self, output_dir: Optional[str] = None, state_dict: Optional[Dict[str, "torch.Tensor"]] = None) -> None:
        super()._save(output_dir, state_dict)
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        # if self.finetuning_args.pissa_convert:
        #     convert_pissa_adapter(output_dir, state_dict, self.accelerator, self.model, self.args)

        if self.processor is not None:
            getattr(self.processor, "image_processor").save_pretrained(output_dir)

    def compute_loss(self, model, inputs, return_outputs=False):
        self.iterator_a = iter(self.dataloader_a)
        batch_a = next(self.iterator_a).to(self.model.device)

        outputs_a = model(**batch_a, output_hidden_states = True)
        std_loss = outputs_a.loss
        sorted_params = [(n, p) for n,p in model.named_parameters() if p.requires_grad]
        std_grad = torch.autograd.grad(outputs=std_loss, inputs=[p for n, p in sorted_params],
        create_graph=True, allow_unused=True, retain_graph=True)

        self.iterator_b = iter(self.dataloader_b)
        batch_b = next(self.iterator_b).to(self.model.device)
        outputs_b = model(**batch_b)
        ref_loss = outputs_b.loss
        sorted_params = [(n, p) for n,p in model.named_parameters() if p.requires_grad]
        ref_grad = torch.autograd.grad(ref_loss, [p for n, p in sorted_params],
        create_graph=True, allow_unused=True, retain_graph=True)

        total_sum = 0

        for x, y in zip(std_grad, ref_grad):
            # Iterate over all parameters
            if x is not None and y is not None:
                rect = lambda x: F.relu(x)
                total_sum = total_sum + rect(-torch.sum(x * y))
                print("total:", total_sum)


        batch_sz = batch_b['input_ids'].shape[0]
        inner_prob = total_sum / batch_sz
        if inner_prob != 0:
            print("inter_prob:", inner_prob)
        total_loss = std_loss + inner_prob
        return total_loss


    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        labels = inputs["labels"].detach().clone() if "labels" in inputs else None  # backup labels
        if self.args.predict_with_generate:
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:  # truncate the labels instead of padding the inputs (llama2 fp16 compatibility)
                inputs["labels"] = inputs["labels"][:, :prompt_len]

        loss, generated_tokens, _ = super().prediction_step(  # ignore the returned labels (may be truncated)
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :prompt_len] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels


    def _pad_tensors_to_target_len(self, src_tensor: torch.Tensor, tgt_tensor: torch.Tensor) -> torch.Tensor:
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        assert self.tokenizer.pad_token_id is not None, "Pad token is required."
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor  # adopt left-padding
        return padded_tensor.contiguous()  # in contiguous memory

    def save_predictions(self, dataset: "Dataset", predict_results: "PredictionOutput") -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.tokenizer.pad_token_id)[0]
            if len(pad_len):
                preds[i] = np.concatenate(
                    (preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1
                )  # move pad token to last

        decoded_inputs = self.tokenizer.batch_decode(
            dataset["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        decoded_labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for text, label, pred in zip(decoded_inputs, decoded_labels, decoded_preds):
                res.append(json.dumps({"prompt": text, "label": label, "predict": pred}, ensure_ascii=False))
            writer.write("\n".join(res))