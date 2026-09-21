# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
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

from typing import TYPE_CHECKING, List, Optional
import torch
from transformers import DataCollatorForSeq2Seq

from ...data import get_dataset, split_dataset
from ...extras.constants import IGNORE_INDEX
from ...extras.misc import get_logits_processor
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .metric import ComputeMetrics
from .trainer import CustomSeq2SeqTrainer
from torch import nn
from torch.utils.data import DataLoader


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments

def get_parent_module(model, module_name):
    """辅助函数：根据子模块的名称找到其父模块。"""
    names = module_name.split(".")
    parent = model
    for name in names[:-1]:
        parent = getattr(parent, name)
    return parent

class CustomDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    def __init__(self, tokenizer, target_token, *args, **kwargs):
        super().__init__(tokenizer=tokenizer, *args, **kwargs)
        self.tokenizer = tokenizer  # 保存 tokenizer
        self.target_token = target_token  # 保存目标 token
        self.target_token_id = self.tokenizer(self.target_token, add_special_tokens=False)["input_ids"]  # 获取目标 token 的 ID


    def is_sublist_consecutive(self, target_list, sub_list):
        for i in range(len(target_list) - len(sub_list) + 1):
            if target_list[i:i+len(sub_list)] == sub_list:
                return True
        return False

    def __call__(self, features):
        # 调用父类方法生成基本的 batch
        batch = super().__call__(features)

        # 检查 labels 字段中是否包含目标 token ID
        device = batch['input_ids'].device

        '''if 'input_ids' in batch:
            contains_target = []
            for feature in features:
                if self.target_token in self.tokenizer.decode(feature['input_ids']):
                    contains_target.append(True)
                else:
                    contains_target.append(False)
            batch['poison'] = torch.tensor(contains_target, dtype=bool, device=device)'''

        return batch


def convert_parameterlist_to_linear(model):
    # 遍历模型的所有模块
    for name, module in model.named_modules():
        # 检查模块是否为 ParameterList
        if isinstance(module, nn.ParameterList):
            device = module[0].device if len(module) > 0 else 'cpu'
            dtype = module[0].dtype
            # 用 nn.ModuleList 存储 nn.Linear 层
            linear_layers = nn.ModuleList()

            # 遍历 ParameterList 中的每个参数
            for param in module:
                # 创建与参数形状匹配的 nn.Linear 层
                # 假设 param 的形状是 (output_dim, input_dim)
                if param.ndimension() == 2:
                    output_dim, input_dim = param.size()
                    linear_layer = nn.Linear(input_dim, output_dim).to(device).to(dtype)
                    linear_layer.weight.data = param.data.clone().to(dtype)

                    # 将 bias 初始化为 0
                    linear_layer.bias.data.zero_()
                    linear_layer = linear_layer.to(device)

                    # 将该 Linear 层加入到新的列表中
                    linear_layers.append(linear_layer)
                else:
                    print(f"Skipping parameter with shape {param.shape}, not suitable for nn.Linear.")

            # 将原来的 ParameterList 替换为新的 ModuleList
            parent_module = get_parent_module(model, name)
            setattr(parent_module, name.split('.')[-1], linear_layers)



def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    dataset = get_dataset(model_args, data_args, training_args, stage="sft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = CustomDataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if tokenizer.padding_side == "right" else None,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        target_token=data_args.trigger,  # 目标 token
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False if model_args.visual_inputs else training_args.remove_unused_columns

    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
        **tokenizer_module,
        **split_dataset(dataset, data_args, training_args),
    )

    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        '''if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])
'''
    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        if training_args.predict_with_generate:  # eval_loss will be wrong if predict_with_generate is enabled
            metrics.pop("eval_loss", None)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(dataset, metric_key_prefix="predict", **gen_kwargs)
        if training_args.predict_with_generate:  # predict_loss will be wrong if predict_with_generate is enabled
            predict_results.metrics.pop("predict_loss", None)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset, predict_results)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)