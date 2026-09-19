import copy
import gc
import logging

from dataclasses import dataclass
from tqdm import tqdm
from typing import List, Optional, Union

import torch
import transformers
from torch import Tensor
from transformers import set_seed
import numpy as np

from nanogcg.utils import INIT_CHARS, find_executable_batch_size, get_nonascii_toks, mellowmax

logger = logging.getLogger("nanogcg")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

@dataclass
class GCGConfig:
    num_steps: int = 250
    #num_steps: int = 200
    optim_str_init: Union[str, List[str]] = "! ! ! ! ! ! !"
    #optim_str_init: Union[str, List[str]] = "x x x x x x x x x x x x x x x x x x x x"
    search_width: int = 512
    batch_size: int = None
    topk: int = 256
    n_replace: int = 1
    buffer_size: int = 0
    use_mellowmax: bool = False
    mellowmax_alpha: float = 1.0
    early_stop: bool = True
    use_prefix_cache: bool = False
    allow_non_ascii: bool = False
    filter_ids: bool = True
    add_space_before_target: bool = False
    seed: int = None
    verbosity: str = "INFO"
    use_route_loss: bool=True
    selected_layer: int = 0
    selected_expert: List[int]= None

    def __post_init__(self):
        # 初始化时赋值为空列表，确保每个实例独立
        if self.selected_expert is None:
            self.selected_expert = []


@dataclass
class GCGResult:
    best_loss: float
    best_string: str
    losses: List[float]
    strings: List[str]

class AttackBuffer:
    def __init__(self, size: int):
        self.buffer = [] # elements are (loss: float, optim_ids: Tensor)
        self.size = size

    def add(self, loss: float, optim_ids: Tensor) -> None:
        if self.size == 0:
            self.buffer = [(loss, optim_ids)]
            return

        if len(self.buffer) < self.size:
            self.buffer.append((loss, optim_ids))
        else:
            self.buffer[-1] = (loss, optim_ids)

        self.buffer.sort(key=lambda x: x[0])

    def get_best_ids(self) -> Tensor:
        return self.buffer[0][1]

    def get_lowest_loss(self) -> float:
        return self.buffer[0][0]

    def get_highest_loss(self) -> float:
        return self.buffer[-1][0]

    def log_buffer(self, tokenizer):
        message = "buffer:"
        for loss, ids in self.buffer:
            optim_str = tokenizer.batch_decode(ids)[0]
            optim_str = optim_str.replace("\\", "\\\\")
            optim_str = optim_str.replace("\n", "\\n")
            message += f"\nloss: {loss}" + f" | string: {optim_str}"
        logger.info(message)

def sample_ids_from_grad(
    ids: Tensor,
    grad: Tensor,
    search_width: int,
    topk: int = 256,
    n_replace: int = 1,
    not_allowed_ids: Tensor = False,
):
    """Returns `search_width` combinations of token ids based on the token gradient.

    Args:
        ids : Tensor, shape = (n_optim_ids)
            the sequence of token ids that are being optimized
        grad : Tensor, shape = (n_optim_ids, vocab_size)
            the gradient of the GCG loss computed with respect to the one-hot token embeddings
        search_width : int
            the number of candidate sequences to return
        topk : int
            the topk to be used when sampling from the gradient
        n_replace: int
            the number of token positions to update per sequence
        not_allowed_ids: Tensor, shape = (n_ids)
            the token ids that should not be used in optimization

    Returns:
        sampled_ids : Tensor, shape = (search_width, n_optim_ids)
            sampled token ids
    """
    n_optim_tokens = len(ids)
    original_ids = ids.repeat(search_width, 1)

    if not_allowed_ids is not None:
        grad[:, not_allowed_ids.to(grad.device)] = float("inf")

    topk_ids = (-grad).topk(topk, dim=1).indices

    sampled_ids_pos = torch.argsort(torch.rand((search_width, n_optim_tokens), device=grad.device))[..., :n_replace]
    sampled_ids_val = torch.gather(
        topk_ids[sampled_ids_pos],
        2,
        torch.randint(0, topk, (search_width, n_replace, 1), device=grad.device)
    ).squeeze(2)

    new_ids = original_ids.scatter_(1, sampled_ids_pos, sampled_ids_val)

    return new_ids

def filter_ids(ids: Tensor, tokenizer: transformers.PreTrainedTokenizer):
    """Filters out sequeneces of token ids that change after retokenization.

    Args:
        ids : Tensor, shape = (search_width, n_optim_ids)
            token ids
        tokenizer : ~transformers.PreTrainedTokenizer
            the model's tokenizer

    Returns:
        filtered_ids : Tensor, shape = (new_search_width, n_optim_ids)
            all token ids that are the same after retokenization
    """
    ids_decoded = tokenizer.batch_decode(ids)
    filtered_ids = []

    for i in range(len(ids_decoded)):
        # Retokenize the decoded token ids
        ids_encoded = tokenizer(ids_decoded[i], return_tensors="pt", add_special_tokens=False).to(ids.device)["input_ids"][0]
        if torch.equal(ids[i], ids_encoded):
           filtered_ids.append(ids[i])

    if not filtered_ids:
        # This occurs in some cases, e.g. using the Llama-3 tokenizer with a bad initialization
        raise RuntimeError(
            "No token sequences are the same after decoding and re-encoding. "
            "Consider setting `filter_ids=False` or trying a different `optim_str_init`"
        )

    return torch.stack(filtered_ids)

class GCG:
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        config: GCGConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.selected_layer = self.config.selected_layer
        self.selected_expert = torch.tensor(self.config.selected_expert).to(model.device)

        self.embedding_layer = model.get_input_embeddings()
        self.not_allowed_ids = None if config.allow_non_ascii else get_nonascii_toks(tokenizer, device=model.device)
        self.prefix_cache = None

        self.stop_flag = False

        if model.dtype in (torch.float32, torch.float64):
            logger.warning(f"Model is in {model.dtype}. Use a lower precision data type, if possible, for much faster optimization.")

        if model.device == torch.device("cpu"):
            logger.warning("Model is on the CPU. Use a hardware accelerator for faster optimization.")

        if not tokenizer.chat_template:
            logger.warning("Tokenizer does not have a chat template. Assuming base model and setting chat template to empty.")
            tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"

    def run(
        self,
        messages: Union[str, List[dict]],
        target: str,
    ) -> GCGResult:
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        if config.seed is not None:
            set_seed(config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        else:
            messages = copy.deepcopy(messages)

        # Append the GCG string at the end of the prompt if location not specified
        if not any(["{optim_str}" in d["content"] for d in messages]):
            messages[-1]["content"] = messages[-1]["content"] + "{optim_str}"

        template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Remove the BOS token -- this will get added when tokenizing, if necessary
        if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
            template = template.replace(tokenizer.bos_token, "")
        before_str, after_str = template.split("{optim_str}")

        target = " " + target if config.add_space_before_target else target
        #truth = " " + target if config.add_space_before_target else truth

        # Tokenize everything that doesn't get optimized
        before_ids = tokenizer([before_str], padding=False, return_tensors="pt")["input_ids"].to(model.device).to(torch.int64)
        #before_ids = tokenizer([before_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device).to(torch.int64)
        after_ids = tokenizer([after_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device).to(torch.int64)
        target_ids = tokenizer([target], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device).to(torch.int64)
        #wqy
        #truth_ids = tokenizer([truth], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device).to(torch.int64)

        # Embed everything that doesn't get optimized
        embedding_layer = self.embedding_layer
        before_embeds, after_embeds, target_embeds = [embedding_layer(ids) for ids in (before_ids, after_ids, target_ids)]

        # Compute the KV Cache for tokens that appear before the optimized tokens
        if config.use_prefix_cache:
            with torch.no_grad():
                output = model(inputs_embeds=before_embeds, use_cache=True)
                self.prefix_cache = output.past_key_values

        self.target_ids = target_ids

        self.before_embeds = before_embeds
        self.after_embeds = after_embeds
        self.target_embeds = target_embeds

        # Initialize the attack buffer
        buffer = self.init_buffer()
        optim_ids = buffer.get_best_ids()

        losses = []
        optim_strings = []

        for _ in tqdm(range(config.num_steps)):
            # Compute the token gradient
            optim_ids_onehot_grad = self.compute_token_gradient(optim_ids)

            with torch.no_grad():

                # Sample candidate token sequences based on the token gradient
                sampled_ids = sample_ids_from_grad(
                    optim_ids.squeeze(0),
                    optim_ids_onehot_grad.squeeze(0),
                    config.search_width,
                    config.topk,
                    config.n_replace,
                    not_allowed_ids=self.not_allowed_ids,
                )

                if config.filter_ids:
                    sampled_ids = filter_ids(sampled_ids, tokenizer)

                new_search_width = sampled_ids.shape[0]

                # Compute loss on all candidate sequences
                batch_size = new_search_width if config.batch_size is None else config.batch_size
                if self.prefix_cache:
                    input_embeds = torch.cat([
                        embedding_layer(sampled_ids),
                        after_embeds.repeat(new_search_width, 1, 1),
                        target_embeds.repeat(new_search_width, 1, 1),
                    ], dim=1)
                else:
                    input_embeds = torch.cat([
                        before_embeds.repeat(new_search_width, 1, 1),
                        embedding_layer(sampled_ids),
                        after_embeds.repeat(new_search_width, 1, 1),
                        target_embeds.repeat(new_search_width, 1, 1),
                    ], dim=1)
                loss = find_executable_batch_size(self.compute_candidates_loss, batch_size)(input_embeds)

                current_loss = loss.min().item()
                optim_ids = sampled_ids[loss.argmin()].unsqueeze(0)

                # Update the buffer based on the loss
                losses.append(current_loss)
                if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                    buffer.add(current_loss, optim_ids)

            optim_ids = buffer.get_best_ids()
            optim_str = tokenizer.batch_decode(optim_ids)[0]
            optim_strings.append(optim_str)

            buffer.log_buffer(tokenizer)

            if self.stop_flag:
                logger.info("Early stopping due to finding a perfect match.")
                break

        min_loss_index = losses.index(min(losses))

        result = GCGResult(
            best_loss=losses[min_loss_index],
            best_string=optim_strings[min_loss_index],
            losses=losses,
            strings=optim_strings,
        )

        return result

    def init_buffer(self) -> AttackBuffer:
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        logger.info(f"Initializing attack buffer of size {config.buffer_size}...")

        # Create the attack buffer and initialize the buffer ids
        buffer = AttackBuffer(config.buffer_size)

        if isinstance(config.optim_str_init, str):
            init_optim_ids = tokenizer(config.optim_str_init, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
            if config.buffer_size > 1:
                init_buffer_ids = tokenizer(INIT_CHARS, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze().to(model.device)
                init_indices = torch.randint(0, init_buffer_ids.shape[0], (config.buffer_size - 1, init_optim_ids.shape[1]))
                init_buffer_ids = torch.cat([init_optim_ids, init_buffer_ids[init_indices]], dim=0)
            else:
                init_buffer_ids = init_optim_ids

        else: # assume list
            if (len(config.optim_str_init) != config.buffer_size):
                logger.warning(f"Using {len(config.optim_str_init)} initializations but buffer size is set to {config.buffer_size}")
            try:
                init_buffer_ids = tokenizer(config.optim_str_init, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
            except ValueError:
                logger.error("Unable to create buffer. Ensure that all initializations tokenize to the same length.")

        true_buffer_size = max(1, config.buffer_size)

        # Compute the loss on the initial buffer entries
        if self.prefix_cache:
            init_buffer_embeds = torch.cat([
                self.embedding_layer(init_buffer_ids),
                self.after_embeds.repeat(true_buffer_size, 1, 1),
                self.target_embeds.repeat(true_buffer_size, 1, 1),
            ], dim=1)
        else:
            init_buffer_embeds = torch.cat([
                self.before_embeds.repeat(true_buffer_size, 1, 1),
                self.embedding_layer(init_buffer_ids),
                self.after_embeds.repeat(true_buffer_size, 1, 1),
                self.target_embeds.repeat(true_buffer_size, 1, 1),
            ], dim=1)

        init_buffer_losses = find_executable_batch_size(self.compute_candidates_loss, true_buffer_size)(init_buffer_embeds)

        # Populate the buffer
        for i in range(true_buffer_size):
            buffer.add(init_buffer_losses[i], init_buffer_ids[[i]])

        buffer.log_buffer(tokenizer)

        logger.info("Initialized attack buffer.")

        return buffer

    def compute_token_gradient(
        self,
        optim_ids: Tensor,
    ) -> Tensor:
        """Computes the gradient of the GCG loss w.r.t the one-hot token matrix.

        Args:
        optim_ids : Tensor, shape = (1, n_optim_ids)
            the sequence of token ids that are being optimized
        """
        model = self.model
        embedding_layer = self.embedding_layer

        # Create the one-hot encoding matrix of our optimized token ids
        optim_ids_onehot = torch.nn.functional.one_hot(optim_ids, num_classes=embedding_layer.num_embeddings)
        optim_ids_onehot = optim_ids_onehot.to(dtype=model.dtype, device=model.device)
        optim_ids_onehot.requires_grad_()

        # (1, num_optim_tokens, vocab_size) @ (vocab_size, embed_dim) -> (1, num_optim_tokens, embed_dim)
        optim_embeds = optim_ids_onehot @ embedding_layer.weight

        if self.prefix_cache:
            input_embeds = torch.cat([optim_embeds, self.after_embeds, self.target_embeds], dim=1)
            output = model(inputs_embeds=input_embeds, past_key_values=self.prefix_cache, output_router_logits = True)
        else:
            input_embeds = torch.cat([self.before_embeds, optim_embeds, self.after_embeds, self.target_embeds], dim=1)
            output = model(inputs_embeds=input_embeds, output_router_logits = True)

        logits = output.logits

        # Shift logits so token n-1 predicts token n
        shift = input_embeds.shape[1] - self.target_ids.shape[1]
        shift_logits = logits[..., shift-1:-1, :].contiguous() # (1, num_target_ids, vocab_size)
        shift_labels = self.target_ids

        if self.config.use_mellowmax:
            label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
            loss = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1)

        elif self.config.use_route_loss:
            gate_pred = output.router_logits[self.selected_layer]
            gate_target = torch.zeros_like(gate_pred).to(self.model.device)
            gate_target[:, self.selected_expert] = 1
            #gate_pred = output.gate_importance[self.config.target_layer]
            #gate_pred = gate_pred[-1, :].unsqueeze(0)
            #gate_target = torch.tensor([1, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).repeat(gate_pred.shape[0], 1).to(self.model.device)
            probabilities = torch.softmax(gate_pred, dim=-1)
            loss = - torch.sum(gate_target * torch.log(probabilities), dim=-1).mean()

            #gate_target = torch.ones(gate_pred.shape[0], device=self.model.device, dtype=torch.int64) * self.config.target_expert
            #loss = torch.nn.functional.cross_entropy(gate_pred.view(-1, gate_pred.size(-1)), gate_target.view(-1))
            #gate_target = torch.full(gate_pred.shape, self.config.target_expert).to(self.model.device)
            #loss = torch.nn.functional.cross_entropy(gate_pred.view(-1, gate_pred.size(-1)), gate_target.view(-1))
            #loss = self.compute_route_loss(route_target_labels, self.config.target_expert).squeeze()
            #loss = torch.nn.functional.l1_loss(gate_pred, gate_target)
            #loss = max(loss, torch.tensor([1e-10]).to(model.device))
            #loss = loss * 0.1
        else:
            loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        #loss2 = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        optim_ids_onehot_grad = torch.autograd.grad(outputs=[loss], inputs=[optim_ids_onehot])[0]

        return optim_ids_onehot_grad

    def compute_route_loss(self, x, target_index, eps=1e-10):
        """
        Encourages a skewed distribution for each row of a 2D tensor,
        maximizing the value at target_index and minimizing others.

        Args:
        x: a 2D Tensor (shape: [batch_size, num_elements]).
        target_index: index of the component to maximize in each row.
        eps: small value for numerical stability.

        Returns:
        A 2D Tensor (same shape as x) where each row has its skewness loss.
        """
        if x.shape[-1] <= 1:
            return torch.zeros(x.shape[0], device=x.device)

        # Convert x to float for calculations
        x_float = x.float()

        normalized_x = (x_float - x_float.min()) / (x_float.max() - x_float.min() + eps)  # 归一化

        # Extract the target values from the specified index in each row
        target_values = normalized_x[:, target_index]

        # Extract the other values for each row (excluding the target index)
        other_values = torch.cat((normalized_x[:, :target_index], normalized_x[:, target_index+1:]), dim=1)

        '''# Calculate variance of the other components for each row
        variance = other_values.var(dim=1)

        # Calculate the skewness as a measure of unevenness for each row
        skewness = target_values / (other_values.mean(dim=1) + eps)

        # Combine both to encourage the desired distribution
        loss = variance - skewness'''
        other_sum = other_values.sum(dim=-1)

        loss = -target_values + other_sum + eps

        return loss

    def top2_ranking_loss(predictions, target, i, j):
        """
        Top-2 排序损失，确保第 i 和第 j 类别排在前两名
        predictions: 预测概率，形状为 (batch_size, num_classes)
        target: 实际标签，形状为 (batch_size,)
        i, j: 要排在前两名的类别索引
        """
        # 获取第 i 和第 j 类别的预测概率
        pred_i = predictions[:, i]
        pred_j = predictions[:, j]

        # 计算Top-2排序损失
        loss = -torch.log(pred_i) - torch.log(pred_j)

        # 确保其他类别的概率小于第 i 和第 j 类别
        predictions[:, i] = 0
        predictions[:, j] = 0

        loss += torch.sum(torch.log(1 - predictions), dim=-1)

        return loss.mean()



    def compute_candidates_loss(
        self,
        search_batch_size: int,
        input_embeds: Tensor,
    ) -> Tensor:
        """Computes the GCG loss on all candidate token id sequences.

        Args:
            search_batch_size : int
                the number of candidate sequences to evaluate in a given batch
            input_embeds : Tensor, shape = (search_width, seq_len, embd_dim)
                the embeddings of the `search_width` candidate sequences to evaluate
        """
        all_loss = []
        prefix_cache_batch = []

        for i in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                input_embeds_batch = input_embeds[i:i+search_batch_size]
                current_batch_size = input_embeds_batch.shape[0]

                if self.prefix_cache:
                    if not prefix_cache_batch or current_batch_size != search_batch_size:
                        prefix_cache_batch = [[x.expand(current_batch_size, -1, -1, -1) for x in self.prefix_cache[i]] for i in range(len(self.prefix_cache))]

                    outputs = self.model(inputs_embeds=input_embeds_batch, past_key_values=prefix_cache_batch, output_router_logits=True)
                else:
                    outputs = self.model(inputs_embeds=input_embeds_batch, output_router_logits=True)

                logits = outputs.logits

                tmp = input_embeds.shape[1] - self.target_ids.shape[1]
                shift_logits = logits[..., tmp-1:-1, :].contiguous()
                shift_labels = self.target_ids.repeat(current_batch_size, 1)
                #shift_truth = self.truth_ids.repeat(current_batch_size, 1)
                #shift_truth = shift_truth[:,:shift_labels.shape[1]]


                if self.config.use_mellowmax:
                    label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
                    loss = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1)

                elif self.config.use_route_loss:
                    gate_pred = outputs.router_logits[self.config.selected_layer]
                    gate_target = torch.zeros_like(gate_pred).to(self.model.device)
                    gate_target[:, self.selected_expert] = 1
                    #gate_pred = outputs.gate_importance[self.config.target_layer]
                    #gate_target = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0]).to(self.model.device)
                    probabilities = torch.softmax(gate_pred, dim=-1)
                    loss = - torch.sum(gate_target * torch.log(probabilities), dim=-1)

                    #gate_target = torch.ones(gate_pred.shape[0], device=self.model.device, dtype=torch.int64) * self.config.target_expert
                    #gate_target[:, self.config.target_expert] = self.target_expert
                    #loss = torch.nn.functional.l1_loss(gate_pred, gate_target, reduction="none")
                    #loss = torch.nn.functional.cross_entropy(gate_pred.view(-1, gate_pred.size(-1)), gate_target.view(-1), reduction="none")
                    #loss = self.compute_route_loss(route_target_labels, self.config.target_expert)
                    #loss = loss * 0.1

                loss = loss.view(current_batch_size, -1).mean(dim=-1)
                #loss = max(loss, torch.full_like(loss, 1e-10).to(self.model.device))
                all_loss.append(loss)

                if self.config.early_stop:
                    if not self.config.use_route_loss and torch.any(torch.all(torch.argmax(shift_logits, dim=-1) == shift_labels, dim=-1)).item():
                        self.stop_flag = True
                    if self.config.use_route_loss and torch.any(torch.all(torch.topk(gate_pred, 2)[-1] == torch.tensor(self.config.selected_expert).to(self.model.device))).item():
                        self.stop_flag = True
                    '''
                    if self.config.use_route_loss:
                        import pdb
                        pdb.set_trace()
                        topk_indices = torch.topk(outputs.gate_importance[self.config.target_layer], 1).indices
                        if torch.any(self.config.target_expert == topk_indices).item():
                            self.stop_flag = True'''

                del outputs
                gc.collect()
                torch.cuda.empty_cache()

        return torch.cat(all_loss, dim=0)

# A wrapper around the GCG `run` method that provides a simple API
def run(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    messages: Union[str, List[dict]],
    target: str,
    config: Optional[GCGConfig] = None,
) -> GCGResult:
    """Generates a single optimized string using GCG.

    Args:
        model: The model to use for optimization.
        tokenizer: The model's tokenizer.
        messages: The conversation to use for optimization.
        target: The target generation.
        config: The GCG configuration to use.

    Returns:
        A GCGResult object that contains losses and the optimized strings.
    """
    if config is None:
        config = GCGConfig()

    logger.setLevel(getattr(logging, config.verbosity))

    gcg = GCG(model, tokenizer, config)
    result = gcg.run(messages, target)
    return result
