import numpy as np
from dataclasses import dataclass, field
from transformers import AdamW, get_scheduler
import os
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import pathlib
from typing import Dict, Optional
import math
from torch.utils.data import Dataset
import torch
import transformers
from transformers.trainer_pt_utils import LabelSmoother
from transformers import AutoTokenizer, AutoModelForCausalLM
from conversation import get_conv_template

#from trl import DPOTrainer
from datasets import load_dataset
from functools import partial
from tqdm import tqdm
from peft import LoraConfig
IGNORE_TOKEN_ID = LabelSmoother.ignore_index
from accelerate import Accelerator
from transformers import Adafactor, AutoTokenizer, HfArgumentParser, pipeline
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed, create_reference_model
from trl.core import LengthSampler
from typing import Dict, List, Tuple
import time
from accelerate.logging import get_logger
from trl.core import (
    WANDB_PADDING,
    PPODecorators,
    clip_by_value,
    convert_to_scalar,
    entropy_from_logits,
    flatten_dict,
    logprobs_from_logits,
    masked_mean,
    masked_var,
    masked_whiten,
    set_seed,
    stack_dicts,
    stats_to_np,
)
import logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)
tqdm.pandas()
good_token = '+'
bad_token = '-'
step_tag = 'ки'
#class MathPPO_Trainer

def find_double_13(lst, line_id):
    result = []
    for i in range(len(lst)-1):
        if lst[i] == line_id and lst[i+1] == line_id:
            result.append(i)
    return result


def find_13_indices(lst, line_id):
    indices = []
    for i, value in enumerate(lst):
        if value == line_id:
            indices.append(i)
    return indices

def is_sublist(sub_list, my_list):
    sub_len = len(sub_list)
    for i in range(len(my_list) - sub_len + 1):
        if my_list[i:i+sub_len] == sub_list:
            return True
    return False
def invsqrt_scheduler(warmup_steps):

    def _invsqrt_lr(step):
        return math.sqrt(warmup_steps) / math.sqrt(max(warmup_steps, step))

    def _warmup_lr(step):
        return max(step / warmup_steps, 0.1)

    def _invsqrt_lr_with_warmup(step):
        return max(_warmup_lr(step) if step < warmup_steps else _invsqrt_lr(step), 1e-8)

    return _invsqrt_lr_with_warmup
def compute_index(value_list, line_id):
    index_list = []
    if value_list[0] == line_id and value_list[1] == line_id:
        #if [13, 13] in value_list[2:]:
        if is_sublist([line_id, line_id], value_list[2:]):
            index_list =  find_double_13(value_list[2:], line_id)
        else:
            index_list = find_13_indices(value_list[2:], line_id)
        index_list = [i + 2 for i in index_list]
        
    else:
        if is_sublist([line_id, line_id], value_list[1:]):
            index_list = find_double_13(value_list[1:], line_id)
            
        else:
            index_list = find_13_indices(value_list[1:], line_id)
            
        index_list = [i + 1 for i in index_list]
    index_list.append(len(value_list)-1)
    return index_list


class PRM_Trainer(PPOTrainer):
    @PPODecorators.empty_device_cache()
    def step(
        self,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
        response_masks: Optional[List[torch.LongTensor]] = None,
        step_scores: List[torch.LongTensor] = None,
        step_index: List[torch.LongTensor] = None,

        
    ):
        """
        Run a PPO optimisation step given a list of queries, model responses, and rewards.

        Args:
            queries (List[`torch.LongTensor`]):
                List of tensors containing the encoded queries of shape (`query_length`)
            responses (List[`torch.LongTensor`]):
                List of tensors containing the encoded responses of shape (`response_length`)
            scores (List[`torch.FloatTensor`]):
                List of tensors containing the scores.
            response_masks (List[`torch.FloatTensor`], *optional*)):
                List of tensors containing masks of the response tokens.

        Returns:
            `dict[str, Any]`: A summary of the training statistics
        """
        bs = self.config.batch_size

        queries, responses, scores, response_masks = self._step_safety_checker(
            bs, queries, responses, scores, response_masks
        )
        scores = torch.tensor(scores, device=self.current_device)
        if self.config.use_score_scaling:
            # Score scaling
            scores_mean, scores_std = self.running.update(scores)
            tensor_to_kwargs = dict(dtype=scores.dtype, device=scores.device)
            score_scaling_factor = self.running.std.to(**tensor_to_kwargs) + torch.finfo(scores.dtype).eps
            if self.config.use_score_norm:
                scores = (scores - self.running.mean.to(**tensor_to_kwargs)) / score_scaling_factor
            else:
                scores /= score_scaling_factor

        if self.config.score_clip is not None:
            # Score clipping
            scores_dtype = scores.dtype
            scores = torch.clip(scores.float(), -self.config.score_clip, self.config.score_clip).to(dtype=scores_dtype)

        # if we want to push best model to the hub
        if hasattr(self, "highest_reward"):
            if self.compare_step % self.config.compare_steps == 0:
                curr_mean_reward = scores.mean()
                # if the best reward ever seen
                if curr_mean_reward > self.highest_reward:
                    self.highest_reward = curr_mean_reward
                    # push model to hub
                    self.push_to_hub(**self.push_to_hub_kwargs)
            self.compare_step += 1

        timing = dict()
        t0 = time.time()

        t = time.time()

        model_inputs = self.prepare_model_inputs(queries, responses)

        if self.is_distributed:
            pad_first = self.tokenizer.padding_side == "left"

            model_inputs["input_ids"] = self.accelerator.pad_across_processes(
                model_inputs["input_ids"],
                dim=1,
                pad_index=self.tokenizer.pad_token_id,
                pad_first=pad_first,
            )
            model_inputs["attention_mask"] = self.accelerator.pad_across_processes(
                model_inputs["attention_mask"], dim=1, pad_index=0, pad_first=pad_first
            )
            if self.is_encoder_decoder:
                model_inputs["decoder_input_ids"] = self.accelerator.pad_across_processes(
                    model_inputs["decoder_input_ids"],
                    dim=1,
                    pad_index=self.tokenizer.pad_token_id,
                    pad_first=pad_first,
                )
                model_inputs["decoder_attention_mask"] = self.accelerator.pad_across_processes(
                    model_inputs["decoder_attention_mask"],
                    dim=1,
                    pad_index=0,
                    pad_first=pad_first,
                )

        model_inputs_names = list(model_inputs.keys())

        full_kl_penalty = self.config.kl_penalty == "full"

        with torch.no_grad():
            all_logprobs, logits_or_none, values, masks = self.batched_forward_pass(
                self.model,
                queries,
                responses,
                model_inputs,
                response_masks=response_masks,
                return_logits=full_kl_penalty,
            )
            with self.optional_peft_ctx():
                ref_logprobs, ref_logits_or_none, _, _ = self.batched_forward_pass(
                    self.model if self.is_peft_model else self.ref_model,
                    queries,
                    responses,
                    model_inputs,
                    return_logits=full_kl_penalty,
                )

        timing["time/ppo/forward_pass"] = time.time() - t

        with torch.no_grad():
            t = time.time()
            if full_kl_penalty:
                active_full_logprobs = logprobs_from_logits(logits_or_none, None, gather=False)
                ref_full_logprobs = logprobs_from_logits(ref_logits_or_none, None, gather=False)

                rewards, non_score_reward, kls = self.compute_rewards(
                    scores, active_full_logprobs, ref_full_logprobs, masks, step_scores, step_index
                )
            else:
                rewards, non_score_reward, kls = self.compute_rewards(scores, all_logprobs, ref_logprobs, masks, step_scores, step_index)
            timing["time/ppo/compute_rewards"] = time.time() - t

            t = time.time()
            values, advantages, returns = self.compute_advantages(values, rewards, masks)
            timing["time/ppo/compute_advantages"] = time.time() - t

        # upcast to float32 to avoid dataset issues
        batch_dict = {
            "queries": queries,
            "responses": responses,
            "logprobs": all_logprobs.to(torch.float32),
            "values": values.to(torch.float32),
            "masks": masks,
            "advantages": advantages,
            "returns": returns,
        }
        batch_dict.update(model_inputs)

        t = time.time()
        all_stats = []
        early_stop = False
        for _ in range(self.config.ppo_epochs):
            if early_stop:
                break
            b_inds = np.random.permutation(bs)
            for backward_batch_start in range(0, bs, self.config.backward_batch_size):
                backward_batch_end = backward_batch_start + self.config.backward_batch_size
                backward_batch_inds = b_inds[backward_batch_start:backward_batch_end]

                for mini_batch_start in range(0, self.config.backward_batch_size, self.config.mini_batch_size):
                    mini_batch_end = mini_batch_start + self.config.mini_batch_size
                    mini_batch_inds = backward_batch_inds[mini_batch_start:mini_batch_end]
                    mini_batch_dict = {
                        "logprobs": batch_dict["logprobs"][mini_batch_inds],
                        "values": batch_dict["values"][mini_batch_inds],
                        "masks": batch_dict["masks"][mini_batch_inds],
                        # hacks: the queries and responses are ragged.
                        "queries": [batch_dict["queries"][i] for i in mini_batch_inds],
                        "responses": [batch_dict["responses"][i] for i in mini_batch_inds],
                        "advantages": batch_dict["advantages"][mini_batch_inds],
                        "returns": batch_dict["returns"][mini_batch_inds],
                    }
                    for k in model_inputs_names:
                        mini_batch_dict[k] = batch_dict[k][mini_batch_inds]
                    with self.accelerator.accumulate(self.model):
                        model_inputs = {k: mini_batch_dict[k] for k in model_inputs_names}

                        logprobs, logits, vpreds, _ = self.batched_forward_pass(
                            self.model,
                            mini_batch_dict["queries"],
                            mini_batch_dict["responses"],
                            model_inputs,
                            return_logits=True,
                        )
                        train_stats = self.train_minibatch(
                            mini_batch_dict["logprobs"],
                            mini_batch_dict["values"],
                            logprobs,
                            logits,
                            vpreds,
                            mini_batch_dict["masks"],
                            mini_batch_dict["advantages"],
                            mini_batch_dict["returns"],
                        )
                        all_stats.append(train_stats)

            # typically, early stopping is done at the epoch level
            if self.config.early_stopping:
                policykl = train_stats["policy/policykl"]
                early_stop = self._early_stop(policykl)
                if early_stop:
                    break

        timing["time/ppo/optimize_step"] = time.time() - t

        t = time.time()
        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        train_stats["policy/advantages"] = torch.flatten(train_stats["policy/advantages"]).unsqueeze(0)
        train_stats["policy/advantages"] = torch.nan_to_num(train_stats["policy/advantages"], WANDB_PADDING)
        train_stats["policy/ratio"] = torch.flatten(train_stats["policy/ratio"]).unsqueeze(0)

        stats = self.record_step_stats(
            scores=scores,
            logprobs=all_logprobs,
            ref_logprobs=ref_logprobs,
            non_score_reward=non_score_reward,
            train_stats=train_stats,
            kl_coef=self.kl_ctl.value,
            masks=masks,
            queries=queries,
            responses=responses,
            kls=kls,
        )
        # Gather/Reduce stats from all processes
        if self.is_distributed:
            stats = self.gather_stats(stats)
        stats = stats_to_np(stats)
        timing["time/ppo/calc_stats"] = time.time() - t
        stats["ppo/learning_rate"] = self.optimizer.param_groups[0]["lr"]

        # Update the KL control - multiply the batch_size by the number of processes
        self.kl_ctl.update(
            stats["objective/kl"],
            self.config.batch_size * self.accelerator.num_processes,
        )

        # Log the total ppo time
        timing["time/ppo/total"] = time.time() - t0
        stats.update(timing)

        # post-process stats for tensorboard and other loggers
        if self.config.log_with != "wandb":
            stats = convert_to_scalar(stats)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return stats
    def compute_rewards(
        self,
        scores: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        ref_logprobs: torch.FloatTensor,
        masks: torch.LongTensor,
        step_scores,
        step_index,
    ):
        """
        Compute per token rewards from scores and KL-penalty.

        Args:
            scores (`torch.FloatTensor`):
                Scores from the reward model, shape (`batch_size`)
            logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape (`batch_size`, `response_length`)
            ref_logprobs (`torch.FloatTensor`):
                Log probabilities of the reference model, shape (`batch_size`, `response_length`)

        Returns:
            `torch.FloatTensor`: Per token rewards, shape (`batch_size`, `response_length`)
            `torch.FloatTensor`: Non score rewards, shape (`batch_size`, `response_length`)
            `torch.FloatTensor`: KL penalty, shape (`batch_size`, `response_length`)
        """

        logger.info("compute logprobs", main_process_only=True)
        logger.info(logprobs[:5], main_process_only=True)
        #logger.info(logprobs.shape, main_process_only=True)

        logger.info("compute ref_logprobs", main_process_only=True)
        logger.info(ref_logprobs[:5], main_process_only=True)
        # logger.info(ref_logprobs.shape, main_process_only=True)



        rewards, non_score_rewards, kls = [], [], []
        for score, logprob, ref_logprob, mask, step_score, step_i in zip(scores, logprobs, ref_logprobs, masks, step_scores, step_index):
            # compute KL penalty (from difference in logprobs)
            kl = self._kl_penalty(logprob, ref_logprob)
            kls.append(kl)
            non_score_reward = -self.kl_ctl.value * kl
            non_score_rewards.append(non_score_reward)
            reward = non_score_reward.clone()
            last_non_masked_index = mask.nonzero()[-1]

            
            step_reward_index = mask.nonzero()[step_i].squeeze(1)
            


            if step_reward_index.shape == step_score.shape:

            
                reward[step_reward_index] += step_score / step_score.shape[0]
                reward[last_non_masked_index] += score


            else:
                reward[last_non_masked_index] += score

                logger.info(step_reward_index)
                logger.info(step_score)
            rewards.append(reward)
        return torch.stack(rewards), torch.stack(non_score_rewards), torch.stack(kls)    


class AverageMeter:
    r"""
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
@dataclass
class DataArguments:
    data_id: str = field(
        default = None, metadata = {"help": "Dataset id name of the training data."}
    )
    
    data_split: str = field(
        default = None, metadata = {"help": "Chosen split of the training data."}
    )
    
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    
    cache_path: str = field(
        default=None, metadata={"help": "Path to cache the training data."}
    )
    
    num_proc: int = field(
        default=32
    )
    
    conv_template: str = field(default = "vicuna-1.1")
    
    json_path: str = field(
        default = None, metadata = {"help": "Path to the json file containing the training data."}
    )

@dataclass
class ScriptArguments(transformers.TrainingArguments):
    """
    The name of the Casual LM model we wish to fine-tune with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="/blob/v-weihaozeng/saved/math_scaling/math_metamathqa_395K/checkpoint-9630/", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(default="/blob/v-weihaozeng/saved/math_scaling/math_metamathqa_395K/checkpoint-9630/", metadata={"help": "the tokenizer name"})
    reward_model_name: Optional[str] = field(default="peiyi9979/math-shepherd-mistral-7b-prm", metadata={"help": "the reward model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    logging_dir: Optional[str] = field(default=None, metadata={"help": "logging dir"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    output_max_length: Optional[int] = field(default=128, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(default=False, metadata={"help": "whether to use the batched text gen"})
    save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="runs/", metadata={"help": "n steps to save the model"})
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})
    steps: Optional[int] = field(default=20000, metadata={"help": "number of epochs"})
    num_epochs : Optional[int] = field(default=1, metadata={"help": "lora r"})
    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )
    warmup_steps: Optional[int] = field(
        default=100, metadata={"help": "the number of gradient accumulation steps"}
    )
    share_layers: Optional[int] = field(
        default=0, metadata={"help": "share layers"}
    )
    adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})
    
    kl_target: Optional[float] = field(
        default=6.0,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )
local_rank = None



# def process_resp(resp):
#     if "\n\n" in resp:
#         temp_gpt = resp.split("\n\n")
#     else:
#         temp_gpt = resp.split("\n")

#     while len(temp_gpt) <= 3:
#         temp_gpt.append(" ")
#     temp_output = ""
#     for index, i in enumerate(temp_gpt[:-2]):
#         temp_output = temp_output + "Step " + str(index + 1) + ": " + i + " " + step_tag + "\n"
#     temp_output = temp_output + "Step " + str(index + 2) + ": " + temp_gpt[-2] + " " + temp_gpt[-1] + " " + step_tag

#     return temp_output

def process_resp(resp):

    resp = resp.strip("\n")

    resp = resp.strip("\n")
    if "\n\n" in resp:
        temp_gpt = resp.split("\n\n")
    else:
        temp_gpt = resp.split("\n")
    temp_output = ""

    if len(temp_gpt) > 1:
        for index, i in enumerate(temp_gpt[:-1]):
            temp_output = temp_output + "Step " + str(index + 1) + ": " + i + " " + step_tag + "\n"
        temp_output = temp_output + "Step " + str(index + 2) + ": " + temp_gpt[-1] + " " + step_tag

    else:
        temp_output = temp_output + "Step " + str(1) + ": " + temp_gpt[-1] + " " + step_tag


    return temp_output



    



# def reward_score(model, tokenizer, question, output, candidate_tokens, step_tag_id):
#     input_for_prm = f"{question} {output}"
#     input_id = tokenizer.encode(input_for_prm)
#     with torch.no_grad():
#         logits = model(input_id).logits[:, :, candidate_tokens]
#         scores = logits.softmax(dim=-1)[:, :, 0]
#         step_scores = scores[input_id == step_tag_id]

#     return min(step_scores.to_list())

# def batch_reward_score(model, tokenizer, questions, outputs, candidate_tokens, step_tag_id):
    
#     model.eval()
#     device = next(model.parameters()).device
    
#     # 合并问题和输出以进行编码
#     inputs = [f"{question} {output}" for question, output in zip(questions, outputs)]
#     # 使用tokenizer一次性编码整个batch，并设置pad_token_id为-100
#     batch_input_ids = tokenizer(inputs, padding=True, return_tensors="pt").input_ids.to(device)
    

#     with torch.cuda.amp.autocast(dtype=torch.bfloat16):

#         logits = model(batch_input_ids).logits[:, :, candidate_tokens]
#         #logits = logits[:, :, candidate_tokens]
#         scores = logits.softmax(dim=-1)[:, :, 0]

#         step_scores = [scores[i, input_ids == step_tag_id] for i, input_ids in enumerate(batch_input_ids)]
#         #print("-----", step_scores)
#         del batch_input_ids, logits, scores

#     return [min(step_score.tolist()) for step_score in step_scores]

# import torch

def batch_reward_score(model, tokenizer, questions, outputs, candidate_tokens, step_tag_id):
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        inputs = [f"{question} {output}" for question, output in zip(questions, outputs)]
        batch_input_ids = tokenizer(inputs, padding=True, return_tensors="pt").input_ids.to(device)
        

        #logger.info("Responses IDS", main_process_only=True)
        #for resp in batch_input_ids:
        #    logger.info(resp, main_process_only=True)
        #    logger.info(resp.shape, main_process_only=True)


        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = model(batch_input_ids).logits[:, :, candidate_tokens]
            step_scores = [logits[i, input_ids == step_tag_id].softmax(dim=-1)[:, 0] for i, input_ids in enumerate(batch_input_ids)]

    del batch_input_ids, logits
    return [min(step_score.tolist()) for step_score in step_scores],step_scores

def reward_score(model, tokenizer, questions, outputs, candidate_tokens, step_tag_id):
    step_scores = []
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for question, output in zip(questions, outputs):
            input_str = f"{question} {output}"
            input_id = torch.tensor([tokenizer.encode(input_str)], device=device)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(input_id).logits[:,:,candidate_tokens]
                scores = logits.softmax(dim=-1)[:,:,0] 
                step_score = scores[input_id == step_tag_id]
                
                step_scores.append(step_score)
                del input_id, logits
                
    return [step_score.tolist()[-1] for step_score in step_scores], step_scores

def reward_score_pen(model, tokenizer, questions, outputs, candidate_tokens, step_tag_id):
    step_scores = []
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for question, output in zip(questions, outputs):

            
            # 检查output中是否包含特定字符串
            if "The answer is" not in output:
                # 如果不包含，则此步骤的reward为0
                step_scores.append(torch.tensor([0.0], device=device))
                continue  # 直接跳到下一个iteration

            input_str = f"{question} {output}"
            input_id = torch.tensor([tokenizer.encode(input_str)], device=device)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(input_id).logits[:,:,candidate_tokens]
                scores = logits.softmax(dim=-1)[:,:,0] 
                step_score = scores[input_id == step_tag_id]
                
                step_scores.append(step_score)
                del input_id, logits
                
    # 修改了这里的返回逻辑，以确保在output中不包含"The answer is"时，reward为0
    # 注意，这里假设torch.tensor([0.0], device=device)能够适当地转换成list[-1]格式以适应原有逻辑
    return [step_score.tolist()[-1] if len(step_score) > 0 else 0 for step_score in step_scores], step_scores
                
                
def reward_score_pen_v2(model, tokenizer, questions, outputs, candidate_tokens, step_tag_id):
    step_scores = []
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for question, output in zip(questions, outputs):

            
            # 检查output中是否包含特定字符串
            temp_text = output.split('The answer is: ')

            if len(temp_text) <= 1:
            #if "The answer is" not in output:
                # 如果不包含，则此步骤的reward为0
                step_scores.append(torch.tensor([0.0], device=device))
                continue  # 直接跳到下一个iteration

            input_str = f"{question} {output}"
            input_id = torch.tensor([tokenizer.encode(input_str)], device=device)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(input_id).logits[:,:,candidate_tokens]
                scores = logits.softmax(dim=-1)[:,:,0] 
                step_score = scores[input_id == step_tag_id]
                
                step_scores.append(step_score)
                del input_id, logits
                
    # 修改了这里的返回逻辑，以确保在output中不包含"The answer is"时，reward为0
    # 注意，这里假设torch.tensor([0.0], device=device)能够适当地转换成list[-1]格式以适应原有逻辑
    return [step_score.tolist()[-1] if len(step_score) > 0 else 0 for step_score in step_scores], step_scores
                
                    
def reward_score_pen_v3(model, tokenizer, questions, outputs, candidate_tokens, step_tag_id):
    step_scores = []
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for question, output in zip(questions, outputs):

            
            # 检查output中是否包含特定字符串
            temp_text = output.split('The answer is: ')

            if len(temp_text) <= 1:
            #if "The answer is" not in output:
                # 如果不包含，则此步骤的reward为0
                step_scores.append(torch.tensor([-1.0], device=device))
                continue  # 直接跳到下一个iteration

            input_str = f"{question} {output}"
            input_id = torch.tensor([tokenizer.encode(input_str)], device=device)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(input_id).logits[:,:,candidate_tokens]
                scores = logits.softmax(dim=-1)[:,:,0] 
                step_score = scores[input_id == step_tag_id]
                
                step_scores.append(step_score)
                del input_id, logits
                
    # 修改了这里的返回逻辑，以确保在output中不包含"The answer is"时，reward为0
    # 注意，这里假设torch.tensor([0.0], device=device)能够适当地转换成list[-1]格式以适应原有逻辑
    return [step_score.tolist()[-1] if len(step_score) > 0 else 0 for step_score in step_scores], step_scores  
        
    
       

# def batch_reward_score(model, tokenizer, questions, outputs, candidate_tokens, step_tag_id):
#     model.eval()
#     device = next(model.parameters()).device

#     with torch.no_grad():
#         inputs = [f"{question} {output}" for question, output in zip(questions, outputs)]
#         batch_input_ids = tokenizer(inputs, padding=True, return_tensors="pt").input_ids.to(device)

#         with torch.cuda.amp.autocast(dtype=torch.bfloat16):
#             logits = model(batch_input_ids).logits[:, :, candidate_tokens].detach()
#             step_scores = [logits[i, input_ids == step_tag_id].softmax(dim=-1)[:, 0] for i, input_ids in enumerate(batch_input_ids)]

#     del batch_input_ids, logits
#     torch.cuda.empty_cache()
#     return [min(step_score.tolist()) for step_score in step_scores]


# def batch_reward_score(model, tokenizer, questions, outputs, candidate_tokens, step_tag_id):
#     model.eval()
#     device = next(model.parameters()).device
#     step_scores = []

#     with torch.no_grad():
#         for question, output in zip(questions, outputs):
#             input_str = f"{question} {output}"
#             input_ids = tokenizer(input_str, return_tensors="pt").input_ids.to(device)

#             with torch.cuda.amp.autocast(dtype=torch.bfloat16):
#                 logits = model(input_ids).logits[:, :, candidate_tokens].detach()
#                 step_score = logits[0, input_ids[0] == step_tag_id].softmax(dim=-1)[0]

#             step_scores.append(step_score.tolist())

#             del input_ids, logits
#             torch.cuda.empty_cache()

#     return [min(score) for score in step_scores]


def preprocess(
    sample,
    tokenizer,
    conv_template = "vicuna-1.1",
) -> Dict:

    conv = get_conv_template(conv_template)

    #print("=================", conv)
    prompt_template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    prompt = prompt_template.format(instruction=sample["conversations"][0]["value"])
    #prompt = conv.system + conv.sep + sample["conversations"][0]["from"] + ": " + sample["conversations"][0]["value"] + conv.sep
    
    # Apply prompt templates
    #chosen_sources = sample["chosen"]
    #chosen_conversations = chosen_sources[1]["role"] + ": " + chosen_sources[1]["content"] + conv.sep2

    #rejected_sources = sample["rejected"]
    #rejected_conversations = rejected_sources[1]["role"] + ": " + rejected_sources[1]["content"] + conv.sep2
    tokenized_question = tokenizer(prompt, truncation=True)
    return dict(
        query=prompt,
        input_ids=tokenized_question["input_ids"],
        question=sample["conversations"][0]["value"],
    )

def make_ppo_dataset(
    tokenizer,
    data_args: DataArguments,
    sanity_check: bool = False
) -> Dataset:
    """Load the stack-exchange-paired dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts are structured as follows:
      "Question: " + <prompt> + "\n\nAnswer: "
    """
    
    data_id: str = data_args.data_id
    data_split: str = data_args.data_split
    data_dir: str = data_args.data_path
    cache_dir: str = data_args.cache_path
    num_proc: int = data_args.num_proc
    conv_template: str =  data_args.conv_template
    
    json_path: str = data_args.json_path
    
    if not json_path:
        dataset = load_dataset(
            data_id,
            split=data_split,
            cache_dir=cache_dir,
            data_dir=data_dir,
        )
    else:
        dataset = load_dataset(
            "json",
            data_files = json_path,
            split = data_split
        )
        
    original_columns = dataset.column_names
    
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    preprocess_with_template = partial(preprocess, tokenizer=tokenizer, conv_template = conv_template)
    ds = dataset.map(
        preprocess_with_template,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    ds.set_format(type="torch")


    return ds
def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}
def train():
    #global local_rank
    #parser = HfArgumentParser(ScriptArguments)

    parser = transformers.HfArgumentParser(
        (DataArguments, ScriptArguments)
    )
    data_args, script_args = parser.parse_args_into_dataclasses()
    reward_model_name = script_args.reward_model_name
    #local_rank = script_args.local_rank
    #print(f"cuda max memory allocated -4: {torch.cuda.max_memory_allocated() // 1024 / 1024 / 1024} GB")


    config = PPOConfig(
        #steps=script_args.steps,
        model_name=script_args.model_name,
        learning_rate=script_args.learning_rate,
        log_with=script_args.log_with,
        #project_kwargs={"logging_dir":script_args.logging_dir},
        #logging_dir=script_args.logging_dir,
        batch_size=script_args.batch_size,
        mini_batch_size=script_args.mini_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        optimize_cuda_cache=True,
        early_stopping=script_args.early_stopping,
        target_kl=script_args.target_kl,
        ppo_epochs=script_args.ppo_epochs,
        seed=script_args.seed,
        init_kl_coef=script_args.init_kl_coef,
        adap_kl_ctrl=script_args.adap_kl_ctrl,
        target=script_args.kl_target,
        gamma=1.0,
        lam=0.95,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.tokenizer_name
        )
    #tokenizer.pad_token = tokenizer.eos_token
    # set seed before initializing value head for deterministic eval
    set_seed(config.seed)
    
    accelerator = Accelerator()
    current_device = accelerator.device


    #logger.info("My log", main_process_only=True)

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name,
        use_flash_attention_2 = True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False


    model_ref = create_reference_model(model, num_shared_layers=script_args.share_layers)

    #print(f"cuda max memory allocated -2: {torch.cuda.max_memory_allocated() // 1024 / 1024 / 1024} GB")

    train_dataset = make_ppo_dataset(tokenizer=tokenizer, data_args=data_args)
    total_train_batch_size = script_args.batch_size * script_args.world_size
    import math
    num_training_steps = math.ceil(len(train_dataset) / total_train_batch_size)


    #print("---------------num_training_steps--------------", num_training_steps)
    #logger.info("num_training_steps", main_process_only=True)
    #logger.info(num_training_steps, main_process_only=True)
    
    logger.info("num_training_steps: %d" % num_training_steps, main_process_only=True)
    #if accelerator.is_local_main_process:
    logger.info("************************** Running training ***************************", main_process_only=True)
    logger.info("  Num examples = {}".format(len(train_dataset)), main_process_only=True)
        
    logger.info("  Instantaneous batch size per device = {}".format(math.ceil(script_args.batch_size/script_args.gradient_accumulation_steps)), main_process_only=True)
    logger.info(
                "  Total train batch size (w. parallel, buffer, distributed & accumulation) = {}".format(
                    total_train_batch_size
                ),
            main_process_only=True
            )
    logger.info("  Gradient Accumulation steps = {}".format(script_args.gradient_accumulation_steps), main_process_only=True)
        
    logger.info("  Num optimization epochs per batch = {}".format(script_args.ppo_epochs), main_process_only=True)
        
    logger.info("  Total training steps = {}".format(num_training_steps), main_process_only=True)
        

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=script_args.learning_rate, eps=1e-6, betas=(0.9, 0.95))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=invsqrt_scheduler(script_args.warmup_steps))
    
    # optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    
    # lr_scheduler = get_scheduler(
    #     script_args.lr_scheduler_type,
    #     optimizer=optimizer,
    #     num_warmup_steps=script_args.get_warmup_steps(num_training_steps),
    #     num_training_steps=num_training_steps,
    # )

    #actor_params = get_optimizer_parameters(actor, training_args)

    if script_args.adafactor:
        optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=config.learning_rate,
        )

    # ppo_trainer = PPOTrainer(
    #     config,
    #     model,
    #     ref_model=model_ref,
    #     tokenizer=tokenizer,
    #     dataset=train_dataset,
    #     data_collator=collator,
    #     optimizer=optimizer,
    #     lr_scheduler=lr_scheduler,
    #     )

    ppo_trainer = PRM_Trainer(
        config,
        model,
        ref_model=model_ref,
        tokenizer=tokenizer,
        dataset=train_dataset,
        data_collator=collator,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        )
    
    device = ppo_trainer.accelerator.device

    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug
    ds_plugin = ppo_trainer.accelerator.state.deepspeed_plugin
    
    
    if ds_plugin is not None and ds_plugin.is_zero3_init_enabled():
        with ds_plugin.zero3_init_context_manager(enable=False):
            
            reward_model = AutoModelForCausalLM.from_pretrained(reward_model_name, 
        #cache_dir=training_args.cache_dir,
        #use_flash_attention_2 = True,
                torch_dtype=torch.bfloat16,)
            #reward_model.to(device) 

            reward_model.eval()
            
    else:
        reward_model = AutoModelForCausalLM.from_pretrained(reward_model_name, 
        #cache_dir=training_args.cache_dir,
        #peft_config=lora_config,
        #use_flash_attention_2 = True,
        torch_dtype=torch.bfloat16,)
        #reward_model.to(device) 

        reward_model.eval()
        
    is_deepspeed_enabled = ppo_trainer.accelerator.distributed_type == "DEEPSPEED" and hasattr(
            ppo_trainer.accelerator.state, "deepspeed_plugin"
        )
    
    if is_deepspeed_enabled:
        reward_model = ppo_trainer._prepare_deepspeed(reward_model)
        
    else:
        reward_model = ppo_trainer.accelerator.prepare_model(reward_model, evaluation_mode=True)
        
        
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
    
    
    reward_tokenizer.pad_token = reward_tokenizer.eos_token

    good_token = '+'
    bad_token = '-'
    step_tag = 'ки'
    candidate_tokens = reward_tokenizer.encode(f"{good_token} {bad_token}")[1:] 
    step_tag_id = reward_tokenizer.encode(f"{step_tag}")[-1]


    # generation_kwargs = {
    #     "min_new_tokens": 2,
    #     "top_k": 5,
    #     "top_p": 0.85,
    #     "do_sample": True,
    #     "pad_token_id": tokenizer.pad_token_id,
    #     "eos_token_id": tokenizer.eos_token_id,
    #     "temperature": 0.95,
    #     "num_beams": 1,
    # }
    generation_kwargs = {
        # "min_length": -1,
        #"temperature": 1.2,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        }

    output_min_length = 32

    output_max_length = script_args.output_max_length

    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    conv = get_conv_template(data_args.conv_template)
    # 使用梯度检查点技术
    #ppo_trainer.model.config.use_cache = False
    #ppo_trainer.model.gradient_checkpointing_enable()
    #unwrapped_model = self.accelerator.unwrap_model(self.model)
    unwrapped_model = ppo_trainer.accelerator.unwrap_model(ppo_trainer.model)
    unwrapped_model.gradient_checkpointing_enable()
    unwrapped_model.config.use_cache = False
    loss_meter = AverageMeter()
    reward_meter = AverageMeter()
    line_id = tokenizer.encode("\n")[-1]
    step = 0
    logger.info("Start training...", main_process_only=True)
    for _ in range(script_args.num_epochs):
        for batch in tqdm(ppo_trainer.dataloader, disable=not accelerator.is_local_main_process):
            unwrapped_model.gradient_checkpointing_disable()
            unwrapped_model.config.use_cache = True
            ppo_trainer.model.eval()
            question_tensors = [q_tensor.to(device) for q_tensor in batch["input_ids"]]
            with torch.no_grad():
                response_tensors = ppo_trainer.generate(
                    question_tensors,
                    return_prompt=False,
                    length_sampler=output_length_sampler,
                **generation_kwargs,
                )
            
            batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
            batch["res_len"] = sum([len(r) for r in batch["response"]]) / len(batch["response"])
            logger.info("Generate Responses", main_process_only=True)
            logger.info(batch["response"], main_process_only=True)
            question_list = [q.split("### Instruction:\n")[1] for q in batch["query"]]

            question_list = [q.split("\n\n### Response:")[0] for q in question_list]
            response_list = [process_resp(r) for r in batch["response"]]
            with torch.no_grad():
            #reward_outputs, step_rewards = batch_reward_score(reward_model, reward_tokenizer, question_list, response_list, candidate_tokens, step_tag_id)
                reward_outputs, step_rewards = reward_score_pen_v2(reward_model, reward_tokenizer, question_list, response_list, candidate_tokens, step_tag_id)
            rewards = [output - script_args.reward_baseline for output in reward_outputs]
            step_rewards = [output - script_args.reward_baseline for output in step_rewards]
            logger.info("Step Rewards", main_process_only=True)
            logger.info(step_rewards[:5], main_process_only=True)
            # Run PPO step
            unwrapped_model.gradient_checkpointing_enable()
            unwrapped_model.config.use_cache = False
        
            ppo_trainer.model.train()      
            rewards = [torch.tensor(reward).to(current_device) for reward in rewards]
        
            step_index = [compute_index(r_tensor.tolist(), line_id) for r_tensor in response_tensors]
        

        
            accelerator.free_memory()  # Free memory before forward pass        
            stats = ppo_trainer.step(question_tensors, response_tensors, rewards, None, step_rewards, step_index)
            logger.info(f"cuda max memory allocated: {torch.cuda.max_memory_allocated() // 1024 / 1024 / 1024} GB", main_process_only=True)
            loss_meter.update(float(stats["ppo/loss/total"]), n=len(rewards))
            reward_meter.update(torch.stack(rewards).mean().item(), n=len(rewards))

            ppo_trainer.log_stats(stats, batch, rewards)           
            if accelerator.is_local_main_process and (step + 1) % script_args.logging_steps == 0:
                logs = dict(
                    loss=round(loss_meter.avg, 4),
                    reward=round(reward_meter.avg, 4),
                    learning_rate=stats["ppo/learning_rate"],
                )
                tqdm.write(str(logs))
                logs["step"] = step
                loss_meter.reset()
                reward_meter.reset()
                
                if script_args.save_freq and step and step % script_args.save_freq == 0:
                    state_dict  = ppo_trainer.accelerator.get_state_dict(unwrapped_model.pretrained_model)

                #print(unwrapped_model.pretrained_model)
                    unwrapped_model.pretrained_model.save_pretrained(os.path.join(script_args.output_dir, "{}-{}".format(PREFIX_CHECKPOINT_DIR, step)),
                                                state_dict=state_dict,
                                                safe_serialization=True)
                    ppo_trainer.tokenizer.save_pretrained(os.path.join(script_args.output_dir, "{}-{}".format(PREFIX_CHECKPOINT_DIR, step)))                  
            step = step + 1            

if __name__ == "__main__":
    train()



























    

