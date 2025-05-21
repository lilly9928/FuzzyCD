import copy
import inspect
import warnings
import skfuzzy as fuzz
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
import transformers
from transformers.generation.utils import SampleOutput


def fuzzification(x_mean, x_std, x_min, x_max, device):
    x_range = torch.linspace(x_min - 1, x_max + 1, 100, device=device)
    low = torch.exp(-((x_range - (x_mean - x_std)) ** 2) / (2 * (x_std ** 2)))
    med = torch.exp(-((x_range - x_mean) ** 2) / (2 * (x_std ** 2)))
    high = torch.exp(-((x_range - (x_mean + x_std)) ** 2) / (2 * (x_std ** 2)))
    # print('fuzzification', x_range, (low, med, high))
    return x_range, (low, med, high)

def jensen_shannon_divergence(p, q):
    # Calculate Jensen-Shannon Divergence between two probability distributions p and q
    m = 0.5 * (p + q)
    kl_pm = torch.nn.functional.kl_div(p.log(), m, reduction='batchmean')
    kl_qm = torch.nn.functional.kl_div(q.log(), m, reduction='batchmean')
    return 0.5 * (kl_pm + kl_qm)

def torch_interp(x, xp, fp):
    x = x.unsqueeze(0) if x.dim() == 0 else x
    xp = xp.unsqueeze(0) if xp.dim() == 0 else xp
    fp = fp.unsqueeze(0) if fp.dim() == 0 else fp

    slopes = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
    intercepts = fp[:-1] - slopes * xp[:-1]

    indices = torch.bucketize(x, xp) - 1
    indices = torch.clamp(indices, 0, len(slopes) - 1)

    return slopes[indices] * x + intercepts[indices]

def ts_output(x_range, x_vals, memfun, next_token_logits, next_token_logits_cds, alpha=0.7):
    low, med, high = memfun
    next_token_logits_cds = next_token_logits_cds[0]

    # 각 규칙의 활성화도 계산 (여러 x_vals에 대해 활성화도 계산)
    activations = []
    for x1_val, x2_val in zip(x_vals, next_token_logits_cds):
        rule1 = torch_interp(x1_val, x_range, low) * torch_interp(x2_val, x_range, low)
        rule2 = torch_interp(x1_val, x_range, low) * torch_interp(x2_val, x_range, med)
        rule3 = torch_interp(x1_val, x_range, low) * torch_interp(x2_val, x_range, high)
        rule4 = torch_interp(x1_val, x_range, med) * torch_interp(x2_val, x_range, low)
        rule5 = torch_interp(x1_val, x_range, med) * torch_interp(x2_val, x_range, med)
        rule6 = torch_interp(x1_val, x_range, med) * torch_interp(x2_val, x_range, high)
        rule7 = torch_interp(x1_val, x_range, high) * torch_interp(x2_val, x_range, low)
        rule8 = torch_interp(x1_val, x_range, high) * torch_interp(x2_val, x_range, med)
        rule9 = torch_interp(x1_val, x_range, high) * torch_interp(x2_val, x_range, high)

        # 총 활성화도 계산
        total_rule_activation = rule1 + rule2 + rule3 + rule4 + rule5 + rule6 + rule7 + rule8 + rule9
        # /print('total_rule_activation',total_rule_activation)
        activations.append((total_rule_activation, [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9]))
        # print('activations',activations)

    # 모든 활성화도가 0인지 확인
    if all(torch.all(total_activation[0] == 0) for total_activation in activations):
        return next_token_logits  # 모든 활성화도가 0이면 원본 logits 반환

    # 퍼지 규칙 활성화도를 기반으로 각 로짓 확률 조정
    combined_probs = torch.zeros_like(next_token_logits, device=next_token_logits.device)
    total_combined_activation = 0

    for (total_activation, rules), logits_cd in zip(activations, next_token_logits_cds):
        if torch.all(total_activation == 0):
            continue

        weight1 = rules[0] + rules[3] + rules[6]  # x1_val과 관련된 규칙 활성화도
        weight2 = rules[1] + rules[4] + rules[7]  # x2_val과 관련된 규칙 활성화도
        weight3 = rules[2] + rules[5] + rules[8]  # 두 값이 함께 높거나 낮은 경우의 규칙 활성화도

        # print(weight1)
        # print(weight2)
        # print(weight3)

        p = torch.softmax(next_token_logits, dim=0)
        q = torch.softmax(logits_cd, dim=0)
        combined_probs += (alpha * p + (1-alpha) * (weight1 * p + weight2 * q + weight3 * (p + q) / 2) / (total_activation + 1e-9))
        # print(p)
        # print(combined_probs)

        total_combined_activation += total_activation

    # 최종 결합된 확률을 통해 로짓 계산
    combined_probs /= len(next_token_logits_cds)  # 평균화
    combined_probs = torch.clamp(combined_probs, min=1e-9, max=1.0)  # 확률이 0 이하가 되지 않도록 보정
    combined_logits = torch.log(combined_probs)  # log(0)을 방지하기 위해 작은 값을 더함

    # Softmax 적용하여 확률 분포 계산
    js_div = jensen_shannon_divergence(torch.softmax(next_token_logits, dim=0), torch.softmax(next_token_logits_cds[0], dim=0))
    js_div = torch.clamp(js_div, max=10.0)
    # JS divergence를 활용하여 최종 조정
    # adjusted_logits = (1 + js_div) * combined_logits + js_div * next_token_logits_cds[0]
    adjusted_logits = (1 + js_div) * next_token_logits - js_div * combined_logits

    # `adjusted_logits`에 대한 NaN 또는 inf 값 검사 및 보정
    adjusted_logits = torch.where(torch.isnan(adjusted_logits) | torch.isinf(adjusted_logits), torch.zeros_like(adjusted_logits), adjusted_logits)
    # print(adjusted_logits)

    return adjusted_logits



def sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[SampleOutput, torch.LongTensor]:
    device = 'cuda'
    use_cd = model_kwargs.get("use_cd")
    use_FLL = model_kwargs.get("use_FLL")

    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id


    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )

    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    this_peer_finished = False  # used by synced_gpus only
    model_kwargs_cd = model_kwargs.copy() # copy model_kwargs for cd only for the first forward process
    # auto-regressive generation
    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, cd=use_cd, **model_kwargs)
        membership_init = model_kwargs.get("membership_init")
    
        logits_mean = model_kwargs.get("logits_mean")
        logits_std = model_kwargs.get("logits_std")
        logits_min = model_kwargs.get("logits_min")
        logits_max = model_kwargs.get("logits_max")
        alpha = model_kwargs.get("cd_alpha")

        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if membership_init:   
            next_token_logits = outputs.logits[:, -1, :] 
            mean_= torch.mean(next_token_logits)
            return mean_


        if logits_mean != None:
            x_range, memfun = fuzzification(logits_mean, logits_std, logits_min, logits_max, device)

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]
        # mean_= torch.mean(next_token_logits)
        ## For contrastive decoding initial

        output_attentions_wo_img = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states_wo_img = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        
        if use_cd:
            
            if use_FLL:
                ## cd_comments: forward pass of the model with distorted image input
                model_inputs_1 = self.prepare_inputs_for_generation_f(input_ids, img_num=1,  **model_kwargs_cd)
                model_inputs_2 = self.prepare_inputs_for_generation_f(input_ids, img_num=2, **model_kwargs_cd)
                model_inputs_3 = self.prepare_inputs_for_generation_f(input_ids, img_num=3, **model_kwargs_cd)
                model_inputs_4 = self.prepare_inputs_for_generation_f(input_ids, img_num=4, **model_kwargs_cd)
                
                outputs_1 = self(
                    **model_inputs_1,
                    return_dict=True,
                    output_attentions=output_attentions_wo_img,
                    output_hidden_states=output_hidden_states_wo_img,
                )
                next_token_logits_1 = outputs_1.logits[:, -1, :]
                
                outputs_2 = self(
                    **model_inputs_2,
                    return_dict=True,
                    output_attentions=output_attentions_wo_img,
                    output_hidden_states=output_hidden_states_wo_img,
                )
                next_token_logits_2 = outputs_2.logits[:, -1, :]
                
                outputs_3= self(
                    **model_inputs_3,
                    return_dict=True,
                    output_attentions=output_attentions_wo_img,
                    output_hidden_states=output_hidden_states_wo_img,
                )
                next_token_logits_3 = outputs_3.logits[:, -1, :]
                
                outputs_4 = self(
                    **model_inputs_4,
                    return_dict=True,
                    output_attentions=output_attentions_wo_img,
                    output_hidden_states=output_hidden_states_wo_img,
                )
                next_token_logits_4 = outputs_4.logits[:, -1, :]

                
                next_token_logits_cds=[next_token_logits_1,next_token_logits_2,next_token_logits_3,next_token_logits_4]
                x_vals = [torch.mean(next_token_logits)] + [torch.mean(cd) for cd in next_token_logits_cds]
                

                adjusted_logits = ts_output(x_range, x_vals, memfun, next_token_logits, next_token_logits_cds, alpha)
                ## cd_comments: pre-process logits from contrastive inputs
                cd_beta = model_kwargs.get("cd_beta") if model_kwargs.get("cd_beta") is not None else 0.1
            

                # version 2 set cutoff for Adaptive Plausibility Constraints
                cutoff = torch.log(torch.tensor(cd_beta)) + next_token_logits.max(dim=-1, keepdim=True).values
                
                diffs = adjusted_logits
                cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))
                # print(cd_logits)

                ## cd_comments: apply temperature warping and top-k filtering in contrastive decoding
                cd_logits = logits_processor(input_ids, cd_logits)
                cd_logits = logits_warper(input_ids, cd_logits)

                next_token_scores = cd_logits
                cd_probs = nn.functional.softmax(cd_logits, dim=-1)
                next_tokens = torch.multinomial(cd_probs, num_samples=1).squeeze(1)
            
            else:
                model_inputs_1 = self.prepare_inputs_for_generation_f(input_ids, img_num=1,  **model_kwargs_cd)
                
                outputs_1 = self(
                    **model_inputs_1,
                    return_dict=True,
                    output_attentions=output_attentions_wo_img,
                    output_hidden_states=output_hidden_states_wo_img,
                )
                next_token_logits_cd = outputs_1.logits[:, -1, :]
                
                aalpha = 1.0
                diffs = (1+aalpha)*next_token_logits - aalpha*next_token_logits_cd
                ## cd_comments: pre-process logits from contrastive inputs
                cd_beta = model_kwargs.get("cd_beta") if model_kwargs.get("cd_beta") is not None else 0.1
            

                # version 2 set cutoff for Adaptive Plausibility Constraints
                cutoff = torch.log(torch.tensor(cd_beta)) + next_token_logits.max(dim=-1, keepdim=True).values
                
                cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))
                # print(cd_logits)

                ## cd_comments: apply temperature warping and top-k filtering in contrastive decoding
                cd_logits = logits_processor(input_ids, cd_logits)
                cd_logits = logits_warper(input_ids, cd_logits)

                next_token_scores = cd_logits
                cd_probs = nn.functional.softmax(cd_logits, dim=-1)
                next_tokens = torch.multinomial(cd_probs, num_samples=1).squeeze(1)

        else:
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)



        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )


        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        ## cd_comments: update model_kwargs_cd for contrastive decoding
        if use_cd:
            if use_FLL:
                model_kwargs_cd_1 = self._update_model_kwargs_for_generation(
                outputs_1, model_kwargs_cd, is_encoder_decoder=self.config.is_encoder_decoder
                )
                model_kwargs_cd_2 = self._update_model_kwargs_for_generation(
                    outputs_2, model_kwargs_cd_1, is_encoder_decoder=self.config.is_encoder_decoder
                )
                model_kwargs_cd_3 = self._update_model_kwargs_for_generation(
                    outputs_3, model_kwargs_cd_2, is_encoder_decoder=self.config.is_encoder_decoder
                )
                model_kwargs_cd = self._update_model_kwargs_for_generation(
                    outputs_4, model_kwargs_cd_3, is_encoder_decoder=self.config.is_encoder_decoder
                )
            else:
                model_kwargs_cd = self._update_model_kwargs_for_generation(
                    outputs_1, model_kwargs_cd, is_encoder_decoder=self.config.is_encoder_decoder
                )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return SampleEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return SampleDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return input_ids

def evolve_vcd_sampling():
    transformers.generation.utils.GenerationMixin.sample = sample
    # sample is now a protected function in the latest Transformers library
    transformers.generation.utils.GenerationMixin._sample = sample
