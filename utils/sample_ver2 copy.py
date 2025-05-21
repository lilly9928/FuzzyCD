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

    import pdb;pdb.set_trace()
    # print('fuzzification', x_range, (low, med, high))
    return x_range, (low, med, high)

# def jensen_shannon_divergence(p, q):
#     # Calculate Jensen-Shannon Divergence between two probability distributions p and q
#     m = 0.5 * (p + q)
#     kl_pm = torch.nn.functional.kl_div(p.log(), m, reduction='batchmean')
#     kl_qm = torch.nn.functional.kl_div(q.log(), m, reduction='batchmean')
#     return 0.5 * (kl_pm + kl_qm)


# def js_divergence(p, q):

#     p = nn.functional.softmax(p, dim=-1)
#     q = nn.functional.softmax(q, dim=-1)

#     m = 0.5 * (p + q) 

#     kl_pm = nn.functional.kl_div(nn.functional.log_softmax(p), m, reduction='batchmean')
#     kl_qm = nn.functional.kl_div(nn.functional.log_softmax(q), m, reduction='batchmean')

#     return 0.5 * (kl_pm + kl_qm)

import torch
import torch.nn as nn

def js_divergence(logits_p, logits_q):
    # logits를 확률 분포로 변환
    p = nn.functional.softmax(logits_p, dim=-1)
    q = nn.functional.softmax(logits_q, dim=-1)
    
    # 혼합 분포 m 계산
    m = 0.5 * (p + q)
    
    # 이미 softmax를 적용했으므로 torch.log를 사용해 로그 확률 계산
    kl_pm = nn.functional.kl_div(torch.log(p), m, reduction='batchmean')
    kl_qm = nn.functional.kl_div(torch.log(q), m, reduction='batchmean')
    
    return 0.5 * (kl_pm + kl_qm)

def torch_interp(x, xp, fp):
    x = x.unsqueeze(0) if x.dim() == 0 else x
    xp = xp.unsqueeze(0) if xp.dim() == 0 else xp
    fp = fp.unsqueeze(0) if fp.dim() == 0 else fp
    # import pdb;pdb.set_trace()

    slopes = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
    intercepts = fp[:-1] - slopes * xp[:-1]

    indices = torch.bucketize(x, xp) - 1
    indices = torch.clamp(indices, 0, len(slopes) - 1)

    return slopes[indices] * x + intercepts[indices]

def ts_output(x_range, x_vals, memfun, next_token_logits, next_token_logits_cds, alpha=0.7):
    low, med, high = memfun
    # import pdb;pdb.set_trace()
    activations = []
    all_y = []

    x1_val, *x2_vals = x_vals
    for idx in range(len(x2_vals)): #x2는 4개 있음 -> 각각 4번 FUZZY 계산을 해야함 
        # 각 필터별 rule에 대한 weight 
        x2 = x2_vals[idx]
        filter_logits = next_token_logits_cds[idx]
        rule1 = torch_interp(x1_val, x_range, low) * torch_interp(x2, x_range, low) #filter logits language prior high
        rule2 = torch_interp(x1_val, x_range, low) * torch_interp(x2, x_range, med)
        rule3 = torch_interp(x1_val, x_range, low) * torch_interp(x2, x_range, high) # filter가 커져야함
        rule4 = torch_interp(x1_val, x_range, med) * torch_interp(x2, x_range, low) #filter logits language prior high 
        rule5 = torch_interp(x1_val, x_range, med) * torch_interp(x2, x_range, med)
        rule6 = torch_interp(x1_val, x_range, med) * torch_interp(x2, x_range, high)
        rule7 = torch_interp(x1_val, x_range, high) * torch_interp(x2, x_range, low) #filter logits language prior high 
        rule8 = torch_interp(x1_val, x_range, high) * torch_interp(x2, x_range, med)
        rule9 = torch_interp(x1_val, x_range, high) * torch_interp(x2, x_range, high)

        #y 값 계산  
        # 원본 confidence score가 높을때 rule7,8,9 
        # rule36_789_y = next_token_logits
        # (1+alpha)*next_token_logits - alpha*filter_logits  
        
        # language prior가 high 일때  rule1 4
        # alpha = 1
        # rule14_y = (1+alpha)*next_token_logits - alpha*filter_logits  
        
        # # language prior가 med 일때  rule2 5
        # alpha = 0.5
        # rule25_y = (1+alpha)*next_token_logits - alpha*filter_logits  

        # #전체 output y 
        # weight_rule_y=rule1*rule14_y+rule2*rule25_y+rule3*rule36_789_y+ rule4*rule14_y+rule5*rule25_y+rule6*rule36_789_y+ rule7*rule36_789_y+rule8*rule36_789_y+rule9*rule36_789_y

        # import pdb;pdb.set_trace()
        a_high = 1.5  # 낮은 priority에서 기본값
        a_med = 1  # 중간 priority에서 약간 증폭
        a_low = 0  # 높은 priority에서 강하게 증폭
        rule3_score= -0.5

        rule1_y = a_high * filter_logits  
        rule2_y = a_med * filter_logits  
        rule3_y = rule3_score * filter_logits  
        rule4_y = a_high * filter_logits  
        rule5_y = a_med * filter_logits  
        rule6_y = a_low * filter_logits  
        rule7_y = a_low * filter_logits  
        rule8_y = a_low * filter_logits  
        rule9_y = a_low * filter_logits 

        # 총 활성화도 계산
        total_rule_activation=rule1 + rule2 + rule3 + rule4 + rule5 + rule6 + rule7 + rule8 + rule9 

        weighted_y = (rule1 * rule1_y + rule2 * rule2_y + rule3 * rule3_y +
                      rule4 * rule4_y + rule5 * rule5_y + rule6 * rule6_y +
                      rule7 * rule7_y + rule8 * rule8_y + rule9 * rule9_y) / total_rule_activation

        # y = weight_rule_y/total_rule_activation
        all_y.append(weighted_y)
    
    all_y_tensor = torch.stack(all_y)
    # import pdb;pdb.set_trace()
    jsd_values = torch.tensor([
        js_divergence(all_y_tensor[i], next_token_logits) for i in range(all_y_tensor.shape[0])
    ])

    top1_jsd, jsd_idx = torch.topk(jsd_values,1)

    alpha = 1
    final_filter_logits= (1+alpha)*next_token_logits - alpha * all_y_tensor[jsd_idx]


    # final_filter_logits = all_y[0]

    return final_filter_logits[0]



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


    ##지우기 
    next_token_logits_list = []
    mynext_token_logits_list = []
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
        
        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :] 


        membership_init = model_kwargs.get("membership_init")
        logits_mean = model_kwargs.get("logits_top_mean")
        logits_std = model_kwargs.get("logits_top_std")
        logits_min = model_kwargs.get("logits_top_min")
        logits_max = model_kwargs.get("logits_top_max")
        alpha = model_kwargs.get("cd_alpha")


        if membership_init:   
            next_token_logits = outputs.logits[:, -1, :] 
            log_softmax=nn.functional.log_softmax(next_token_logits, dim=-1)

            logp_top1_token = log_softmax.argmax(dim=-1, keepdim=True)  # (1, 1) 형태의 인덱스
            top1_log_prob = log_softmax.gather(dim=-1, index=logp_top1_token)  # (1, 1) 형태

            # mean_= torch.mean(next_token_logits)
            return top1_log_prob


        if logits_mean != None:
            # import pdb;pdb.set_trace()
            x_range, memfun = fuzzification(logits_mean, logits_std, logits_min, logits_max, device)
            # import pdb;pdb.set_trace()
        # next_token_logits = outputs.logits[:, -1, :]
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
                ## cd_comments: forward pass of the model with distorted image input 0
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

                # import pdb;pdb.set_trace()
                
                next_token_logits_cds=[next_token_logits_1,next_token_logits_2,next_token_logits_3,next_token_logits_4]
                # next_token_logits_cds=[next_token_logits_1]


                log_probs = nn.functional.log_softmax(next_token_logits, dim=-1)
                log_probs_cds = [nn.functional.log_softmax(cd, dim=-1) for cd in next_token_logits_cds]

                logp_top1_token = log_probs.argmax(dim=-1, keepdim=True)  # (1, 1) 형태의 인덱스
                top1_log_prob = log_probs.gather(dim=-1, index=logp_top1_token)  # (1, 1) 형태

                log_probs_cds_token = [lp.argmax(dim=-1, keepdim=True) for lp in log_probs_cds]  # 리스트로 수정
                top1_log_probs_cds = [lp.gather(dim=-1, index=idx) for lp, idx in zip(log_probs_cds, log_probs_cds_token)]


                # next_token_logits_cds=[next_token_logits_1,next_token_logits_2]
                # x_vals = [torch.mean(next_token_logits)] + [torch.mean(cd) for cd in next_token_logits_cds]
                x_vals = [top1_log_prob] + top1_log_probs_cds
                # nn.functional.log_softmax(next_token_logits, dim=-1)
                x1_val= torch.max(next_token_logits)
                
                # import pdb;pdb.set_trace()
                #지우기 
                next_token_logits_list.append(next_token_logits)

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

                mynext_token_logits_list.append(cd_logits)

                # import pdb;pdb.set_trace()
                _,next_tokens = torch.topk(cd_logits,1)

                # next_token_scores = cd_logits
                # cd_probs = nn.functional.softmax(cd_logits, dim=-1)
                # next_tokens = torch.multinomial(cd_probs, num_samples=1).squeeze(1)
            
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
        # import pdb;pdb.set_trace()
        # update generated ids, model inputs, and length for next step
        # input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        input_ids = torch.cat([input_ids, next_tokens[0][:, None]], dim=-1) #topk
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
        return input_ids,mynext_token_logits_list,next_token_logits_list

def evolve_vcd_sampling():
    transformers.generation.utils.GenerationMixin.sample = sample
    # sample is now a protected function in the latest Transformers library
    transformers.generation.utils.GenerationMixin._sample = sample
