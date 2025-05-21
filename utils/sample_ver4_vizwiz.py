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

fuzzy_list = {
        "top10_logits": [],
        "top10_token": [],
        "top10_low": [],
        "top10_high": []
    }

def fuzzification(x_mean, x_std, x_min, x_max, device):
    # import pdb;pdb.set_trace()
    # 리스트를 텐서로 변환
    x_mean = torch.tensor(x_mean, device=device)
    x_std = torch.tensor(x_std, device=device)
    x_min = torch.tensor(x_min, device=device)
    x_max = torch.tensor(x_max, device=device)

    # x_range를 담을 리스트 생성
    x_ranges = []
    
    # 각 min, max 값을 이용해서 개별적으로 linspace 수행
    for min_val, max_val in zip(x_min, x_max):
        x_ranges.append(torch.linspace(min_val - 1, max_val + 1, 100, device=device))
    
    # 리스트를 텐서로 변환 (10, 100) 크기로 만듦
    x_range = torch.stack(x_ranges)

    # Low, High 계산 (브로드캐스팅 활용)
    # import pdb;pdb.set_trace()
    low = torch.exp(-((x_range - (x_mean - x_std).unsqueeze(1)) ** 2) / (2 * (x_std.unsqueeze(1) ** 2)))
    mid = torch.exp(-((x_range - x_mean.view(-1, 1)) ** 2) / (2 * (x_std.view(-1, 1) ** 2)))
    high = torch.exp(-((x_range - (x_mean + x_std).unsqueeze(1)) ** 2) / (2 * (x_std.unsqueeze(1) ** 2)))

    return x_range, (low, mid, high)




    # x_range = torch.linspace(x_min - 1, x_max + 1, 100, device=device)
    # low = torch.exp(-((x_range - (x_mean - x_std)) ** 2) / (2 * (x_std ** 2)))
    # # med = torch.exp(-((x_range - x_mean) ** 2) / (2 * (x_std ** 2)))
    # high = torch.exp(-((x_range - (x_mean + x_std)) ** 2) / (2 * (x_std ** 2)))

    # # import pdb;pdb.set_trace()
    # # print('fuzzification', x_range, (low, med, high))
    # return x_range, (low, high)

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
    # import pdb;pdb.set_trace()

    slopes = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
    intercepts = fp[:-1] - slopes * xp[:-1]

    indices = torch.bucketize(x, xp) - 1
    indices = torch.clamp(indices, 0, len(slopes) - 1)

    result = slopes[indices] * x + intercepts[indices]

    return torch.clamp(result, 0, 1)



def ts_output(x_range_top, memfun_top, top10, next_token_logits, next_token_logits_cds, alpha=0.7):
    '''
    logits top-10 간의 T-S Fuzzy 계산 (top-10 리스트 입력 지원)
    '''
    # import pdb;pdb.set_trace()
    ts_output_debug = {
        "top_low": [],
        "top_high": [],
        "filter_top_low":[],
        "filter_top_high":[],
        "filter_top10_token":[],
        "filter_tok10_logits":[]
    }

    activations = []

    next_token_logits_cds = next_token_logits_cds[0]  # 기존 코드 유지
    f_top10,f_token=torch.topk(next_token_logits_cds,10)
    ts_output_debug['filter_top10_token'].append(f_token.cpu().tolist())
    ts_output_debug['filter_tok10_logits'].append(f_top10.cpu().tolist())

    low_k_list = []
    high_k_list = []
    high_low=[]
    
    filter_low_k_list = []
    filter_high_k_list = []
    filter_high_low=[]
    # import pdb;pdb.set_trace()
    low,mid,high = memfun_top 
    # 각 top-k 값에 대해 low/high 계산
    for k in range(10):
        top_k = top10[0][k]
        x_range_k_k = x_range_top[k]  # 해당 top-k에 대한 x_range
        low_k = low[k]  # 해당 top-k에 대한 memfun
        mid_k = mid[k]
        high_k = high[k]  # 해당 top-k에 대한 memfun
        # import pdb;pdb.set_trace()
        # 인터폴레이션 수행
        top_low_k = torch_interp(top_k, x_range_k_k, low_k)
        top_mid_k = torch_interp(top_k, x_range_k_k, mid_k)
        top_high_k = torch_interp(top_k, x_range_k_k, high_k)
        
        
        high_low_item = top_high_k.item()-top_low_k.item()

        low_k_list.append(top_low_k.item())
        high_k_list.append(top_high_k.item())
        high_low.append(high_low_item)

    #필터
    for k in range(10):
        top_k = f_top10[0][k]
        x_range_k_k = x_range_top[k]  # 해당 top-k에 대한 x_range
        low_k = low[k]  # 해당 top-k에 대한 memfun
        high_k = high[k]  # 해당 top-k에 대한 memfun
        # import pdb;pdb.set_trace()
        # 인터폴레이션 수행
        top_low_k = torch_interp(top_k, x_range_k_k, low_k)
        top_high_k = torch_interp(top_k, x_range_k_k, high_k)
        high_low_item = top_high_k.item()-top_low_k.item()

        filter_low_k_list.append(top_low_k.item())
        filter_high_k_list.append(top_high_k.item())
        filter_high_low.append(high_low_item)
        # 디버깅 정보 저장
    ts_output_debug['top_low'].append(low_k_list)
    ts_output_debug['top_high'].append(high_k_list)
    ts_output_debug['filter_top_low'].append(filter_low_k_list)
    ts_output_debug['filter_top_high'].append(filter_high_k_list)


    t3_9_rule=min(high_low[4],high_low[5],high_low[6],high_low[7]) 
    
    min_max=abs(min(high_low) - max(high_low)) ## min값과 max 값 차이 
    
    f_t3_9_rule=min(filter_high_low[4],filter_high_low[5],filter_high_low[6],filter_high_low[7]) 
    

    rule1 = t3_9_rule > 0 #원본 이미지 할루시네이션 일어났을 수 있음 
    rule2 = low_k_list[0]< high_k_list[1] #원본 이미지 할루시네이션 일어났을 수 있음 
    rule3 = min_max < 0.25 # 원본 변동성이 적음
    rule4 = high_low[0] > high_low[1] # 할루시네이션이 아님 
    rule5 = filter_high_low[9] > 0 # 필터는 할루시네이션이지만 원본이 할루시네이션이 아닐 수 있음 

    rule6 = high_low[1]>0.25


    cd_alpha = 0 
    # rule5 = f_t3_9_rule > 0 #필터 이미지 할루시네이션 일어났을 수 있음 
    # rule6 = filter_low_k_list[0]< filter_high_k_list[1] #필터 이미지 할루시네이션 일어났을 수 있음 
    
    # if rule3:  # 원본 변동성이 적은 경우 할루시네이션이 아님 
    #     cd_alpha = 0
    # if rule4: # top2 값이 더 적은 경우 할루시네이션이 아님 
    #     cd_alpha = 0
    # elif ~rule4 and rule5:
    #     cd_alpha = 0
    # elif high_low[0] <t3_9_rule: #할루시네이션일 수 있음 
    #     cd_alpha = 1
    # else:
    #     cd_alpha = 1

    if rule6: #할루시네이션
        # import pdb;pdb.set_trace()
        next_token_logits[0][0],next_token_logits[0][1] =next_token_logits[0][1],next_token_logits[0][0] 
    else:
        cd_alpha = 0
    
    
    # if rule1 or rule2 and rule3 or rule4: #둘다 할루시네이션이 일어난 경우
    #     cd_alpha = 1
    # elif ~rule1 or ~rule2 and ~rule3 or ~rule4:# 둘다 할루시네이션이 아닌 경우 
    #     cd_alpha = 0
    # elif rule1 or rule2 and ~rule3 or ~rule4:# 원본은 할루시네이션이지만 필터는 할루시네이션이 아닌 경우
    #     cd_alpha = -1.1
    # else:
    #     cd_alpha = 0
        
        
    # adjusted_logits = adjust_logits(next_token_logits, hallucination_detected)
    
    adjusted_logits = (1 + cd_alpha) * next_token_logits - cd_alpha * next_token_logits_cds

    # adjusted_logits = next_token_logits

    return next_token_logits, adjusted_logits, ts_output_debug



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

    fuzzy_list = {
        "top10_logits": [],
        "top10_token": [],
        "filter_top10_logits":[],
        "filter_top10_token":[],
        "top10_low": [],
        "top10_high": [],
        "filter_top10_low":[],
        "filter_top10_high":[],
        "orignal_next_tokens":[],
        "filter_top10_tokens":[],
        'filter_1_logits':[],
        'filter_2_logits':[],
        'filter_3_logits':[],
        'filter_4_logits':[],
        "next_token_logits":[]
    }
    statics_list = {
        "viz_top1_probs": [],
        "viz_top1_log_probs": [],
        "viz_top1_mins": [],
        "viz_top1_maxs": [],
        "viz_top1_means": [],
        "viz_top1_preds": [],
    }

    


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
        
        logits_top_mean = model_kwargs.get("logits_top_mean")
        logits_top_std = model_kwargs.get("logits_top_std")
        logits_top_min = model_kwargs.get("logits_top_min")
        logits_top_max = model_kwargs.get("logits_top_max")

        
        alpha = model_kwargs.get("cd_alpha")


        # if membership_init:   
        #     # next_token_logits = outputs.logits[:, -1, :] 
        #     # import pdb;pdb.set_trace()
        #     next_token_logits = outputs.logits[:, -1, :] 
        #     log_softmax = nn.functional.log_softmax(next_token_logits, dim=-1)

        #     logp_top1_token = log_softmax.argmax(dim=-1, keepdim=True)  # (1, 1) 형태의 인덱스
        #     top1_log_prob = log_softmax.gather(dim=-1, index=logp_top1_token)  # (1, 1) 형태

            
        #     return logp_top1_token, top1_log_prob
        if membership_init:   
            # next_token_logits = outputs.logits[:, -1, :] 
            # import pdb;pdb.set_trace()
            next_token_logits = outputs.logits[:, -1, :] 
            prob = nn.functional.softmax(next_token_logits, dim=-1)
            log_prob = nn.functional.log_softmax(next_token_logits, dim=-1)

            top1_prob, top1_p_token = torch.topk(prob, k=1, dim=-1)
            top1_log_prob, top1_logp_token = torch.topk(log_prob, k=1, dim=-1)
            top1_min, min_token = torch.min(next_token_logits, dim=-1)
            top1_max, max_token = torch.max(next_token_logits, dim=-1)
            top1_mean = torch.mean(next_token_logits, dim=-1)

            # logp_top1_token = log_softmax.argmax(dim=-1, keepdim=True)  # (1, 1) 형태의 인덱스
            # top1_log_prob = log_softmax.gather(dim=-1, index=logp_top1_token)  # (1, 1) 형태

            
            return top1_p_token, top1_prob, top1_log_prob, top1_min, top1_max, top1_mean


        if logits_top_mean != None:
            # import pdb;pdb.set_trace()
            x_range_list, memfun_list = fuzzification(logits_top_mean, logits_top_std, logits_top_min, logits_top_max, device)
            
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

                # import pdb;pdb.set_trace()
                
                next_token_logits_cds=[next_token_logits_1,next_token_logits_2,next_token_logits_3,next_token_logits_4]
                # next_token_logits_cds_output=[next_token_logits_1.cpu().tolist(),next_token_logits_2.cpu().tolist(),next_token_logits_3.cpu().tolist(),next_token_logits_4.cpu().tolist()]
                x_vals = [torch.max(cd) for cd in next_token_logits_cds]
                
                f1_top10,f1_token=torch.topk(next_token_logits_1,10)
                f2_top10,f2_token=torch.topk(next_token_logits_2,10)
                f3_top10,f3_token=torch.topk(next_token_logits_3,10)
                f4_top10,f4_token=torch.topk(next_token_logits_4,10)

                fuzzy_list['filter_top10_tokens'].append({
                    "f1_token":f1_token.cpu().tolist(),
                    "f2_token":f2_token.cpu().tolist(),
                    "f3_token":f3_token.cpu().tolist(),
                    "f4_token":f4_token.cpu().tolist()
                })

                fuzzy_list['filter_1_logits'].extend(next_token_logits_1.cpu().tolist())
                fuzzy_list['filter_2_logits'].extend(next_token_logits_2.cpu().tolist())
                fuzzy_list['filter_3_logits'].extend(next_token_logits_3.cpu().tolist())
                fuzzy_list['filter_4_logits'].extend(next_token_logits_4.cpu().tolist())
                
                fuzzy_list['next_token_logits'].extend(next_token_logits.cpu().tolist())
                
                top10,token=torch.topk(next_token_logits,10)
                # top1 = topk[0][0]
                # top2 = topk[0][1]
                
                # x1_val= torch.max(next_token_logits)


                fuzzy_list['top10_logits'].append(top10.cpu().tolist())
                fuzzy_list['top10_token'].append(token.cpu().tolist())

                #T-S Fuzzy
                orginal_logits,adjusted_logits, ts_output_debug = ts_output(x_range_list,memfun_list,top10, next_token_logits, next_token_logits_cds, alpha)
                

                # import pdb;pdb.set_trace()
                fuzzy_list['top10_low'].extend(ts_output_debug['top_low'])
                fuzzy_list['top10_high'].extend(ts_output_debug['top_high'])
                fuzzy_list['filter_top10_low'].extend(ts_output_debug['filter_top_low'])
                fuzzy_list['filter_top10_high'].extend(ts_output_debug['filter_top_high'])
                fuzzy_list['filter_top10_token'].extend(ts_output_debug['filter_top10_token'])
                fuzzy_list['filter_top10_logits'].extend(ts_output_debug['filter_tok10_logits'])

                
                
                ## cd_comments: pre-process logits from contrastive inputs
                cd_beta = model_kwargs.get("cd_beta") if model_kwargs.get("cd_beta") is not None else 0.1
            

                # version 2 set cutoff for Adaptive Plausibility Constraints
                cutoff = torch.log(torch.tensor(cd_beta)) + next_token_logits.max(dim=-1, keepdim=True).values
                
                diffs = adjusted_logits
                cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))

                ## cd_comments: apply temperature warping and top-k filtering in contrastive decoding
                cd_logits = logits_processor(input_ids, cd_logits)
                cd_logits = logits_warper(input_ids, cd_logits)

                next_token_scores = cd_logits
                cd_probs = nn.functional.softmax(cd_logits, dim=-1)
                next_tokens = torch.multinomial(cd_probs, num_samples=1).squeeze(1)

                logits= logits_processor(input_ids, orginal_logits)
                logits = logits_warper(input_ids, logits)

                logits_probs = nn.functional.softmax(logits, dim=-1)
                orignal_next_tokens = torch.multinomial(logits_probs, num_samples=1).squeeze(1)
                
                fuzzy_list['orignal_next_tokens'].append(orignal_next_tokens.item())
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
                # cd_probs = nn.functional.softmax(cd_logits, dim=-1)
                # next_tokens = torch.multinomial(cd_probs, num_samples=1).squeeze(1)
                _, next_tokens = torch.topk(next_token_scores, 1)

        else:
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            # next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            _, next_tokens = torch.topk(next_token_scores, 1)
            #### store the statics informations
            next_token_logits = outputs.logits[:, -1, :] 
            prob = nn.functional.softmax(next_token_logits, dim=-1)
            log_prob = nn.functional.log_softmax(next_token_logits, dim=-1)

            top1_prob, top1_p_token = torch.topk(prob, k=1, dim=-1)
            top1_log_prob, top1_logp_token = torch.topk(log_prob, k=1, dim=-1)
            top1_min, min_token = torch.min(next_token_logits, dim=-1)
            top1_max, max_token = torch.max(next_token_logits, dim=-1)
            top1_mean = torch.mean(next_token_logits, dim=-1)
            statics_list["viz_top1_probs"].append(top1_prob)
            statics_list["viz_top1_log_probs"].append(top1_log_prob)
            statics_list["viz_top1_mins"].append(top1_min)
            statics_list["viz_top1_maxs"].append(top1_max)
            statics_list["viz_top1_means"].append(top1_mean)
            statics_list["viz_top1_preds"].append(top1_p_token)
            
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
        # input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        input_ids = torch.cat([input_ids, next_tokens], dim=-1)
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
        # import pdb;pdb.set_trace()
        return {
                "sequences":input_ids,
                "scores":scores,
                "attentions":decoder_attentions,
                "hidden_states":decoder_hidden_states,
                "statics_list": statics_list
                # "fuzzydecoding" :fuzzy_list
        }
        # import pdb;pdb.set_trace()
        # if self.config.is_encoder_decoder:
        #     return SampleEncoderDecoderOutput(
        #         sequences=input_ids,
        #         scores=scores,
        #         encoder_attentions=encoder_attentions,
        #         encoder_hidden_states=encoder_hidden_states,
        #         decoder_attentions=decoder_attentions,
        #         cross_attentions=cross_attentions,
        #         decoder_hidden_states=decoder_hidden_states,
        #     )
        # else:
        #     return SampleDecoderOnlyOutput(
        #         sequences=input_ids,
        #         scores=scores,
        #         attentions=decoder_attentions,
        #         hidden_states=decoder_hidden_states,
        #         fuzzydecoding = fuzzy_list
        #     )
    else:
        return input_ids

def evolve_vcd_sampling():
    transformers.generation.utils.GenerationMixin.sample = sample
    # sample is now a protected function in the latest Transformers library
    transformers.generation.utils.GenerationMixin._sample = sample
