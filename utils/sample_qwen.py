import copy
import inspect
import warnings
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
# from transformers.generation.utils import SampleOutput
from transformers.generation.utils import GenerateNonBeamOutput
from transformers.generation.configuration_utils import (
    NEED_SETUP_CACHE_CLASSES_MAPPING,
    QUANT_BACKEND_CLASSES_MAPPING,
    GenerationConfig,
    GenerationMode,
)

def fuzzification(x_mean, x_std, x_min, x_max, device):
    x_range = torch.linspace(x_min - 1, x_max + 1, 100, device=device)
    low = torch.exp(-((x_range - (x_mean - x_std)) ** 2) / (2 * (x_std ** 2)))
    med = torch.exp(-((x_range - x_mean) ** 2) / (2 * (x_std ** 2)))
    high = torch.exp(-((x_range - (x_mean + x_std)) ** 2) / (2 * (x_std ** 2)))

    return x_range, (low, med, high)

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

    debug_rule = []
    # jsd_values_list=[]
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

        debug_rule.append({
            "x1_low":torch_interp(x2, x_range, low).item(),
            "x1_med":torch_interp(x2, x_range, med).item(),
            "x1_high":torch_interp(x2, x_range, high).item(),
            "x2_low":torch_interp(x2, x_range, low).item(),
            "x2_med":torch_interp(x2, x_range, med).item(),
            "x2_high":torch_interp(x2, x_range, high).item(),
        })
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
        # import pdb;pdb.set_trace()
        
        # y = weight_rule_y/total_rule_activation
        all_y.append(weighted_y)
    # import pdb;pdb.set_trace()
    all_y_tensor = torch.stack(all_y)
    # import pdb;pdb.set_trace()
    jsd_values = torch.tensor([
        js_divergence(all_y_tensor[i], next_token_logits) for i in range(all_y_tensor.shape[0])
    ])
    # jsd_values_list.append(jsd_values.tolist())
    # debug_rule.append({"jsd_values":jsd_values_list})
    top1_jsd, jsd_idx = torch.topk(jsd_values,1)
    # debug_rule.append({"jsd_idx":jsd_idx.item()})

    alpha = 1
    final_filter_logits= (1+alpha)*next_token_logits - alpha * all_y_tensor[jsd_idx]


    # final_filter_logits = all_y[0]
    
    # import pdb;pdb.set_trace()

    return final_filter_logits[0],debug_rule

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
        generation_config: GenerationConfig = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        device = input_ids.device
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
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
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
        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

        model_forward = self.__call__
        compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
        if compile_forward:
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            model_forward = self.get_compiled_call(generation_config.compile_config)

        if generation_config.prefill_chunk_size is not None:
            model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
            is_prefill = False
        else:
            is_prefill = True
        model_kwargs_cd = model_kwargs.copy()
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})
            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            # model_kwargs = self._update_model_kwargs_for_generation(
            #     outputs,
            #     model_kwargs,
            #     is_encoder_decoder=self.config.is_encoder_decoder,
            # )
            if synced_gpus and this_peer_finished:
                continue

            # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
            
            ## FuzzyCD Algorithms
            membership_init = model_kwargs.get("membership_init")
            logits_mean = model_kwargs.get("logits_top_mean")
            logits_std = model_kwargs.get("logits_top_std")
            logits_min = model_kwargs.get("logits_top_min")
            logits_max = model_kwargs.get("logits_top_max")
            alpha = model_kwargs.get("cd_alpha")

            if membership_init:   
                # import pdb;pdb.set_trace()
                next_token_logits = outputs.logits[:, -1, :] 
                log_softmax=nn.functional.log_softmax(next_token_logits, dim=-1)

                logp_top1_token = log_softmax.argmax(dim=-1, keepdim=True)  # (1, 1) 형태의 인덱스
                top1_log_prob = log_softmax.gather(dim=-1, index=logp_top1_token)  # (1, 1) 형태

                # mean_= torch.mean(next_token_logits)
                return top1_log_prob

            if logits_mean != None:
                x_range, memfun = fuzzification(logits_mean, logits_std, logits_min, logits_max, device)
            
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
# outputs_1 = self(**model_inputs_1, return_dict=True, output_attentions=output_attentions_wo_img, output_hidden_states=output_hidden_states_wo_img,)
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
                    # next_token_logits_list.append(next_token_logits)

                    adjusted_logits,debug = ts_output(x_range, x_vals, memfun, next_token_logits, next_token_logits_cds, alpha)
                    ## cd_comments: pre-process logits from contrastive inputs
                    cd_beta = model_kwargs.get("cd_beta") if model_kwargs.get("cd_beta") is not None else 0.1
                

                    # version 2 set cutoff for Adaptive Plausibility Constraints
                    cutoff = torch.log(torch.tensor(cd_beta)) + next_token_logits.max(dim=-1, keepdim=True).values
                    
                    diffs = adjusted_logits
                    cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))
                    # print(cd_logits)

                    ## cd_comments: apply temperature warping and top-k filtering in contrastive decoding
                    cd_logits = logits_processor(input_ids, cd_logits)
                    # cd_logits = logits_warper(input_ids, cd_logits)

                    # mynext_token_logits_list.append(cd_logits)

                    # import pdb;pdb.set_trace()
                    _, next_tokens = torch.topk(cd_logits,1)
                    
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
                    # cd_logits = logits_warper(input_ids, cd_logits)

                    next_token_scores = cd_logits
                    cd_probs = nn.functional.softmax(cd_logits, dim=-1)
                    next_tokens = torch.multinomial(cd_probs, num_samples=1).squeeze(1)
            
            else:
                next_token_scores = logits_processor(input_ids, next_token_logits)    
                # next_token_scores = logits_warper(input_ids, next_token_scores)
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # pre-process distribution
            # next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
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

            # token selection
            # if do_sample:
            #     probs = nn.functional.softmax(next_token_scores, dim=-1)
            #     # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
            #     next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            # else:
            #     next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            if next_tokens.ndim == 1:
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            else:
                input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1
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
            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids #,mynext_token_logits_list,next_token_logits_list,debug

def evolve_vcd_sampling():
    transformers.generation.utils.GenerationMixin.sample = sample
    # sample is now a protected function in the latest Transformers library
    transformers.generation.utils.GenerationMixin._sample = sample
