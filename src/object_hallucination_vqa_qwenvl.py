
import os
import sys
import json
import math
import random
import argparse
import shortuuid
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.sample_qwen import evolve_vcd_sampling
evolve_vcd_sampling()

from PIL import Image
from tqdm import tqdm

# import kornia
import torch
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM, AutoProcessor, Qwen2VLImageProcessor
# from transformers import (
#     Qwen2VLForConditionalGeneration,
#     Qwen2_5_VLForConditionalGeneration,
# )

from Qwen_VL2.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from utils.image_filter import add_filter, add_filter_adjust_sigma
# from utils.qwen_processor import Qwen2_5_VLProcessor

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

# def membership_init(args,questions,model,tokenizer,image_processor):
def membership_init(args, questions, model, processor):
    args.use_FLL = True
    sample_size = int(len(questions) * 0.1) 
    sampled_questions = random.sample(questions, sample_size)
    image_processor = processor.image_processor

    logits_means = []
    for line in tqdm(sampled_questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": os.path.join(args.image_folder, image_file),
                    },
                    {"type": "text", "text": qs+"Please answer this question with one word."},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        # inputs = processor(
        #     text=[text],
        #     images=image_inputs,
        #     videos=video_inputs,
        #     padding=True,
        #     return_tensors="pt",
        # )
        # inputs = inputs.to(model.device)
        # # image_tensor = inputs['pixel_values']
        # if args.use_cd:
        #     image_tensors = [image_inputs[0]]
        #     if args.use_FLL:
        #         filter_images = add_filter(args,image_file=image_file, image_processor=image_processor)
        #     else:
        #         filter_images = add_filter_adjust_sigma(args,image_file=image_file, sigma=args.sigma, image_processor=image_processor)
        #     for filter_image in filter_images:
        #         if "diffusion" in args.filter_type:
        #             image_tensors.append(filter_image.to(model.device))
        #         else:
        #             # image_tensors.append(image_processor.preprocess(filter_image, return_tensors='pt')['pixel_values'].to(model.device))
        #             image_tensors.append(filter_image)
        #     # image_tensors = torch.stack(image_tensors, dim=0)
        # else:
        #     image_tensor_cd = None   
        
        # inputs = processor(
        #     text=[text for i in range(5)],
        #     images=image_tensors,
        #     videos=video_inputs,
        #     padding=True,
        #     return_tensors="pt",
        # )
        inputs = processor(
            text=[text],
            images=image_inputs[0],
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        input_ids, attention_mask, pixel_values, image_grid_thw = inputs['input_ids'], inputs['attention_mask'], inputs['pixel_values'], inputs['image_grid_thw']
        # pixel_values = pixel_values.reshape(5, -1, pixel_values.size(-1))
        with torch.inference_mode():
            output_logits = model.generate(
                # **inputs,
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                pad_token_id=model.config.eos_token_id,
                cd_alpha = args.cd_alpha,
                cd_beta = args.cd_beta,
                membership_init = True,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=5,
                use_cache=True,
                use_cd=args.use_cd,
                use_FLL=args.use_FLL)
            
            mean=output_logits
            logits_means.append(mean.item())
    
    logits_means_mean = np.mean(logits_means)
    logits_min = np.min(logits_means)
    logits_max = np.max(logits_means)
    logits_means_std = np.std(logits_means,ddof=1)    

    return logits_means_mean, logits_means_std, logits_min, logits_max

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = 'qwen2.5'
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            cache_dir=args.cache_dir,
            attn_implementation=args.attn_implementation,
            torch_dtype=(torch.bfloat16 if args.bf16 else None),
            device_map='auto',
            low_cpu_mem_usage=False,
    ).eval()
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    args.use_FLL = False
    ## for membership function
    if args.filter_type.startswith('FLL'):
        args.use_FLL = True
        # logits_means_mean, logits_means_std, logits_min, logits_max = membership_init(args,questions,model,tokenizer,image_processor)
        logits_means_mean, logits_means_std, logits_min, logits_max = membership_init(args, questions, model, processor)
    else:
        logits_means_mean = None
        logits_means_std = None
        logits_min = None
        logits_max = None
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    image_processor = processor.image_processor
    tokenizer = processor.tokenizer

    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs+"Please answer this question with one word."
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": os.path.join(args.image_folder, image_file),
                    },
                    {"type": "text", "text": qs+"Please answer this question with one word."},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        if args.use_cd:
            image_tensors = [image_inputs[0]]
            if args.use_FLL:
                filter_images = add_filter(args,image_file=image_file, image_processor=image_processor)
            else:
                filter_images = add_filter_adjust_sigma(args,image_file=image_file, sigma=args.sigma, image_processor=image_processor)
            for filter_image in filter_images:
                if "diffusion" in args.filter_type:
                    image_tensors.append(filter_image.to(model.device))
                else:
                    # image_tensors.append(image_processor.preprocess(filter_image, return_tensors='pt')['pixel_values'].to(model.device))
                    image_tensors.append(filter_image)
            # image_tensors = torch.stack(image_tensors, dim=0)
        else:
            image_tensors = [image_inputs[0]]
        
        image_tensors = processor(
            text=[text for i in range(len(image_tensors))],
            images=image_tensors,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )['pixel_values']
        image_tensors = image_tensors.reshape(len(image_tensors), -1, image_tensors.size(-1))
        inputs = processor(
            text=[text],
            images=image_inputs[0],
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        input_ids, attention_mask, pixel_values, image_grid_thw = inputs['input_ids'], inputs['attention_mask'], inputs['pixel_values'], inputs['image_grid_thw']
        
        with torch.inference_mode():
                output_ids = model.generate(
                # **inputs,
                input_ids=input_ids,
                images=image_tensors,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                pad_token_id=model.config.eos_token_id,
                cd_alpha = args.cd_alpha,
                cd_beta = args.cd_beta,
                membership_init = False,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=5,
                logits_top_mean = logits_means_mean,
                logits_top_std = logits_means_std,
                logits_top_min = logits_min,
                logits_top_max = logits_max,
                use_cache=True,
                use_cd=args.use_cd,
                use_FLL=args.use_FLL)
                output_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, output_ids)
                ]
                # output_ids, mynext_token_logits_list, next_token_logits_list, _ = model.generate(
                #     input_ids=input_ids,
                #     attention_mask = attention_mask, 
                #     pad_token_id = model.config.eos_token_id,
                #     images = image_tensors.half().cuda(),
                #     cd_alpha = args.cd_alpha,
                #     cd_beta = args.cd_beta,
                #     do_sample=True,
                #     membership_init = False,
                #     temperature = args.temperature,
                #     top_p = args.top_p,
                #     top_k = args.top_k,
                #     max_new_tokens = 5,
                #     logits_top_mean = logits_means_mean,
                #     logits_top_std = logits_means_std,
                #     logits_top_min = logits_min,
                #     logits_top_max = logits_max,
                #     use_cache = True,
                #     use_cd = args.use_cd,
                #     use_FLL = args.use_FLL)

        # print(output_ids)
        # input_token_len = input_ids.shape[1]
        # print(input_token_len)
        # print(output_ids[:, :input_token_len:])
    
        # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        # if n_diff_input_output > 0:
        #     print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        # outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        # outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = processor.batch_decode(
            output_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        # print(outputs)
        outputs = outputs.strip()
        if outputs.endswith('<|im_end|>'):
            outputs = outputs[:-len('<|im_end|>')]
        elif outputs.endswith('<|endoftext|>'):
            outputs = outputs[:-len('<|endoftext|>')]
            
        outputs = outputs.strip()

        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "model_id": model_name,
                                   "image": image_file,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question_file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)

    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--filter_type", type=str, default='sobel', help="Filter type", choices=["FLL+gussian", "FLL+median", "FLL+sharpen", "FLL+sobel", "FLL+bilateral", "FLL+diffusion", "gaussian", "median", "sharpen", "sobel", "diffusion"])
    parser.add_argument("--use_cd", type=bool, default=False)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--sigma", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--bf16", action='store_true', help="using bf16")
    parser.add_argument("--cache_dir", type=str, default="/data3/hg_weight/hg_weight")

    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
