import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math
import random
import numpy as np
# import kornia
from transformers import set_seed
from utils.image_filter import add_filter, add_filter_adjust_sigma
# from utils.sample_ver2 import evolve_vcd_sampling
# evolve_vcd_sampling()



def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        # breakpoint()
        conv.append_message(conv.roles[0], qs+"Please answer this question with one word.")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # print(prompt)

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
        image = Image.open(os.path.join(args.image_folder, image_file))
        
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        image_tensors = [image_tensor]
        image_tensors = torch.stack(image_tensors, dim=0)
       
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        
        with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    attention_mask = attention_mask, 
                    pad_token_id = model.config.eos_token_id,
                    images = image_tensors.half().cuda(),
                    # images_1=(image_tensors[0].unsqueeze(0).half().cuda() if image_tensors is not None else None),
                    # images_2=(image_tensors[1].unsqueeze(0).half().cuda() if image_tensors is not None else None),
                    # images_3=(image_tensors[2].unsqueeze(0).half().cuda() if image_tensors is not None else None),
                    # images_4=(image_tensors[3].unsqueeze(0).half().cuda() if image_tensors is not None else None),
                    cd_alpha = args.cd_alpha,
                    cd_beta = args.cd_beta,
                    do_sample=True,
                    membership_init = False,
                    temperature = args.temperature,
                    top_p = args.top_p,
                    top_k = args.top_k,
                    max_new_tokens = 2,
                    use_cache = True)

        # print(output_ids)
        input_token_len = input_ids.shape[1]
        # print(input_token_len)
        # print(output_ids[:, :input_token_len:])
    
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        # print(outputs)
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
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
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)

    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--filter_type", type=str, default='sharpen', help="Filter type", choices=["FLL+gussian", "FLL+median", "FLL+sharpen", "FLL+sobel", "FLL+bilateral", "FLL+diffusion", "gaussian", "median", "sharpen", "sobel", "diffusion"])
    parser.add_argument("--use_cd", action='store_true', default=False)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--sigma", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
