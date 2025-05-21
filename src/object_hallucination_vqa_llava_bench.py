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
from datasets import load_dataset
from PIL import Image
import math
import random
import numpy as np
import shortuuid
# import kornia
from transformers import set_seed
from utils.image_filter import add_filter
from utils.sample import evolve_vcd_sampling

def eval_model(args):
    disable_torch_init()
    model_path = args.model_path
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # Load dataset from Hugging Face
    dataset = load_dataset("lmms-lab/llava-bench-in-the-wild", cache_dir="/data3/KJE/code/WIL_DeepLearningProject_2/Hallu_MLLM/data/LLaVA_Bench")
    dataset = dataset['train']

    sample_size = int(len(dataset) * 0.1)
    sampled_questions = random.sample(list(dataset), sample_size)

    logits_means = []
    for line in tqdm(sampled_questions):
        idx = line["question_id"]
        image = line["image"]
        qs = line["question"]

        # 모델에 대한 질문 형식 설정
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs + " Please answer this question with one word.")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # 토크나이저로 입력 생성
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        attention_mask = input_ids.ne(tokenizer.pad_token_id).long()

        # 이미지 전처리
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        if args.use_cd:
            image_tensors = []
            filter_images = add_filter(args, image_file=image)
            for filter_image in filter_images:
                image_tensors.append(image_processor.preprocess(filter_image, return_tensors='pt')['pixel_values'][0])
        else:
            image_tensors = None

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        # 모델 출력 생성
        with torch.inference_mode():
            output_logits = model.generate(
                input_ids,
                attention_mask=attention_mask,
                pad_token_id=model.config.eos_token_id,
                images=image_tensor.unsqueeze(0).half().cuda(),
                images_1=(image_tensors[0].unsqueeze(0).half().cuda() if image_tensors else None),
                images_2=(image_tensors[1].unsqueeze(0).half().cuda() if image_tensors else None),
                images_3=(image_tensors[2].unsqueeze(0).half().cuda() if image_tensors else None),
                images_4=(image_tensors[3].unsqueeze(0).half().cuda() if image_tensors else None),
                cd_alpha=args.cd_alpha,
                cd_beta=args.cd_beta,
                membership_init=True,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=1024,
                use_cache=True
            )

            logits_means.append(output_logits.float().mean().item())

    logits_means_mean, logits_means_std, logits_min, logits_max = np.mean(logits_means), np.std(logits_means, ddof=1), np.min(logits_means), np.max(logits_means)

    # 모델 출력 저장
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    with open(answers_file, "w") as ans_file:
        for line in tqdm(dataset):
            idx = line["question_id"]
            image = line["image"]
            qs = line["question"]
            image_id = line["image_id"]
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            
            if args.use_cd:
                image_tensors = []
                filter_images = add_filter(args, image_file=image)
                for filter_image in filter_images:
                    image_tensors.append(image_processor.preprocess(filter_image, return_tensors='pt')['pixel_values'][0])
            else:
                image_tensors = None

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=model.config.eos_token_id,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    images_1=(image_tensors[0].unsqueeze(0).half().cuda() if image_tensors else None),
                    images_2=(image_tensors[1].unsqueeze(0).half().cuda() if image_tensors else None),
                    images_3=(image_tensors[2].unsqueeze(0).half().cuda() if image_tensors else None),
                    images_4=(image_tensors[3].unsqueeze(0).half().cuda() if image_tensors else None),
                    cd_alpha=args.cd_alpha,
                    cd_beta=args.cd_beta,
                    do_sample=True,
                    membership_init=False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_new_tokens=1024,
                    logits_mean=logits_means_mean,
                    logits_std=logits_means_std,
                    logits_min=logits_min,
                    logits_max=logits_max,
                    use_cache=True
                )

            outputs = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)].strip()

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx, "prompt": qs, "text": outputs, "model_id": model_name,"answer_id": ans_id, "image_id": image_id, "metadata": {}}) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--filter_type", type=str, default='gaussian')
    parser.add_argument("--use_cd", action='store_true')
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)