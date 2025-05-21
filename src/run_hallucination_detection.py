
import sys
sys.path.append('/data3/KJE/code/WIL_DeepLearningProject_2/NS3')

import os
import re
import json
import argparse
import random
import torch
from tqdm import tqdm
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from torch.nn.parallel import DistributedDataParallel as DDP


def load_data(args):

    adversarial_list = [json.loads(q) for q in open(os.path.join(args.data_root,args.data_name,args.data_name+'_pope_adversarial.json'), 'r')]
    popular_list = [json.loads(q) for q in open(os.path.join(args.data_root,args.data_name,args.data_name+'_pope_popular.json'), 'r')]
    random_list = [json.loads(q) for q in open(os.path.join(args.data_root,args.data_name,args.data_name+'_pope_random.json'), 'r')]

    return adversarial_list,popular_list,random_list


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/data3/KJE/code/WIL_DeepLearningProject_2/Hallu_MLLM/data/POPE')
    parser.add_argument('--img_data_root', type=str, default='/data3/KJE/code/WIL_DeepLearningProject_2/Hallu_MLLM/data/COCO/images/val2014')
    parser.add_argument('--data_name', type=str, default='coco')
    parser.add_argument('--output_root', type=str, default='/data3/KJE/code/WIL_DeepLearningProject_2/Hallu_MLLM/results')
    parser.add_argument('--model', type=str, default='llava-hf/llava-v1.6-mistral-7b-hf')

    # user options
    parser.add_argument('--label', type=str, default='exp0')
    parser.add_argument('--debug', default=False,action='store_true')
    parser.add_argument('--seed', type=int, default=10, help='random seed')
    
    #huggingface settings 
    parser.add_argument('--hg_token', type=str, default='')
    parser.add_argument('--cache_dir', type=str, default='/data3/hg_weight/hg_weight/')
    
    # Model settings
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_tokens',
                        type=int,
                        default=5,
                        help='The maximum number of tokens allowed for the generated answer.')
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    
    random.seed(args.seed)

    adversarial_list,popular_list,random_list = load_data(args)

    lists_dict = {
    "adversarial_list": adversarial_list,
    "popular_list": popular_list,
    "random_list": random_list
    }


    dictory=os.path.join(args.output_root,'Hallucination_Detection',args.model)
    if not os.path.exists(dictory):
        os.makedirs(dictory)
        print(f"디렉토리 생성 완료: {dictory}")


    processor = LlavaNextProcessor.from_pretrained(args.model, cache_dir=args.cache_dir)
    model = LlavaNextForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float16, use_flash_attention_2=True, cache_dir=args.cache_dir)
    model.to(local_rank)
    vlm_model = DDP(model, device_ids=[local_rank])

    
    first_hidden_embedding=[]
    second_hidden_embedding=[]
    last_hidden_embedding=[]



    for list_name, list_data in lists_dict.items():
        answer_list = []
        for item in tqdm(list_data, desc=f"Processing {list_name}"):
            question_id, image_name, text, label = item['question_id'],item['image'],item['text'],item['label']

            images = Image.open(os.path.join(args.img_data_root,image_name))

            prompt = f'<image>{text}'

            vlm_inputs = processor(prompt, images, return_tensors="pt")
            vlm_inputs = {k: v.to(local_rank) for k, v in vlm_inputs.items()}
            input_length = vlm_inputs['input_ids'].shape[1]

            outputs = model.generate(**vlm_inputs, max_new_tokens=5,output_hidden_states=True,return_dict_in_generate=True,pad_token_id=processor.tokenizer.pad_token_id)
            
            outputs_first_hidden_embeddings = [s[0][:,-1,:] for s in outputs.hidden_states]
            first_hidden_embedding.append(outputs_first_hidden_embeddings)
            
            outputs_second_hidden_embeddings = [s[1][:,-1,:] for s in outputs.hidden_states]
            second_hidden_embedding.append(outputs_second_hidden_embeddings)
        
            outputs_last_hidden_embeddings = [s[-1][:,-1,:] for s in outputs.hidden_states]
            last_hidden_embedding.append(outputs_last_hidden_embeddings)
                    
            
            answer = processor.decode(outputs[0][0][input_length:], skip_special_tokens=True).strip()

            if label == answer.lower():
                predict_label = 1
            else:
                predict_label = 0

            answer_list.append({"question_id":question_id,"question":text,"answer":answer,'predict_label':predict_label, 
                                'first_hidden_embedding':outputs_first_hidden_embeddings, 'second_hidden_embedding':outputs_second_hidden_embeddings,
                                'last_hidden_embedding': outputs_last_hidden_embeddings})

            
            break

