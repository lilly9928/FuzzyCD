import sys
sys.path.append('/data3/KJE/code/WIL_DeepLearningProject_2/HALLU_MLLM')
from datasets import load_dataset
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
import torch.distributed as dist

def dataloader(args):
    if args.data_name in ['pope','mmhal','whoops']:
        dataset = load_dataset(args.data_root, cache_dir=args.cache_dir)
        return dataset['test']

    elif args.data_name in ['pope']:
        dataset = [json.loads(q) for q in open(args.data_root, 'r')]
        
        return dataset
    

def load_example(args,data):
    if args.data_name == 'whoops':
        image =  data['image']
        return image,data
    
def save_results(args, output_list, append_mode=False):
    result_file_output = "{}/{}/{}/_{}.json".format(args.output_root, args.label, args.model,'output_list')

    write_mode = 'a' if append_mode else 'w' 

    with open(result_file_output, write_mode) as f:
        json.dump(output_list, f, indent=2, separators=(',', ': '))



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='nlphuji/whoops')
    parser.add_argument('--img_data_root', type=str, default='/data3/KJE/code/WIL_DeepLearningProject_2/Hallu_MLLM/data/COCO/images/val2014')
    parser.add_argument('--data_name', type=str, default='whoops')
    parser.add_argument('--output_root', type=str, default='/data3/KJE/code/WIL_DeepLearningProject_2/Hallu_MLLM/results')
    parser.add_argument('--model', type=str, default='llava-hf/llava-v1.6-mistral-7b-hf')
    parser.add_argument("--local-rank", type=int, default=0, help="Local rank for distributed training")
    
    # user options
    parser.add_argument('--label', type=str, default='hallu_check_orginal')
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
    
    torch.distributed.init_process_group(backend='nccl')
    local_rank = args.local_rank  # Use the parsed `local_rank`
    torch.cuda.set_device(local_rank)


    random.seed(args.seed)

    dictory=os.path.join(args.output_root,args.label,args.model)
    if not os.path.exists(dictory):
        os.makedirs(dictory)
        print(f"디렉토리 생성 완료: {dictory}")


    samples=dataloader(args)
    processor = LlavaNextProcessor.from_pretrained(args.model, cache_dir=args.cache_dir)
    model = LlavaNextForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float16, use_flash_attention_2=True, cache_dir=args.cache_dir)
    model.to(local_rank)
    vlm_model = DDP(model, device_ids=[local_rank])

    save_data=[]
    for idx,sample in enumerate(tqdm(samples)):
        image,data=load_example(args,sample)
        #TODO prompt 수정
        prompt = '<image>\n Question: Why is it weird? Answer:'

        vlm_inputs = processor(prompt, image, return_tensors="pt")
        vlm_inputs = {k: v.to(local_rank) for k, v in vlm_inputs.items()}
        input_length = vlm_inputs['input_ids'].shape[1]

        outputs = vlm_model.module.generate(
            **vlm_inputs, 
            max_new_tokens=100, 
            output_hidden_states=True,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            return_legacy_cache=True
        )


        logits = outputs['scores']
        output_sequence = []
        top_5_sequence = []
        product = torch.tensor(1.0, device=local_rank)
        count = 0
        
        for i in logits:
            pt = torch.softmax(torch.Tensor(i[0]), dim=0)
            max_loc = torch.argmax(pt)
            top_5_values, top_5_indices = torch.topk(pt, 5)
            if max_loc in processor("</s>")['input_ids'][0].to(local_rank):
                break
            else:
                # 각 단계에서 상위 5개 토큰 ID와 텍스트를 구성하여 저장
                top_5_text = [processor.decode([token_id.item()]).strip() for token_id in top_5_indices]
                top_5_data = [
                    {"Token ID": token_id.item(), "Text": text, "Logit": logit.item()}
                    for token_id, text, logit in zip(top_5_indices, top_5_text, top_5_values)
                ]
                
                top_5_sequence.append(top_5_data)
                output_sequence.append(max_loc)
                product *= torch.max(pt)
                count += 1

        output_text = processor.decode(output_sequence) if output_sequence else "FAIL"

        data.pop('image')
        save_data.append(data)
        save_data[idx]['predict_crowd_explanations'] = output_text
        save_data[idx]['top_5_data'] = top_5_data

    save_results(args, save_data,append_mode=True)
        



        
