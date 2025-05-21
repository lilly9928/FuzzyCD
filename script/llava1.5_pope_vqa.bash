#!/bin/bash

# Default parameter values
seed=${1:-55}
model_path=${4:-"liuhaotian/llava-v1.5-7b"}
cd_alpha=${5:-1}  # Uncommented this line to define cd_alpha
cd_beta=${6:-0.2}
noise_step=${7:-500}

# Types and dataset names
datasets=("aokvqa")

# Loop over dataset names
for dataset_name in "${datasets[@]}"; do
  # Determine image folder based on dataset name
  if [[ $dataset_name == 'coco' || $dataset_name == 'aokvqa' ]]; then
    image_folder="/data3/KJE/code/WIL_DeepLearningProject_2/Hallu_MLLM/data/COCO/images/val2014"
  else
    image_folder="/data3/KJE/code/WIL_DeepLearningProject_2/Hallu_MLLM/data/GQA/images/images"
  fi

  # Run the Python script with the parameters
  CUDA_VISIBLE_DEVICES=1,2 python ./src/object_hallucination_vqa.py \
        --model-path ${model_path} \
        --question-file /data3/KJE/code/WIL_DeepLearningProject_2/Hallu_MLLM/data/aokvqa/test.json \
        --image-folder ${image_folder} \
        --answers-file ./output/llava15_gussian_${dataset_name}_aokvqa_answers_seed${seed}_alpha_${cd_alpha}.jsonl \
        --filter_type gussian \
        --cd_alpha $cd_alpha \
        --cd_beta $cd_beta \
        --noise_step $noise_step \
        --seed ${seed}
done
