#!/bin/bash

# Default parameter values
seed=${1:-55}
model_path=${4:-"liuhaotian/llava-v1.5-7b"}
cd_alpha=${5:-1}
cd_beta=${6:-0.2}
noise_step=${7:-500}


# Display which dataset and type are currently running
echo "Running with dataset: ${dataset_name}, type: ${type}"

# Run the Python command
CUDA_VISIBLE_DEVICES=0 python ./src/object_hallucination_vqa_llava_whoo.py \
  --model-path ${model_path} \
  --question-file /data3/KJE/code/WIL_DeepLearningProject_2/Hallu_MLLM/data/whoops/whoops.json \
  --answers-file ./output/llava15_whoops_answers_seed${seed}.jsonl \
  --cd_alpha $cd_alpha \
  --cd_beta $cd_beta \
  --noise_step $noise_step \
  --seed ${seed}

# Print completion message for this specific combination
echo "Completed dataset: ${dataset_name}, type: ${type}"


# Print final message
echo "All runs completed."
