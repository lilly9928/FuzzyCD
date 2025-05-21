#!/bin/bash

# Default parameter values
seed=${1:-55}
model_path=${4:-"liuhaotian/llava-v1.6-vicuna-7b"}
cd_alpha=${5:-1}
cd_beta=${6:-0.2}
noise_step=${7:-500}

dataset_name="llavabench"
# Display which dataset and type are currently running
echo "Running with dataset: ${dataset_name}, type: ${type}"

# Run the Python command
CUDA_VISIBLE_DEVICES=4 python ./src/yoon_llava_bench.py \
  --model-path ${model_path} \
  --answers-file ./output/llavabench/llava1.6_ours_answers_seed${seed}.jsonl \
  --cd_alpha $cd_alpha \
  --cd_beta $cd_beta \
  --filter_type 'gussian' \
  --noise_step $noise_step \
  --seed ${seed}

# Print completion message for this specific combination
echo "Completed dataset: ${dataset_name}, type: ${type}"


# Print final message
echo "All runs completed."
