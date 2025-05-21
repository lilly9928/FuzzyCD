#!/bin/bash

# Default parameter values
seed=${1:-55}
model_path=${2:-"liuhaotian/llava-v1.6-vicuna-7b"}
cd_alpha=${3:-1}
cd_beta=${4:-0.2}
noise_step=${5:-500}

# Dataset name (only using llavabench)
dataset_name="llavabench"

# Display which dataset is currently running
echo "Running with dataset: ${dataset_name}"

# Run the Python command
CUDA_VISIBLE_DEVICES=4 python ./src/object_hallucination_vqa_llava_bench.py \
  --model-path ${model_path} \
  --answers-file ./output/llavabench/llava16_${dataset_name}_answers_seed${seed}_2.jsonl \
  --filter_type median \
  --cd_alpha $cd_alpha \
  --cd_beta $cd_beta \
  --noise_step $noise_step \
  --seed ${seed}

# Print final message
echo "All runs completed."