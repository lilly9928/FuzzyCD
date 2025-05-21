#!/bin/bash

# Default parameter values
seed=${1:-55}
model_path=${4:-"Salesforce/instructblip-vicuna-7b"}
cd_alpha=${5:-1}
cd_beta=${6:-0.2}
noise_step=${7:-500}

# Types and dataset names
types=("adversarial" "popular" "random")
datasets=("coco")
# types=("adversarial")
# datasets=("coco")

# Loop over dataset names
for dataset_name in "${datasets[@]}"; do
  # Determine image folder based on dataset name
  if [[ $dataset_name == 'coco' || $dataset_name == 'aokvqa' ]]; then
    image_folder="/data3/KJE/code/WIL_DeepLearningProject_2/Hallu_MLLM/data/COCO/images/val2014"
  else
    image_folder="/data3/KJE/code/WIL_DeepLearningProject_2/Hallu_MLLM/data/GQA/images/images"
  fi
  
  # Loop over types
  for type in "${types[@]}"; do
    # Display which dataset and type are currently running
    echo "Running with dataset: ${dataset_name}, type: ${type}"
    
    # Run the Python command
    CUDA_VISIBLE_DEVICES=4 python ./src/object_hallucination_vqa_instructblip_Fuzzy.py \
      --question-file /data3/KJE/code/WIL_DeepLearningProject_2/Hallu_MLLM/data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
      --image-folder ${image_folder} \
      --answers-file ./output/final/instructblip_Fuzzy_${dataset_name}_pope_${type}_answers_seed${seed}_final_Sobal_again.jsonl \
      --cd_alpha $cd_alpha \
      --cd_beta $cd_beta \
      --noise_step $noise_step \
      --seed ${seed}

    # Print completion message for this specific combination
    echo "Completed dataset: ${dataset_name}, type: ${type}"
  done
done

# Print final message
echo "All runs completed."
