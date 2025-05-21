#!/bin/bash

seed=${1:-55}
model_path=${4:-"liuhaotian/llava-v1.5-7b"}
cd_alpha=${5:-0.7}
cd_beta=${6:-0.2}
noise_step=${7:-500}

types=("popular" "random" "adversarial")
ratios=(0.1 0.3 0.5 1.0)
datasets=("coco")

for dataset_name in "${datasets[@]}"; do
  if [[ $dataset_name == 'coco' || $dataset_name == 'aokvqa' ]]; then
    image_folder="/data3/KJE/code/WIL_DeepLearningProject_2/Hallu_MLLM/data/COCO/images/val2014"
  else
    image_folder="/data3/KJE/code/WIL_DeepLearningProject_2/Hallu_MLLM/data/GQA/images/images"
  fi

  for type in "${types[@]}"; do
    for ratio in "${ratios[@]}"; do
      echo "Running with dataset: ${dataset_name}, type: ${type}, ratio: ${ratio}"

      CUDA_VISIBLE_DEVICES=1 python ./src/object_hallucination_vqa_llava_ver2_ablation_mf.py \
        --model-path ${model_path} \
        --question-file /data3/KJE/code/WIL_DeepLearningProject_2/Hallu_MLLM/data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
        --image-folder ${image_folder} \
        --answers-file ./output/membership_ab/llava15M_${dataset_name}_pope_${type}_ratio${ratio}_answers_seed${seed}_debuging_log_softmax_jsd_final_rebuttal_ablation_mf.jsonl \
        --cd_alpha $cd_alpha \
        --cd_beta $cd_beta \
        --noise_step $noise_step \
        --seed ${seed} \
        --data ${type} \
        --sampling_ratio ${ratio}  

      echo "Completed dataset: ${dataset_name}, type: ${type}, ratio: ${ratio}"
    done
  done
done
