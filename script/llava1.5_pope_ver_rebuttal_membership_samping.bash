#!/bin/bash

# 기본 seed (외부에서 안 주면 55)
seed_base=${1:-55}
# model_path, cd_alpha 등은 그대로
model_path=${4:-"liuhaotian/llava-v1.5-7b"}
cd_alpha=${5:-0.7}
cd_beta=${6:-0.2}
noise_step=${7:-500}

# 반복 횟수
num_reps=10

types=("popular" "random" "adversarial")
ratios=(0.1)
datasets=("coco")

for dataset_name in "${datasets[@]}"; do
  if [[ $dataset_name == 'coco' || $dataset_name == 'aokvqa' ]]; then
    image_folder="/data3/KJE/code/WIL_DeepLearningProject_2/Hallu_MLLM/data/COCO/images/val2014"
  else
    image_folder="/data3/KJE/code/WIL_DeepLearningProject_2/Hallu_MLLM/data/GQA/images/images"
  fi

  for type in "${types[@]}"; do
    for ratio in "${ratios[@]}"; do

      for rep in $(seq 1 $num_reps); do
        # 각 반복마다 seed를 base+offset으로 설정
        seed=$(( seed_base + rep - 1 ))

        echo "Running dataset=${dataset_name}, type=${type}, ratio=${ratio}, run=${rep}/${num_reps}, seed=${seed}"

        CUDA_VISIBLE_DEVICES=2 python ./src/object_hallucination_vqa_llava_ver2_ablation_mf.py \
          --model-path "${model_path}" \
          --question-file "/data3/KJE/code/WIL_DeepLearningProject_2/Hallu_MLLM/data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json" \
          --image-folder "${image_folder}" \
          --answers-file "./output/membership_ab/llava15M_${dataset_name}_pope_${type}_ratio${ratio}_seed${seed}_run${rep}_debugging_log_softmax_jsd_final_rebuttal_ablation_mf.jsonl" \
          --cd_alpha "${cd_alpha}" \
          --cd_beta "${cd_beta}" \
          --noise_step "${noise_step}" \
          --seed "${seed}" \
          --data "${rep}_${num_reps}" \
          --sampling_ratio "${ratio}"

        echo "Completed run ${rep}/${num_reps} for dataset=${dataset_name}, type=${type}, ratio=${ratio}"
      done

    done
  done
done
rrr