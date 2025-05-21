# '''
# 더 큰 모델 돌리기 
# '''

# CUDA_VISIBLE_DEVICES=4 python ./src/object_hallucination_vqa_llava_visualization.py \
#   --model-path liuhaotian/llava-v1.5-7b \
#   --question-file /data3/KJE/code/WIL_DeepLearningProject_2/Hallu_MLLM/data/whoops/test.json \
#   --answers-file ./output/llava1.5_dddd_whoops_answers_seed55_jm.jsonl \
#   --cd_alpha 0.7 \
#   --cd_beta 0.2 \
#   --filter_type 'sharpen' \
#   --noise_step 500 \
#   --seed 55

seed=${1:-55}
# model_path=${4:-"liuhaotian/llava-v1.5-7b"}
model_path=${4:-"Qwen/Qwen2.5-VL-72B-Instruct"}
cd_alpha=${5:-0.7}
cd_beta=${6:-0.2}
noise_step=${7:-500}

types=("popular" "adversarial" "random")
# types=("popular" "adversarial")
# types=("popular")
datasets=("coco")
# datasets=("coco" "aokvqa" "gqa")
# for debugging
# types=("random")
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
    # Display which dataset, type, and sigma are currently running
    echo "Running with dataset: ${dataset_name}, type: ${type}"
    
    # Run the Python command
    CUDA_VISIBLE_DEVICES=0,3,4 python ./src/object_hallucination_vqa_qwenvl.py \
      --model-path ${model_path} \
      --question_file /data3/KJE/code/WIL_DeepLearningProject_2/Hallu_MLLM/data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
      --image-folder ${image_folder} \
      --answers-file ./output/final/qwen25vl_${dataset_name}_pope_${type}_answers_seed${seed}_test_sobel.jsonl \
      --cd_alpha $cd_alpha \
      --cd_beta $cd_beta \
      --noise_step $noise_step \
      --seed ${seed} \
      --bf16 \

    # Print completion message for this specific combination
    echo "Completed dataset: ${dataset_name}, type: ${type}"
  done
done
