seed=${1:-55}
model_path=${4:-"Salesforce/instructblip-vicuna-7b"}
cd_alpha=${5:-1}
cd_beta=${6:-0.2}
noise_step=${7:-500}

types=("random")
datasets=("chair")

# Loop over dataset names
for dataset_name in "${datasets[@]}"; do
  # Determine image folder based on dataset name
  if [[ $dataset_name == 'coco' || $dataset_name == 'aokvqa' ]]; then
    image_folder="/data3/KJE/code/WIL_DeepLearningProject_2/Hallu_MLLM/data/COCO/images/val2014"
  else
    image_folder="/data3/KJE/code/WIL_DeepLearningProject_2/Hallu_MLLM/base/ICD/experiments/data/chair/chair-500/"
  fi
  # Loop over types
  for type in "${types[@]}"; do
    # Display which dataset, type, and sigma are currently running
    echo "Running with dataset: ${dataset_name}, type: ${type}"
    
    # Run the Python command
    CUDA_VISIBLE_DEVICES=3 python /data3/KJE/code/WIL_DeepLearningProject_2/Hallu_MLLM/src/object_hallucination_vqa_instructblip_Fuzzy_chair_yj.py  \
      --question-file /data3/KJE/code/WIL_DeepLearningProject_2/Hallu_MLLM/base/ICD/experiments/data/chair/chair-500.jsonl \
      --image-folder ${image_folder} \
      --answers-file ./output/final/linstructblip_${dataset_name}-32.jsonl \
      --cd_alpha $cd_alpha \
      --cd_beta $cd_beta \
      --noise_step $noise_step \
      --seed ${seed} \

    # Print completion message for this specific combination
    echo "Completed dataset: ${dataset_name}, type: ${type}"
  done
done
