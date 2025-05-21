seed=${1:-55}
model_path=${4:-"liuhaotian/llava-v1.5-7b"}
cd_alpha=${5:-0.7}
cd_beta=${6:-0.2}
noise_step=${7:-500}

# for debugging
types=("random")
datasets=("mmbench")
# Loop over dataset names
for dataset_name in "${datasets[@]}"; do

  image_folder="/data3/KJE/code/WIL_DeepLearningProject_2/Hallu_MLLM/data/LMUData/images"
  # Loop over types
  for type in "${types[@]}"; do
    # Display which dataset, type, and sigma are currently running
    echo "Running with dataset: ${dataset_name}, type: ${type}"
    
    # Run the Python command
    CUDA_VISIBLE_DEVICES=0 python ./src/object_hallucination_vqa_llava_mmbench.py \
      --model-path ${model_path} \
      --question-file /data3/KJE/code/WIL_DeepLearningProject_2/Hallu_MLLM/data/MMbench/dev_split_json/filtered_mmb_dev_merged.json \
      --image-folder ${image_folder} \
      --answers-file ./output/test/llava15M_${dataset_name}_pope_${type}_answers_seed${seed}_diffusion_noj.jsonl \
      --cd_alpha $cd_alpha \
      --cd_beta $cd_beta \
      --noise_step $noise_step \
      --seed ${seed} \

    # Print completion message for this specific combination
    echo "Completed dataset: ${dataset_name}, type: ${type}"
  done
done
