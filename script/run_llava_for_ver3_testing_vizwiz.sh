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
model_path=${4:-"liuhaotian/llava-v1.5-7b"}
cd_alpha=${5:-0.7}
cd_beta=${6:-0.2}
noise_step=${7:-500}

# types=("adversarial" "popular" "random")
# datasets=("coco" "aokvqa" "gqa")

# for debugging
types=("popular")
datasets=("coco")


# Types and dataset names
types=("random")
datasets=("vizwiz")
image_folder="/data3/KJE/Data/vqa/vizwiz/visual_question_answering/Images/train/train/"
echo "Running with dataset: vizwiz"
    #    --model-path ${model_path} \
    # Run the Python command
  CUDA_VISIBLE_DEVICES=0 python  ./src/object_hallucination_vqa_llava_ver3_vizwiz.py \
    --question-file /data3/KJE/Data/vqa/vizwiz/visual_question_answering/Annotations/train.json  \
    --model-path ${model_path} \
    --image-folder ${image_folder} \
    --answers-file ./output/final/llava15M_vizwiz_answers_seed${seed}_final_debug_2.jsonl \
    --cd_alpha $cd_alpha \
    --cd_beta $cd_beta \
    --noise_step $noise_step \
    --seed ${seed}\
  
    # Print completion message for this specific combination
echo "Completed dataset: vizwiz"

