dataset_name="chair"

type="random"
# type="popular"
# type="adversarial"

model="llava15"
# model="instructblip"

coco_path=/data3/KJE/code/WIL_DeepLearningProject_2/Hallu_MLLM/base/ICD/experiments/data/chair/MSCOCO/annotation/annotations
seed=55


# python chair.py \
# --coco_path CHAIR-eval/MSCOCO/annotations \
# --cache CHAIR-eval/data/chair.pkl \
# --cap_file $MODEL_NAME/answer-chair.jsonl \
# --save_path $MODEL_NAME/eval-chair.json

# --cap_file /data3/KJE/code/WIL_DeepLearningProject_2/Hallu_MLLM/output/final/llava15M_chair_prompt_max128.jsonl \
python /data3/KJE/code/WIL_DeepLearningProject_2/Hallu_MLLM/src/eval/eval_chair_.py \
--coco_path ${coco_path} \
--cap_file /data3/KJE/code/WIL_DeepLearningProject_2/Hallu_MLLM/base/ICD/experiments/JE_Result/chair-500-2048/llava/icd/0/normal.json \
--save_path /data3/KJE/code/WIL_DeepLearningProject_2/Hallu_MLLM/base/ICD/experiments/data/chair/MSCOCO/annotation/annotations/captions_val2014.json \
--image_id_key image \
--caption_key text

# --save_path /data3/KJE/code/WIL_DeepLearningProject_2/Hallu_MLLM/output/final/${MODEL_NAME}_eval_chair_prompt.json \
# --save_path /data3/KJE/code/WIL_DeepLearningProject_2/Hallu_MLLM/output/final/${MODEL_NAME}_eval_chair_prompt_max30.json \
# --save_path /data3/KJE/code/WIL_DeepLearningProject_2/Hallu_MLLM/output/final/${MODEL_NAME}_eval_chair_prompt_max1024.json \
# python ./experiments/eval/chair.py \
# --cap_file ./experiments/output/${dataset_name}/llava15_${dataset_name}_pope_${type}_answers_baseline_seed${seed}.jsonl --image_id_key image_id --caption_key caption \
# --coco_path ${coco_path}

# python ./experiments/eval/chair.py \
# --cap_file ./experiments/output/${dataset_name}/llava15_${dataset_name}_pope_${type}_answers_cd_seed${seed}.jsonl --image_id_key image_id --caption_key caption \
# --coco_path ${coco_path}



# python ./experiments/eval/chair.py \
# --cap_file ./experiments/output/${dataset_name}/llava15_${dataset_name}_pope_${type}_answers_icd_seed${seed}.jsonl --image_id_key image_id --caption_key caption \
# --coco_path ${coco_path}

# python ./experiments/eval/chair.py \
# --cap_file ./experiments/output/${dataset_name}/llava15_${dataset_name}_pope_${type}_answers_imccd_seed${seed}.jsonl --image_id_key image_id --caption_key caption \
# --coco_path ${coco_path}