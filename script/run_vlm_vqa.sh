#!/bin/bash


NUM_GPUS=2
NODE_RANK=0
NUM_NODES=1
MASTER_ADDR=localhost
MASTER_PORT=25677
# GPUs to be visible to the script
export CUDA_VISIBLE_DEVICES=0,2,3

# Run Python script with torch.distributed.launch for distributed training
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/run_vlm.py
