#!/bin/bash
cd ../
source ../.venv/bin/activate
# CUDA_VISIBLE_DEVICES=7 python train_pretrain.py --data_path /home/zxy/.cache/modelscope/hub/datasets/gongjy/minimind-dataset/pretrain_hq.jsonl --use_moe True
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port 30000 \
  train_full_sft.py \
    --epochs 6\
  --batch_size 64\
  --data_path /home/zxy/.cache/modelscope/hub/datasets/gongjy/minimind_dataset/sft_mini_512.jsonl \
  --use_moe True \
  --use_wandb
