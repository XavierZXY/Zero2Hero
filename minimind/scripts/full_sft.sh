#!/bin/bash
cd ../
source ../.venv/bin/activate
# CUDA_VISIBLE_DEVICES=7 python train_pretrain.py --data_path /home/zxy/.cache/modelscope/hub/datasets/gongjy/minimind-dataset/pretrain_hq.jsonl --use_moe True
# TODO: Use config file to specify 
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 \
  train_full_sft.py \
  --epochs 4\
  --batch_size 128\
  --data_path /home/zxy/.cache/modelscope/hub/datasets/gongjy/minimind-dataset/sft_mini_512.jsonl \
  --use_wandb \
