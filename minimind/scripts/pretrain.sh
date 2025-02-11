#!/bin/bash
cd ../
source ../.venv/bin/activate
# CUDA_VISIBLE_DEVICES=7 python train_pretrain.py --data_path /home/zxy/.cache/modelscope/hub/datasets/gongjy/minimind-dataset/pretrain_hq.jsonl --use_moe True
torchrun --nproc_per_node 8 train_pretrain.py --data_path /home/zxy/.cache/modelscope/hub/datasets/gongjy/minimind-dataset/pretrain_hq.jsonl --use_moe True --use_wandb