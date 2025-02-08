#!/bin/bash
source ../.venv/bin/activate

CUDA_VISIBLE_DEVICES=6,7 python train.py