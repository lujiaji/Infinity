#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7
IMAGE_ROOT_PATH=$1
RESOLUTION=$2
PIC_NUM=${PIC_NUM:-4}
PROCESSES=${PROCESSES:-7}
PORT=${PORT:-29500}


accelerate launch --num_machines 1 --num_processes $PROCESSES --multi_gpu --mixed_precision "fp16" --main_process_port $PORT \
  /home/jiaji_lu/AR/Infinity/evaluation/DPG/compute_dpg_bench.py \
  --image-root-path $IMAGE_ROOT_PATH \
  --resolution $RESOLUTION \
  --pic-num $PIC_NUM \
  --vqa-model mplug

# python3 /home/jiaji_lu/AR/Infinity/evaluation/DPG/compute_dpg_bench.py --image-root-path /home/jiaji_lu/AR/Infinity/output/infinity_2b_evaluation/DPG_cfg3_tau1_cfg_insertion_layer0/images/dpg_images --resolution 512 --pic-num 4 --vqa-model mplug
