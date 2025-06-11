#!/bin/bash
GPU=$1
q_bits=$2
q_dim=$3

# 切换到脚本所在目录
cd /home/jiaji_lu/AR/Infinity/ljj_pl/

# 创建log目录（如果不存在）
mkdir -p log/

CUDA_VISIBLE_DEVICES=$GPU nohup python pipeline.py --q_bits $q_bits --q_dim $q_dim > log/q_bits_${q_bits}_q_dim_${q_dim}.log 2>&1 &

# ./ljj_pl/run_with_Q.sh 0 4 per-head+per-dim
# ./ljj_pl/run_with_Q.sh 1 8 per-head+per-dim