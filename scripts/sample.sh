#!/bin/bash

MODELS=("Qwen-Image")
DIMENSION="C-MI, C-MA, C-MR, C-TR, R-LR, R-BR, R-CR, R-RR, C-MI, C-MA, C-MR, C-TR"

GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

for MODEL in "${MODELS[@]}"; do

    # generate images
    torchrun \
        --nnodes=1 \
        --node_rank=0 \
        --nproc_per_node=$GPUS \
        --master_addr=127.0.0.1 \
        --master_port=12138 \
        sample.py \
        --model "$MODEL" \
        --gen_eval_file "$DIMENSION"

done
