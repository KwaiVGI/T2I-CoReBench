#!/bin/bash

MODELS=("Qwen-Image")
DIMENSION="C-MI, C-MA, C-MR, C-TR, R-LR, R-BR, R-HR, R-PR, R-GR, R-AR, R-CR, R-RR"

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
