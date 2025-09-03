#!/bin/bash

MODELS="Qwen-Image"
DIMENSION="C-MI, C-MA, C-MR, C-TR, R-LR, R-BR, R-CR, R-RR, C-MI, C-MA, C-MR, C-TR"

MLLM="Qwen2_5_VL_72B"  # Qwen2_5_VL_72B, Gemini_2_5_Flash

# start evaluation
python evaluate.py \
    --model "$MODELS" \
    --mllm "$MLLM" \
    --gen_eval_file "$DIMENSION"