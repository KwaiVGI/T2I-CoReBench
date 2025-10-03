#!/bin/bash

MODELS="Qwen-Image"
DIMENSION="C-MI, C-MA, C-MR, C-TR, R-LR, R-BR, R-CR, R-RR, C-MI, C-MA, C-MR, C-TR"

MLLM="Qwen3_VL_235B_Thinking"  # Qwen2_5_VL_72B, Qwen3_VL_235B_Thinking, Gemini_2_5_Flash

# start evaluation
python evaluate.py \
    --model "$MODELS" \
    --mllm "$MLLM" \
    --gen_eval_file "$DIMENSION"