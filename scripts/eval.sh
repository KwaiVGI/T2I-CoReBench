#!/bin/bash

MODELS="Qwen-Image"
DIMENSION="C-MI, C-MA, C-MR, C-TR, R-LR, R-BR, R-HR, R-PR, R-GR, R-AR, R-CR, R-RR"

MLLM="Qwen3_VL_30B_Thinking"  # Qwen2_5_VL_72B, Qwen3_VL_30B_Thinking, Gemini_2_5_Flash

# start evaluation
python evaluate.py \
    --model "$MODELS" \
    --mllm "$MLLM" \
    --gen_eval_file "$DIMENSION"