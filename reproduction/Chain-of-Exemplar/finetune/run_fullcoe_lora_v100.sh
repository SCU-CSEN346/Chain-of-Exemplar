#!/bin/bash

MODEL="Qwen/Qwen-VL-Chat"
DATA="data/ScienceQA_train_multitask_fullcoe.json"
OUT="output/fullcoe_lora_v100_2ep_retrievalfix"

python finetune.py \
  --model_name_or_path $MODEL \
  --data_path $DATA \
  --fp16 True \
  --bf16 False \
  --fix_vit True \
  --output_dir $OUT \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 500 \
  --learning_rate 1e-5 \
  --logging_steps 10 \
  --model_max_length 1024 \
  --lazy_preprocess True \
  --gradient_checkpointing \
  --use_lora 
