#!/usr/bin/env bash
# Supervised fine-tuning (SFT) for Qwen2.5-VL on pathology multimodal data.
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to Qwen2.5-VL-7B-Instruct (or your base checkpoint)}"
: "${DATA_DIR:?Set DATA_DIR to directory containing sftdata_*.json files}"

MAX_PIXELS="${MAX_PIXELS:-200704}"
OUTPUT_DIR="${OUTPUT_DIR:-output_sft}"

swift sft \
  --model_type qwen2_5_vl \
  --model "${MODEL_PATH}" \
  --attn_impl flash_attn \
  --train_type lora \
  --lora_rank 16 \
  --lora_alpha 64 \
  --target_modules all-linear \
  --freeze_aligner false \
  --freeze_vit false \
  --torch_dtype bfloat16 \
  --dataset \
    "${DATA_DIR}/sftdata_CLS_train.json" \
    "${DATA_DIR}/sftdata_DET_train.json" \
    "${DATA_DIR}/sftdata_VQA_train.json" \
  --num_train_epochs 10 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --learning_rate 1e-4 \
  --gradient_accumulation_steps 8 \
  --eval_steps 10000 \
  --save_steps 200 \
  --save_total_limit 200 \
  --logging_steps 5 \
  --max_length 1024 \
  --output_dir "${OUTPUT_DIR}" \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 8 \
  --dataset_num_proc 8 \
  --temperature 1.0 \
  --system 'You are a helpful assistant.'
