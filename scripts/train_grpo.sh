#!/usr/bin/env bash
# GRPO with task-accuracy + format rewards (external_acc, external_format).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

: "${MODEL_PATH:?Set MODEL_PATH to merged SFT checkpoint}"
: "${DATA_DIR:?Set DATA_DIR to directory containing rldata_*.json files}"

MAX_PIXELS="${MAX_PIXELS:-200704}"
OUTPUT_DIR="${OUTPUT_DIR:-output/grpo_cls}"

swift rlhf \
  --rlhf_type grpo \
  --model_type qwen2_5_vl \
  --model "${MODEL_PATH}" \
  --external_plugins grpo/plugin.py \
  --reward_funcs external_acc external_format \
  --train_type lora \
  --lora_rank 16 \
  --lora_alpha 64 \
  --target_modules all-linear \
  --freeze_aligner false \
  --freeze_vit false \
  --torch_dtype bfloat16 \
  --dataset \
    "${DATA_DIR}/rldata_CLS_ESCA_train.json" \
    "${DATA_DIR}/rldata_CLS_UniToPatho_train.json" \
  --max_length 2048 \
  --max_completion_length 1024 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --learning_rate 1e-5 \
  --gradient_accumulation_steps 1 \
  --save_strategy steps \
  --eval_strategy steps \
  --eval_steps 1000 \
  --save_steps 100 \
  --save_total_limit 1000 \
  --logging_steps 1 \
  --output_dir "${OUTPUT_DIR}" \
  --warmup_ratio 0.001 \
  --dataloader_num_workers 4 \
  --num_generations 8 \
  --temperature 1.0 \
  --system grpo/prompt.txt \
  --log_completions true \
  --num_iterations 1 \
  --num_infer_workers 2 \
  --async_generate false \
  --beta 0.001
