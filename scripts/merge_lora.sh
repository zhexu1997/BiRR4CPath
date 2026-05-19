#!/usr/bin/env bash
# Merge LoRA adapters into base weights for downstream GRPO or inference.
set -euo pipefail

: "${ADAPTER_PATH:?Set ADAPTER_PATH to SFT/GRPO checkpoint directory with LoRA weights}"

swift export --adapters "${ADAPTER_PATH}" --merge_lora true
