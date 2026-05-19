# BiRR4CPath

Official implementation for **Efficient Multimodal Computational Pathology via Bilateral Reinforcement Reasoning** .

BiRR4CPath fine-tunes **Qwen2.5-VL** on computational pathology tasks with [ms-swift](https://github.com/modelscope/ms-swift) **v3.4**: supervised fine-tuning (SFT), LoRA merge, GRPO with pathology-aware rewards, and two-stage LoRA inference (token allocation → task reasoning).

## Features

- **Multimodal pathology tasks**: classification (CLS), detection (DET), and VQA
- **GRPO rewards** via `grpo/plugin.py`:
  - `external_acc` — task accuracy (choice / detection AP / open BLEU)
  - `external_format` — enforces `<think>` + `<answer>` structure
  - `external_acc_token` — vision-token budget allocation with a task-performer model
- **Data scripts** to convert SFT JSON → RL JSON and token-allocation format

## Repository layout

```
BiRR4CPath/
├── grpo/
│   ├── plugin.py          # GRPO reward plugin (ms-swift ORM)
│   └── prompt.txt         # System prompt for GRPO
├── scripts/
│   ├── build_rldata_from_sft.py   # SFT → RL JSON
│   ├── build_rldata_token_alloc.py    # RL → token-allocation JSON
│   ├── train_sft.sh
│   ├── merge_lora.sh
│   ├── train_grpo.sh
│   └── train_grpo_token.sh
├── docs/
│   └── DATA_FORMAT.md
├── examples/
│   └── sample_rl_record.json
├── infer.py                   # Two-stage LoRA inference
├── requirements.txt
└── .env.example
```

## Installation

```bash
cd BiRR4CPath
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# ms-swift is pinned to v3.4.x (see requirements.txt). To install a specific release:
# pip install 'ms-swift>=3.4.0,<3.5.0'
# pip install ms-swift==3.4.1
# pip install 'git+https://github.com/modelscope/ms-swift.git@v3.4.0'
```

**Version note:** This repo targets **ms-swift 3.4** (`swift.llm.PtEngine`, `swift plugin` ORM API). Newer 3.5+ releases use a different inference engine layout; stay on 3.4.x unless you update `infer.py` and training scripts accordingly.

Download [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) and set paths (see [Environment variables](#environment-variables)).

For open-ended rewards with BERT-score:

```bash
export OPEN_QUESTION_METRIC=bert_score
export MODEL_DIR=/path/to/cache   # BERT weights cached here
pip install bert-score huggingface_hub
```

## Quick start

### 1. Prepare data

**SFT format** (user + assistant messages + images):

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "<answer>(A) benign</answer>"}
  ],
  "images": [{"path": "/path/to/slide.png"}]
}
```

Convert to RL format for GRPO:

```bash
python scripts/build_rldata_from_sft.py \
  --input data/sftdata_CLS_train.json \
  --output data/rldata_CLS_train.json
```

See [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) for detection, choice, and token-allocation schemas.

### 2. Supervised fine-tuning

```bash
export MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct
export DATA_DIR=/path/to/json_datasets
export MAX_PIXELS=200704

bash scripts/train_sft.sh
```

### 3. Merge LoRA

```bash
export ADAPTER_PATH=output_sft/v0-.../checkpoint-XXX
bash scripts/merge_lora.sh
```

### 4. GRPO (accuracy reward)

```bash
export MODEL_PATH=/path/to/merged-checkpoint
export DATA_DIR=/path/to/rl_json
bash scripts/train_grpo.sh
```

### 5. GRPO (token allocation)

Build token datasets, then train:

```bash
python scripts/build_rldata_token_alloc.py \
  --input-json data/rldata_CLS_train.json \
  --output-json data/rldata_CLS_train_token.json \
  --path-prefix /local/image/root

export TASK_PERFORMER_MODEL_PATH=/path/to/merged/task-performer
bash scripts/train_grpo_token.sh
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | — | Base or merged Qwen2.5-VL checkpoint |
| `DATA_DIR` | — | Directory with `sftdata_*.json` / `rldata_*.json` |
| `MAX_PIXELS` | `200704` | Max vision pixels (Qwen2.5-VL) |
| `OPEN_QUESTION_METRIC` | `bleu` | Open VQA metric: `bleu` or `bert_score` |
| `MODEL_DIR` | `path/to/MODEL` | Cache dir for BERT-score weights |
| `BERTSCORE_MODEL_ID` | `roberta-large` | Hugging Face model for BERT-score |
| `TASK_PERFORMER_MODEL_PATH` | — | Merged VLM used inside `external_acc_token` |

Copy `.env.example` and adjust for your cluster.

## Reward functions

Registered in `grpo/plugin.py` for `--reward_funcs`:

| Name | Class | Behavior |
|------|-------|----------|
| `external_format` | `ResponseFormatORM` | 1.0 if completion matches CoT + answer tags |
| `external_acc` | `MultiModalAccuracyORM` | Task score on policy completion |
| `external_acc_token` | `MultiModalAccuracyORM_token` | Policy predicts token count; task performer runs at that budget |

**Task-type detection** (from ground-truth `solution`):

- `[[x1,y1,x2,y2],...]` → detection (AP50 default)
- `(A)`, `(B)`, … → multiple choice (exact match)
- otherwise → open (BLEU-4 or BERT-score F1)

## Inference (two-stage LoRA)

Use a **merged SFT** checkpoint as the base, then run token LoRA → task LoRA in sequence:

```bash
export MODEL_PATH=/path/to/merged-sft
python infer.py \
  --model "${MODEL_PATH}" \
  --lora-stage1 /path/to/token-lora-checkpoint \
  --lora-stage2 /path/to/task-lora-checkpoint \
  --dataset-json /path/to/sftdata_test.json \
  --path-prefix /local/image/root \
  --output-json results/predictions.json
```

Output JSON contains `stage1_output` (predicted token budget), `predicted_tokens`, and `stage2_output` (final answer) per sample.

## Citation

If you use this repository, please cite:

```bibtex
@article{xu2025discovering,
  title={Discovering pathology rationale and token allocation for efficient multimodal pathology reasoning},
  author={Xu, Zhe and Jin, Cheng and Wang, Yihui and Liu, Ziyi and Chen, Hao},
  journal={arXiv preprint arXiv:2505.15687},
  year={2025}
}
```

## Acknowledgments

We thank the authors of [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) and [ms-swift](https://github.com/modelscope/ms-swift) for their open-source frameworks, which this project builds upon (ms-swift v3.4).

## License

This project is released under the Apache License 2.0 unless otherwise noted.
