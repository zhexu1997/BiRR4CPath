# Data formats

All datasets are JSON arrays of records compatible with [ms-swift](https://github.com/modelscope/ms-swift) multimodal training.

## SFT (supervised fine-tuning)

Each record has `messages` (user + assistant) and `images`:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Given the histopathology image, classify the lesion. Options: (A) benign (B) malignant."
    },
    {
      "role": "assistant",
      "content": "<answer>(A) benign</answer>"
    }
  ],
  "images": [
    {"path": "/data/slides/sample_001.png"}
  ]
}
```

Assistant content should use the same tag structure as `grpo/prompt.txt`.

## RL (task performer)

Convert from SFT with `scripts/build_rldata_from_sft.py`. Ground truth moves to `solution`; only user messages remain:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Given the histopathology image, classify the lesion. Options: (A) benign (B) malignant."
    }
  ],
  "images": [{"path": "/data/slides/sample_001.png"}],
  "solution": "<answer>(A) benign</answer>"
}
```

### Detection

Ground-truth boxes inside `<answer>`:

```text
<answer>[[120.0, 80.0, 200.0, 160.0], [300.0, 100.0, 400.0, 220.0]]</answer>
```

### Diagnosis and Close VQA

Option label inside `<answer>` :
```text
<answer> (A) benign </answer>
```

### Open VQA

Free text inside `<answer>`.

## RL (token allocator)

Build with `scripts/build_rldata_token_alloc.py`. The user prompt asks for an optimal vision token count; `solution` includes metadata tags:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Allocate the optimal token number for the image based on the pathology task. ..."
    }
  ],
  "images": [{"path": "/data/slides/sample_001.png"}],
  "solution": "<prompt>Given the histopathology image, classify...</prompt> <img_path>/data/slides/sample_001.png</img_path> <max_token>512</max_token> <answer>(A) benign</answer>"
}
```

The policy model predicts a positive integer in `<answer>`; the task performer (see `TASK_PERFORMER_MODEL_PATH`) runs pathology QA at that vision-token budget.

## Image paths

- Use absolute paths or a consistent root; for token script, pass `--path-prefix` to remap cluster mounts to local storage.
- Qwen2.5-VL resizes by `MAX_PIXELS` (environment variable, default `200704` in training scripts).

## File naming convention

| Pattern | Stage |
|---------|--------|
| `sftdata_{CLS,DET,VQA}_train.json` | SFT |
| `rldata_{task}_{dataset}_train.json` | GRPO (accuracy) |
| `rldata_{task}_{dataset}_train_token.json` | GRPO (token allocation) |
