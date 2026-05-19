"""GRPO reward plugin for ms-swift pathology vision-language training.

Registers three outcome reward models (ORMs):
  - external_format: checks CoT + answer tag structure
  - external_acc: task accuracy on model completion
  - external_acc_token: token-budget allocation + task performer accuracy
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from qwen_vl_utils import process_vision_info
from swift.plugin import ORM, orms
from swift.utils import get_logger
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

logger = get_logger()

# Open-ended task metric: 'bleu' | 'bert_score' (override via OPEN_QUESTION_METRIC)
OPEN_QUESTION_METRIC = os.environ.get('OPEN_QUESTION_METRIC', 'bleu').strip().lower()
MODEL_DIR = os.environ.get('MODEL_DIR', 'path/to/MODEL')
BERTSCORE_MODEL_ID = os.environ.get('BERTSCORE_MODEL_ID', 'roberta-large')
TASK_PERFORMER_MODEL_PATH = os.environ.get(
    'TASK_PERFORMER_MODEL_PATH',
    'path/to/merged/task/performer',
)

_bert_scorer = None

# COCO-style AP50:95 IoU grid (0.5, 0.55, ..., 0.95)
_COCO_IOU_THRESHOLDS = [0.5 + 0.05 * i for i in range(10)]

# Match standard boxes [x1, y1, x2, y2] only (comma-separated quadruples)
_BOX_TUPLE_RE = re.compile(
    r'\[\s*([+-]?(?:\d+\.?\d*|\.\d+))\s*,\s*([+-]?(?:\d+\.?\d*|\.\d+))\s*,\s*'
    r'([+-]?(?:\d+\.?\d*|\.\d+))\s*,\s*([+-]?(?:\d+\.?\d*|\.\d+))\s*\]'
)


def determine_question_type(solution: str) -> str:
    """Infer task type from ground-truth solution string.

    Returns:
        'detection' for bounding-box answers,
        'choice' for multiple-choice (A)/(B)/... patterns,
        'open' for free-text VQA.
    """
    if '[[' in solution:
        return 'detection'

    if re.search(r'([A-Za-z]|[1-9]|1[0-9]|2[0-9]|3[0-2])\)', solution):
        return 'choice'

    return 'open'


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Intersection-over-union for axis-aligned boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def _first_answer_inner(text: str) -> Optional[str]:
    """Return inner text of the first <answer>...</answer> block, or None."""
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


def extract_boxes(solution: str) -> List[List[float]]:
    """Parse all [x1, y1, x2, y2] tuples from a solution string."""
    return [
        [float(m.group(i)) for i in range(1, 5)]
        for m in _BOX_TUPLE_RE.finditer(solution)
    ]


def calculate_ap(gt_boxes, pred_boxes, iou_threshold: float = 0.5) -> float:
    """Average precision at a single IoU threshold (greedy matching)."""
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    gt_matched = [False] * len(gt_boxes)

    for pred_idx, pred_box in enumerate(pred_boxes):
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_matched[gt_idx]:
                continue
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold:
            tp[pred_idx] = 1
            gt_matched[best_gt_idx] = True
        else:
            fp[pred_idx] = 1

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recall = tp_cumsum / len(gt_boxes)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    ap = 0.0
    for i in range(len(recall) - 1):
        ap += (recall[i + 1] - recall[i]) * precision[i + 1]

    return float(ap)


def calculate_detection_aps(pred_solution: str, true_solution: str) -> Dict[str, float]:
    """Compute detection AP50, AP75, and AP50:95."""
    pred_inner = _first_answer_inner(pred_solution)
    if pred_inner is None:
        return {'ap50': 0.0, 'ap75': 0.0, 'ap50_95': 0.0}
    pred_boxes = extract_boxes(pred_inner)

    true_inner = _first_answer_inner(true_solution)
    true_text = true_inner if true_inner is not None else true_solution.strip()
    true_boxes = extract_boxes(true_text)

    if not pred_boxes or not true_boxes:
        return {'ap50': 0.0, 'ap75': 0.0, 'ap50_95': 0.0}

    ap50 = calculate_ap(true_boxes, pred_boxes, iou_threshold=0.5)
    ap75 = calculate_ap(true_boxes, pred_boxes, iou_threshold=0.75)
    ap50_95 = float(np.mean([
        calculate_ap(true_boxes, pred_boxes, iou_threshold=t)
        for t in _COCO_IOU_THRESHOLDS
    ]))
    return {'ap50': ap50, 'ap75': ap75, 'ap50_95': ap50_95}


def calculate_detection_score(
    pred_solution: str,
    true_solution: str,
    metric: str = 'ap50',
) -> float:
    """Detection reward: one of ap50, ap75, ap50_95 (default ap50)."""
    aps = calculate_detection_aps(pred_solution, true_solution)
    return aps[metric]


def calculate_choice_score(pred_solution: str, true_solution: str) -> float:
    """Multiple-choice exact match (0 or 1)."""
    sol_match = re.search(r'<answer>(.*?)</answer>', true_solution)
    ground_truth = (
        sol_match.group(1).strip().split(')')[0]
        if sol_match
        else true_solution.strip().split(')')[0]
    )

    content_match = re.search(r'<answer>(.*?)</answer>', pred_solution)
    if not content_match:
        return 0.0

    inner = content_match.group(1).strip()
    if ')' in inner:
        student_answer = inner.split(')')[0]
    else:
        student_answer = inner

    return 1.0 if ground_truth == student_answer else 0.0


def _extract_open_answer_text(pred_solution: str, true_solution: str) -> Tuple[str, str]:
    sol_match = re.search(r'<answer>(.*?)</answer>', true_solution)
    true_text = sol_match.group(1).strip() if sol_match else true_solution.strip()

    pred_match = re.search(r'<answer>(.*?)</answer>', pred_solution)
    pred_text = pred_match.group(1).strip() if pred_match else pred_solution.strip()
    return pred_text, true_text


def _bertscore_local_dir() -> Path:
    safe_name = BERTSCORE_MODEL_ID.replace('/', '_')
    return Path(MODEL_DIR) / 'bert-score' / safe_name


def _ensure_bertscore_model() -> str:
    """Download/cache BERT-score weights under MODEL_DIR; return local path."""
    local_dir = _bertscore_local_dir()
    if (local_dir / 'config.json').exists():
        return str(local_dir)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            'bert_score requires huggingface_hub: pip install huggingface_hub'
        ) from e
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f'Downloading BERT-score model {BERTSCORE_MODEL_ID} -> {local_dir}')
    snapshot_download(repo_id=BERTSCORE_MODEL_ID, local_dir=str(local_dir))
    return str(local_dir)


def _get_bert_scorer():
    global _bert_scorer
    if _bert_scorer is None:
        import importlib.util

        if importlib.util.find_spec('bert_score') is None:
            raise ImportError('bert_score requires: pip install bert-score')
        from bert_score import BERTScorer
        from bert_score.utils import model2layers

        model_path = _ensure_bertscore_model()
        num_layers = model2layers.get(BERTSCORE_MODEL_ID)
        if num_layers is None:
            raise ValueError(
                f'Unknown BERTSCORE_MODEL_ID={BERTSCORE_MODEL_ID!r}; '
                'use a supported Hugging Face model id.'
            )
        _bert_scorer = BERTScorer(
            model_type=model_path,
            num_layers=num_layers,
            lang='en',
        )
    return _bert_scorer


def calculate_open_score_bleu(pred_solution: str, true_solution: str) -> float:
    """Open-ended task: smoothed BLEU-4 in [0, 1]."""
    pred_text, true_text = _extract_open_answer_text(pred_solution, true_solution)
    reference = [true_text.split()]
    candidate = pred_text.split()
    smoothie = SmoothingFunction().method4
    return sentence_bleu(
        reference,
        candidate,
        smoothing_function=smoothie,
        weights=(0.25, 0.25, 0.25, 0.25),
    )


def calculate_open_score_bert(pred_solution: str, true_solution: str) -> float:
    """Open-ended task: BERT-score F1 in [0, 1]."""
    pred_text, true_text = _extract_open_answer_text(pred_solution, true_solution)
    if not pred_text.strip() or not true_text.strip():
        return 0.0
    scorer = _get_bert_scorer()
    _, _, f1 = scorer.score([pred_text], [true_text])
    return float(f1[0].item())


def calculate_open_score(
    pred_solution: str,
    true_solution: str,
    metric: Optional[str] = None,
) -> float:
    """Open-ended task reward (bleu or bert_score)."""
    m = (metric or OPEN_QUESTION_METRIC).strip().lower()
    if m == 'bert_score':
        return calculate_open_score_bert(pred_solution, true_solution)
    if m == 'bleu':
        return calculate_open_score_bleu(pred_solution, true_solution)
    raise ValueError(f"Unknown open question metric: {m!r} (use 'bleu' or 'bert_score')")


def calculate_score(
    pred_solution: str,
    true_solution: str,
    open_metric: Optional[str] = None,
) -> float:
    """Dispatch to detection, choice, or open scoring based on ground truth."""
    question_type = determine_question_type(true_solution)

    if question_type == 'detection':
        return calculate_detection_score(pred_solution, true_solution)
    if question_type == 'choice':
        return calculate_choice_score(pred_solution, true_solution)
    return calculate_open_score(pred_solution, true_solution, metric=open_metric)


class ResponseFormatORM(ORM):
    """Reward completions that follow <think> ... <answer> structure."""

    def __call__(self, completions, **kwargs) -> List[float]:
        pattern = (
            r'^<think>.*?</think>\s*'
            r'<answer>.*?</answer>(?![\s\S])'
        )
        matches = [
            re.match(pattern, content, re.DOTALL | re.MULTILINE)
            for content in completions
        ]
        return [1.0 if match else 0.0 for match in matches]


class MultiModalAccuracyORM(ORM):
    """Task accuracy reward on the policy model's own completion."""

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = []
        for content, sol in zip(completions, solution):
            try:
                reward = calculate_score(content, sol)
            except Exception:
                reward = 0.0
            rewards.append(reward)
        return rewards


class MultiModalAccuracyORM_token(ORM):
    """Token-budget reward: policy predicts vision tokens; task performer runs at that budget."""

    def __init__(self):
        self.model_path = TASK_PERFORMER_MODEL_PATH
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto',
        )

    def get_qwen25_response(self, image_path: str, prompt: str, max_tokens: int = 256) -> List[str]:
        """Run the task-performer VLM at a given vision-token budget."""
        min_pixels = 1 * 28 * 28
        max_pixels = max_tokens * 28 * 28
        processor = AutoProcessor.from_pretrained(
            self.model_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        messages = [
            {
                'role': 'system',
                'content': (
                    'A conversation between User and Assistant. The user asks a question, '
                    'and the Assistant solves it. The assistant first thinks about the '
                    'reasoning process in the mind and then provides the user with the answer. '
                    'The reasoning process and answer are enclosed within '
                    '<think> </think> and <answer> </answer> tags, '
                    'respectively.'
                ),
            },
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': image_path},
                    {'type': 'text', 'text': prompt},
                ],
            },
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors='pt',
        )
        inputs = inputs.to('cuda')

        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = []
        for content, sol in zip(completions, solution):
            reward = 0.0
            try:
                sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                prompt_match = re.search(r'<prompt>(.*?)</prompt>', sol, re.DOTALL)
                prompt = prompt_match.group(1).strip()
                img_path_match = re.search(r'<img_path>(.*?)</img_path>', sol, re.DOTALL)
                img_path = img_path_match.group(1).strip()
                max_token_match = re.search(r'<max_token>(.*?)</max_token>', sol, re.DOTALL)
                max_token = min(int(max_token_match.group(1).strip()), 256)

                pre_tokens_match = re.search(r'<answer>(.*?)</answer>', content)
                pre_tokens_ = (
                    pre_tokens_match.group(1).strip()
                    if pre_tokens_match
                    else content.strip()
                )
                pre_tokens = list(map(int, re.findall(r'\b([2-9]|[1-9]\d+)\b', pre_tokens_)))

                if pre_tokens:
                    pre_vqa = self.get_qwen25_response(img_path, prompt, pre_tokens[0])
                    question_type = determine_question_type(ground_truth)

                    if question_type == 'detection':
                        score = calculate_detection_score(pre_vqa[0], ground_truth)
                    elif question_type == 'choice':
                        score = calculate_choice_score(pre_vqa[0], ground_truth)
                    else:
                        score = calculate_open_score(pre_vqa[0], ground_truth)

                    reward = score if pre_tokens[0] < max_token else 0.5 * score
            except Exception:
                pass
            rewards.append(reward)
        return rewards


# ms-swift reward registry (names used in --reward_funcs)
orms['external_format'] = ResponseFormatORM
orms['external_acc'] = MultiModalAccuracyORM
orms['external_acc_token'] = MultiModalAccuracyORM_token
