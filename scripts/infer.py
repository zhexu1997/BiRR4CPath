#!/usr/bin/env python3
"""Two-stage LoRA inference on Qwen2.5-VL with a merged SFT base model.

Pipeline:
  1. Load merged SFT weights as the base model.
  2. Stage 1 (LoRA-1): vision token cap (default 256) → predict token budget N.
  3. Stage 2 (LoRA-2): run the pathology task at vision token cap N.

Usage:
  python infer.py \\
    --model /path/to/merged-sft \\
    --lora-stage1 /path/to/token-lora \\
    --lora-stage2 /path/to/task-lora \\
    --dataset-json data/sftdata_CLS_test.json \\
    --path-prefix /local/data/root \\
    --output-json results/predictions.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

# Project targets ms-swift v3.4 (swift.llm.PtEngine). Fallback imports support newer releases only.
def _import_swift_infer():
    last_err: Optional[BaseException] = None
    for importer in (
        lambda: _import_from_swift_llm(),
        lambda: _import_from_swift_infer_engine(),
        lambda: _import_from_swift_top_level(),
    ):
        try:
            return importer()
        except ImportError as e:
            last_err = e
    raise ImportError(
        'Cannot import inference engine from ms-swift. '
        'For v3.4 use: from swift.llm import PtEngine, InferRequest, RequestConfig, AdapterRequest'
    ) from last_err


def _import_from_swift_llm():
    from swift.llm import AdapterRequest, InferRequest, PtEngine, RequestConfig

    return PtEngine, InferRequest, RequestConfig, AdapterRequest


def _import_from_swift_infer_engine():
    from swift.infer_engine import (
        AdapterRequest,
        InferRequest,
        RequestConfig,
        TransformersEngine,
    )

    return TransformersEngine, InferRequest, RequestConfig, AdapterRequest


def _import_from_swift_top_level():
    from swift import InferRequest, RequestConfig, TransformersEngine
    from swift.infer_engine.utils import AdapterRequest

    return TransformersEngine, InferRequest, RequestConfig, AdapterRequest


SwiftInferEngine, InferRequest, RequestConfig, AdapterRequest = _import_swift_infer()

PATCH_FACTOR = 28
STAGE1_VISION_TOKEN_CAP = 256

STAGE1_PROMPT_TEMPLATE = (
    'Allocate the optimal token number for the image based on the pathology task. '
    'Generally, simple images and tasks receive fewer tokens and complex ones receive more tokens. '
    'The current input token number is {current_token} and a maximum limit is {max_token}. '
    'The pathology task is: {pathology_prompt}. '
    'The answer should be a positive integer of the image token number.'
)

SYSTEM_PROMPT = (
    'A conversation between User and Assistant. The user asks a question, and the Assistant '
    'solves it. The assistant first thinks about the reasoning process in the mind and then '
    'provides the user with the answer. The reasoning process and answer are enclosed within '
    '<think> </think> and <answer> </answer> tags, respectively, i.e., '
    '<think> reasoning process here </think><answer> answer here </answer>'
)


def tokens_to_max_pixels(num_tokens: int) -> int:
    return int(num_tokens) * PATCH_FACTOR * PATCH_FACTOR


def set_vision_pixel_budget(
    max_tokens: int,
    min_tokens: Optional[int] = None,
) -> None:
    """Set Qwen2.5-VL MAX_PIXELS (= max vision tokens × 28²)."""
    max_pixels = tokens_to_max_pixels(max_tokens)
    os.environ['MAX_PIXELS'] = str(max_pixels)
    try:
        from qwen_vl_utils import vision_process

        vision_process.MAX_PIXELS = max_pixels
        if min_tokens is not None:
            min_pixels = tokens_to_max_pixels(min_tokens)
            os.environ['MIN_PIXELS'] = str(min_pixels)
            vision_process.MIN_PIXELS = min_pixels
    except ImportError:
        if min_tokens is not None:
            os.environ['MIN_PIXELS'] = str(tokens_to_max_pixels(min_tokens))


def qwen25vl_vision_tokens(
    height: int,
    width: int,
    *,
    max_token_cap: Optional[int] = None,
) -> int:
    from qwen_vl_utils.vision_process import MAX_PIXELS, MIN_PIXELS, smart_resize

    max_pixels = tokens_to_max_pixels(max_token_cap) if max_token_cap is not None else MAX_PIXELS
    h_bar, w_bar = smart_resize(
        height,
        width,
        factor=PATCH_FACTOR,
        min_pixels=MIN_PIXELS,
        max_pixels=max_pixels,
    )
    return (h_bar // PATCH_FACTOR) * (w_bar // PATCH_FACTOR)


def read_image_hw(image_path: str) -> tuple[int, int]:
    from PIL import Image

    with Image.open(image_path) as im:
        w, h = im.size
    return h, w


def build_stage1_prompt(
    pathology_prompt: str,
    image_path: str,
    stage1_cap: int = STAGE1_VISION_TOKEN_CAP,
) -> tuple[str, int, int]:
    h, w = read_image_hw(image_path)
    current_token = qwen25vl_vision_tokens(h, w, max_token_cap=stage1_cap)
    max_token = qwen25vl_vision_tokens(h, w, max_token_cap=None)
    text = STAGE1_PROMPT_TEMPLATE.format(
        current_token=current_token,
        max_token=max_token,
        pathology_prompt=pathology_prompt,
    )
    return text, current_token, max_token


def parse_predicted_token_count(
    text: str,
    default: int = 256,
    min_n: int = 2,
    max_n: int = 8192,
) -> int:
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    body = m.group(1).strip() if m else text.strip()
    nums = [int(x) for x in re.findall(r'\b([2-9]|[1-9]\d+)\b', body)]
    if not nums:
        return default
    return max(min_n, min(nums[0], max_n))


def resolve_image_path(path: str, path_prefix: str) -> Path:
    p = Path(path)
    if p.is_file():
        return p
    if not path_prefix:
        return p
    root = Path(path_prefix)
    alt = root / path.lstrip('/')
    if alt.is_file():
        return alt
    parts = path.split('/')
    for i in range(len(parts)):
        candidate = root / '/'.join(parts[i:])
        if candidate.is_file():
            return candidate
    return p


def extract_user_prompt(item: Dict[str, Any]) -> str:
    msgs = item.get('messages') or []
    user_msg = next((m for m in msgs if m.get('role') == 'user'), msgs[0] if msgs else {})
    content = user_msg.get('content', '')
    if isinstance(content, list):
        return next((c.get('text', '') for c in content if c.get('type') == 'text'), '').strip()
    return str(content).strip()


def extract_assistant_reference(item: Dict[str, Any]) -> Optional[str]:
    msgs = item.get('messages') or []
    asst = next((m for m in msgs if m.get('role') == 'assistant'), None)
    if not asst:
        return None
    content = asst.get('content', '')
    if isinstance(content, list):
        return next((c.get('text', '') for c in content if c.get('type') == 'text'), '').strip() or None
    text = str(content).strip()
    return text or None


def extract_image_path(item: Dict[str, Any]) -> Optional[str]:
    img_entry = item.get('images') or item.get('image')
    if not img_entry:
        return None
    if isinstance(img_entry, list):
        first = img_entry[0]
        if isinstance(first, dict):
            return str(first.get('path') or first.get('image') or '')
        return str(first)
    if isinstance(img_entry, dict):
        return str(img_entry.get('path') or img_entry.get('image') or '')
    return str(img_entry)


def load_sft_json_samples(
    dataset_json: str,
    max_samples: int,
    path_prefix: str = '',
    on_missing: str = 'skip',
) -> List[Dict[str, str]]:
    with open(dataset_json, encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f'{dataset_json}: expected a JSON array of samples')

    samples: List[Dict[str, str]] = []
    skipped = 0
    consecutive_missing = 0
    for idx, item in enumerate(data):
        if len(samples) >= max_samples:
            break
        raw_path = extract_image_path(item)
        if not raw_path:
            skipped += 1
            continue
        prompt = extract_user_prompt(item)
        if not prompt:
            skipped += 1
            continue
        resolved = resolve_image_path(raw_path, path_prefix)
        if not resolved.is_file():
            consecutive_missing += 1
            if (
                not path_prefix
                and len(samples) == 0
                and consecutive_missing >= 32
            ):
                raise FileNotFoundError(
                    f'First {consecutive_missing} image paths in {dataset_json} are not '
                    f'available locally (e.g. {raw_path}). Set --path-prefix.'
                )
            if on_missing == 'skip':
                skipped += 1
                continue
            raise FileNotFoundError(
                f'[{idx}] image not found: {raw_path} (resolved: {resolved})'
            )
        consecutive_missing = 0
        row: Dict[str, str] = {
            'image': str(resolved),
            'prompt': prompt,
            'image_orig': raw_path,
        }
        ref = extract_assistant_reference(item)
        if ref:
            row['reference'] = ref
        samples.append(row)

    if not samples:
        raise ValueError(
            f'No usable samples from {dataset_json} (skipped {skipped}).'
        )
    if skipped:
        print(f'  skipped {skipped} item(s) (missing image or empty prompt)')
    return samples


def load_image_prompt_pairs(
    image_paths: Sequence[str],
    prompts: Sequence[str],
    dataset_json: Optional[str],
    max_samples: int,
    path_prefix: str = '',
    on_missing: str = 'skip',
) -> List[Dict[str, str]]:
    samples: List[Dict[str, str]] = []

    if dataset_json:
        samples = load_sft_json_samples(
            dataset_json,
            max_samples,
            path_prefix=path_prefix,
            on_missing=on_missing,
        )

    for img, pr in zip(image_paths, prompts):
        if len(samples) >= max_samples:
            break
        resolved = resolve_image_path(img, path_prefix)
        if not resolved.is_file():
            if on_missing == 'skip':
                continue
            raise FileNotFoundError(f'image not found: {img} (resolved: {resolved})')
        samples.append({'image': str(resolved), 'prompt': pr, 'image_orig': img})

    if not samples:
        raise ValueError(
            'No samples: pass --dataset-json and/or --image with valid paths.'
        )

    missing = [s['image'] for s in samples if not Path(s['image']).is_file()]
    if missing:
        raise FileNotFoundError(
            f'{len(missing)} image(s) not found, e.g. {missing[0]}. Set --path-prefix.'
        )
    return samples[:max_samples]


def apply_peft_qwen2vl_lora_patch() -> None:
    """PEFT LoRA + Qwen2.5-VL ViT: drop mismatched adapter_names batch dim."""
    try:
        from peft.tuners.lora import layer as lora_layer
    except ImportError:
        return
    if getattr(lora_layer, '_birr4cpath_qwen2vl_lora_patch', False):
        return

    _orig_check = lora_layer.LoraLayer._check_forward_args

    def _check_forward_args(self, x, *args, **kwargs):
        adapter_names = kwargs.get('adapter_names')
        if adapter_names is not None:
            batch_size = x.shape[0] if isinstance(x, torch.Tensor) and x.ndim > 0 else len(x)
            if len(adapter_names) != batch_size:
                kwargs.pop('adapter_names', None)
                return
        return _orig_check(self, x, *args, **kwargs)

    lora_layer.LoraLayer._check_forward_args = _check_forward_args
    lora_layer._birr4cpath_qwen2vl_lora_patch = True


def build_engine(model: str, dtype: str, attn_impl: Optional[str]):
    apply_peft_qwen2vl_lora_patch()
    torch_dtype = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
        'float32': torch.float32,
    }[dtype]
    kwargs: Dict[str, Any] = {'torch_dtype': torch_dtype}
    if attn_impl:
        kwargs['attn_impl'] = attn_impl
    return SwiftInferEngine(model, max_batch_size=1, **kwargs)


def activate_peft_adapter(engine, adapter_name: str, adapter_path: Optional[str] = None) -> None:
    apply_peft_qwen2vl_lora_patch()
    if not hasattr(engine, '_adapters_pool'):
        engine._adapters_pool = {}
    if adapter_path is not None and adapter_name not in engine._adapters_pool:
        engine._adapters_pool[adapter_name] = AdapterRequest(adapter_name, adapter_path)
        engine._add_adapter(adapter_path, adapter_name)

    model = engine.model
    if hasattr(model, 'set_adapter'):
        model.set_adapter(adapter_name)
        return
    if hasattr(model, 'set_active_adapters'):
        model.set_active_adapters(adapter_name)
        return
    raise RuntimeError(
        f'Cannot activate adapter {adapter_name!r}: no set_adapter / set_active_adapters'
    )


def infer_with_adapter(engine, infer_requests, request_config, adapter_name: str, adapter_path: str):
    apply_peft_qwen2vl_lora_patch()
    activate_peft_adapter(engine, adapter_name, adapter_path)
    return engine.infer(infer_requests, request_config)


def _message_content(text: str) -> List[Dict[str, str]]:
    return [{'role': 'user', 'content': text}]


def infer_sample_two_stage(
    engine,
    sample: Dict[str, str],
    lora_stage1: str,
    lora_stage2: str,
    *,
    stage1_max_tokens: int,
    stage1_max_new_tokens: int,
    stage2_max_new_tokens: int,
    default_n: int,
    temperature: float,
    system_prompt: Optional[str],
) -> Dict[str, Any]:
    """Run stage-1 (token LoRA) then stage-2 (task LoRA) on one sample."""
    messages_system = (
        [{'role': 'system', 'content': system_prompt}]
        if system_prompt
        else []
    )

    # Stage 1: token budget prediction
    set_vision_pixel_budget(stage1_max_tokens)
    s1_text, stage1_vt, native_max_token = build_stage1_prompt(
        sample['prompt'], sample['image'], stage1_cap=stage1_max_tokens
    )
    req1 = InferRequest(
        messages=messages_system + _message_content(s1_text),
        images=[sample['image']],
    )
    cfg1 = RequestConfig(max_tokens=stage1_max_new_tokens, temperature=temperature)
    resp1 = infer_with_adapter(engine, [req1], cfg1, 'lora_stage1', lora_stage1)[0]
    stage1_output = resp1.choices[0].message.content
    predicted_n = parse_predicted_token_count(stage1_output, default=default_n)

    # Stage 2: pathology task at predicted vision token cap
    set_vision_pixel_budget(predicted_n)
    req2 = InferRequest(
        messages=messages_system + _message_content(sample['prompt']),
        images=[sample['image']],
    )
    cfg2 = RequestConfig(max_tokens=stage2_max_new_tokens, temperature=temperature)
    resp2 = infer_with_adapter(engine, [req2], cfg2, 'lora_stage2', lora_stage2)[0]
    stage2_output = resp2.choices[0].message.content

    result: Dict[str, Any] = {
        'image': sample['image'],
        'image_orig': sample.get('image_orig', sample['image']),
        'prompt': sample['prompt'],
        'stage1_prompt': s1_text,
        'stage1_vision_tokens': stage1_vt,
        'native_max_vision_tokens': native_max_token,
        'stage1_output': stage1_output,
        'predicted_tokens': predicted_n,
        'stage2_output': stage2_output,
    }
    if 'reference' in sample:
        result['reference'] = sample['reference']
    return result


def run_inference(
    engine,
    samples: List[Dict[str, str]],
    lora_stage1: str,
    lora_stage2: str,
    *,
    stage1_max_tokens: int,
    stage1_max_new_tokens: int,
    stage2_max_new_tokens: int,
    default_n: int,
    temperature: float,
    system_prompt: Optional[str],
) -> List[Dict[str, Any]]:
    activate_peft_adapter(engine, 'lora_stage1', lora_stage1)
    activate_peft_adapter(engine, 'lora_stage2', lora_stage2)

    predictions: List[Dict[str, Any]] = []
    for i, sample in enumerate(samples):
        pred = infer_sample_two_stage(
            engine,
            sample,
            lora_stage1,
            lora_stage2,
            stage1_max_tokens=stage1_max_tokens,
            stage1_max_new_tokens=stage1_max_new_tokens,
            stage2_max_new_tokens=stage2_max_new_tokens,
            default_n=default_n,
            temperature=temperature,
            system_prompt=system_prompt,
        )
        predictions.append(pred)
        print(
            f'  [{i + 1}/{len(samples)}] N={pred["predicted_tokens"]} '
            f'vision_tok={pred["stage1_vision_tokens"]}'
        )
    return predictions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        '--model',
        type=str,
        default=os.environ.get('MODEL_PATH', ''),
        help='Merged SFT checkpoint (base weights for inference)',
    )
    p.add_argument('--lora-stage1', type=str, required=True, help='LoRA-1: token allocation')
    p.add_argument('--lora-stage2', type=str, required=True, help='LoRA-2: pathology task')
    p.add_argument('--image', action='append', default=[], help='Image path(s)')
    p.add_argument('--prompt', action='append', default=[], help='Task prompt(s)')
    p.add_argument('--dataset-json', type=str, default=None, help='SFT/RL test JSON')
    p.add_argument('--path-prefix', type=str, default='', help='Local root for JSON image paths')
    p.add_argument('--on-missing', choices=['skip', 'error'], default='skip')
    p.add_argument('--max-samples', type=int, default=0, help='0 = all samples in JSON')
    p.add_argument(
        '--stage1-max-tokens',
        type=int,
        default=STAGE1_VISION_TOKEN_CAP,
        help='Stage-1 vision token upper cap',
    )
    p.add_argument('--default-n', type=int, default=256, help='Fallback N if stage-1 parse fails')
    p.add_argument('--stage1-max-new-tokens', type=int, default=64)
    p.add_argument('--stage2-max-new-tokens', type=int, default=512)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--dtype', choices=['bfloat16', 'float16', 'float32'], default='bfloat16')
    p.add_argument('--attn-impl', type=str, default=None, help='e.g. flash_attention_2')
    p.add_argument('--cuda-device', type=str, default='0')
    p.add_argument(
        '--system-prompt',
        type=str,
        default=None,
        help='System prompt (default: grpo/prompt.txt if present, else built-in)',
    )
    p.add_argument('--no-system-prompt', action='store_true')
    p.add_argument('--output-json', type=str, default='results/predictions.json')
    return p.parse_args()


def resolve_system_prompt(args: argparse.Namespace) -> Optional[str]:
    if args.no_system_prompt:
        return None
    if args.system_prompt is not None:
        return args.system_prompt
    prompt_file = Path(__file__).resolve().parent / 'grpo' / 'prompt.txt'
    if prompt_file.is_file():
        return prompt_file.read_text(encoding='utf-8').strip()
    return SYSTEM_PROMPT


def main() -> None:
    args = parse_args()
    if not args.model:
        raise SystemExit('error: set --model or MODEL_PATH to merged SFT checkpoint')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device

    max_samples = args.max_samples if args.max_samples > 0 else 10**9
    prompts = list(args.prompt)
    if not prompts and args.image:
        prompts = ['']

    samples = load_image_prompt_pairs(
        args.image,
        prompts,
        args.dataset_json,
        max_samples,
        path_prefix=args.path_prefix,
        on_missing=args.on_missing,
    )

    system_prompt = resolve_system_prompt(args)
    set_vision_pixel_budget(args.stage1_max_tokens)

    print(f'Base model (merged SFT): {args.model}')
    print(f'LoRA stage 1 (token):    {args.lora_stage1}')
    print(f'LoRA stage 2 (task):     {args.lora_stage2}')
    print(f'Samples: {len(samples)}')

    engine = build_engine(args.model, args.dtype, args.attn_impl)

    predictions = run_inference(
        engine,
        samples,
        args.lora_stage1,
        args.lora_stage2,
        stage1_max_tokens=args.stage1_max_tokens,
        stage1_max_new_tokens=args.stage1_max_new_tokens,
        stage2_max_new_tokens=args.stage2_max_new_tokens,
        default_n=args.default_n,
        temperature=args.temperature,
        system_prompt=system_prompt,
    )

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'model': args.model,
        'lora_stage1': args.lora_stage1,
        'lora_stage2': args.lora_stage2,
        'stage1_max_tokens': args.stage1_max_tokens,
        'num_samples': len(predictions),
        'dataset_json': args.dataset_json,
        'path_prefix': args.path_prefix or None,
        'predictions': predictions,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Wrote {len(predictions)} prediction(s) -> {out_path.resolve()}')

    del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
