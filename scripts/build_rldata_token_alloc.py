#!/usr/bin/env python3
"""Convert RL JSON for vision-token allocation GRPO training (Qwen2.5-VL).

Reads Swift-style RL JSON (messages + images + solution), rewrites the user
prompt to ask for an optimal image token count, and augments solution with
<prompt>, <img_path>, and <max_token> metadata for external_acc_token reward.
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

try:
    from qwen_vl_utils.vision_process import (
        IMAGE_FACTOR,
        MAX_PIXELS,
        MIN_PIXELS,
        smart_resize,
    )
except ImportError as e:
    raise SystemExit(
        'qwen-vl-utils is required: pip install qwen-vl-utils'
    ) from e

PATCH_FACTOR = IMAGE_FACTOR  # 28 for Qwen2.5-VL
CURRENT_TOKEN_CAP = 256  # fixed input budget; images using fewer tokens keep actual count


def tokens_to_max_pixels(num_tokens: int) -> int:
    return int(num_tokens) * PATCH_FACTOR * PATCH_FACTOR


USER_PROMPT_TEMPLATE = (
    'Allocate the optimal token number for the image based on the pathology task. '
    'Generally, simple images and tasks receive fewer tokens and complex ones receive more tokens. '
    'The current input token number is {current_token} and a maximum limit is {max_token}. '
    'The pathology task is: {pathology_prompt}. '
    'The answer should be a positive integer of the image token number.'
)


def qwen25vl_max_tokens(
    height: int,
    width: int,
    *,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> int:
    """Vision token upper bound for an image at native resolution (Qwen2.5-VL)."""
    if min_pixels is None:
        min_pixels = MIN_PIXELS
    if max_pixels is None:
        max_pixels = MAX_PIXELS
    h_bar, w_bar = smart_resize(
        height,
        width,
        factor=PATCH_FACTOR,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return (h_bar // PATCH_FACTOR) * (w_bar // PATCH_FACTOR)


def qwen25vl_current_tokens(height: int, width: int, cap: int = CURRENT_TOKEN_CAP) -> int:
    """Vision tokens under the fixed input budget (cap); below cap, use actual count."""
    return qwen25vl_max_tokens(
        height,
        width,
        max_pixels=tokens_to_max_pixels(cap),
    )


def resolve_image_path(path: str, path_prefix: str) -> Path:
    p = Path(path)
    if p.is_file():
        return p
    if path_prefix:
        alt = Path(path_prefix) / path.lstrip('/')
        if alt.is_file():
            return alt
        parts = path.split('/')
        for i in range(len(parts)):
            candidate = Path(path_prefix) / '/'.join(parts[i:])
            if candidate.is_file():
                return candidate
    return p


def read_image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as im:
        im = im.convert('RGB')
        return im.height, im.width


def extract_user_prompt(item: Dict[str, Any]) -> str:
    msgs = item.get('messages') or []
    user_msg = next((m for m in msgs if m.get('role') == 'user'), msgs[0] if msgs else {})
    content = user_msg.get('content', '')
    if isinstance(content, list):
        return next((c.get('text', '') for c in content if c.get('type') == 'text'), '').strip()
    return str(content).strip()


def extract_image_path(item: Dict[str, Any]) -> str:
    img_entry = item.get('images') or item.get('image')
    if not img_entry:
        raise ValueError('missing images field')
    if isinstance(img_entry, list):
        return str(img_entry[0]['path'] if isinstance(img_entry[0], dict) else img_entry[0])
    if isinstance(img_entry, dict):
        return str(img_entry['path'])
    return str(img_entry)


def augment_solution(
    original_solution: str,
    pathology_prompt: str,
    img_path: str,
    max_token: int,
) -> str:
    """Prepend metadata tags; skip if already present."""
    if re.search(r'<max_token>', original_solution):
        return original_solution
    meta = (
        f'<prompt>{pathology_prompt}</prompt> '
        f'<img_path>{img_path}</img_path> '
        f'<max_token>{max_token}</max_token> '
    )
    return meta + original_solution


def transform_item(
    item: Dict[str, Any],
    path_prefix: str,
    on_missing: str,
    default_size: Optional[Tuple[int, int]],
) -> Optional[Dict[str, Any]]:
    pathology_prompt = extract_user_prompt(item)
    img_path = extract_image_path(item)
    resolved = resolve_image_path(img_path, path_prefix)

    if resolved.is_file():
        height, width = read_image_size(resolved)
    elif on_missing == 'skip':
        return None
    elif on_missing == 'default' and default_size is not None:
        height, width = default_size
    else:
        raise FileNotFoundError(f'image not found: {img_path} (resolved: {resolved})')

    max_token = qwen25vl_max_tokens(height, width)
    current_token = qwen25vl_current_tokens(height, width)
    new_item = json.loads(json.dumps(item))
    new_item['messages'] = [
        {
            'role': 'user',
            'content': USER_PROMPT_TEMPLATE.format(
                current_token=current_token,
                max_token=max_token,
                pathology_prompt=pathology_prompt,
            ),
        }
    ]
    new_item['solution'] = augment_solution(
        item.get('solution', ''),
        pathology_prompt,
        img_path,
        max_token,
    )
    return new_item


def _worker(args: Tuple[int, Dict[str, Any], str, str, Optional[Tuple[int, int]]]) -> Tuple[int, Optional[Dict[str, Any]], Optional[str]]:
    idx, item, path_prefix, on_missing, default_size = args
    try:
        out = transform_item(item, path_prefix, on_missing, default_size)
        return idx, out, None
    except Exception as e:
        return idx, None, str(e)


def convert_dataset(
    data: List[Dict[str, Any]],
    path_prefix: str,
    on_missing: str,
    default_size: Optional[Tuple[int, int]],
    workers: int,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    if workers <= 1:
        out: List[Dict[str, Any]] = []
        errors: List[str] = []
        for i, item in enumerate(data):
            try:
                row = transform_item(item, path_prefix, on_missing, default_size)
                if row is not None:
                    out.append(row)
            except Exception as e:
                errors.append(f'[{i}] {e}')
        return out, errors

    indexed_args = [
        (i, item, path_prefix, on_missing, default_size) for i, item in enumerate(data)
    ]
    results: Dict[int, Dict[str, Any]] = {}
    errors: List[str] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker, a) for a in indexed_args]
        for fut in as_completed(futures):
            idx, row, err = fut.result()
            if err:
                errors.append(f'[{idx}] {err}')
            elif row is not None:
                results[idx] = row
    out = [results[i] for i in sorted(results)]
    return out, errors


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--input-json',
        type=Path,
        default=Path('data/rldata_CLS_ESCA_train.json'),
        help='Input Swift RL JSON',
    )
    p.add_argument(
        '--output-json',
        type=Path,
        default=None,
        help='Output path (default: <input_stem>_token.json)',
    )
    p.add_argument(
        '--path-prefix',
        type=str,
        default='',
        help='Local root to resolve image paths when originals are on another mount',
    )
    p.add_argument(
        '--on-missing',
        choices=('error', 'skip', 'default'),
        default='error',
        help='Behavior when image file is missing',
    )
    p.add_argument(
        '--default-size',
        type=str,
        default='',
        help='HxW used when --on-missing=default, e.g. 224x224',
    )
    p.add_argument('--workers', type=int, default=8, help='Parallel workers (0/1 = serial)')
    p.add_argument('--max-samples', type=int, default=0, help='Limit records (0 = all)')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = args.input_json.resolve()
    out_path = args.output_json
    if out_path is None:
        out_path = in_path.with_name(f'{in_path.stem}_token.json')
    else:
        out_path = out_path.resolve()

    default_size: Optional[Tuple[int, int]] = None
    if args.default_size:
        h, w = args.default_size.lower().split('x')
        default_size = (int(h), int(w))

    with open(in_path, encoding='utf-8') as f:
        data = json.load(f)
    if args.max_samples > 0:
        data = data[: args.max_samples]

    workers = max(1, args.workers) if args.workers > 1 else 1
    converted, errors = convert_dataset(
        data,
        args.path_prefix,
        args.on_missing,
        default_size,
        workers,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
        f.write('\n')

    print(f'Input:  {in_path} ({len(data)} records)')
    print(f'Output: {out_path} ({len(converted)} records)')
    if errors:
        print(f'Errors: {len(errors)} (showing up to 5)')
        for line in errors[:5]:
            print('  ', line)
        if len(errors) > 5:
            print(f'  ... and {len(errors) - 5} more')


if __name__ == '__main__':
    main()
