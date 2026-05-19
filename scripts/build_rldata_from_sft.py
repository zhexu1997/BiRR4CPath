#!/usr/bin/env python3
"""Convert SFT JSON (user+assistant) to RL JSON (user + solution) for GRPO training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def sft_to_rl(sample: Dict[str, Any]) -> Dict[str, Any]:
    messages = sample.get("messages") or []
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if not user_msgs:
        raise ValueError("sample has no user message")

    solution = sample.get("solution")
    if solution is None:
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        if not assistant_msgs:
            raise ValueError("sample has no assistant message or solution field")
        solution = assistant_msgs[-1].get("content", "")
        if isinstance(solution, list):
            solution = next(
                (c.get("text", "") for c in solution if c.get("type") == "text"),
                "",
            ).strip()
        else:
            solution = str(solution).strip()

    return {
        "messages": user_msgs,
        "images": sample["images"],
        "solution": solution,
    }


def convert(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [sft_to_rl(s) for s in samples]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(""),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(""),
    )
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as f:
        samples = json.load(f)

    rl_samples = convert(samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(rl_samples, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"converted {len(rl_samples)} samples -> {args.output}")


if __name__ == "__main__":
    main()
