"""MATH-500 (Hendrycks et al., 2021; Lightman et al. subset), 500 problems."""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

NAME = "math500"

SYSTEM = (
    "You are an expert competition mathematician. "
    "Reason carefully and show every step. "
    "Always end with \\boxed{<answer>}."
)


def normalize(s: str) -> str:
    """Light normalisation of MATH answers (no symbolic equivalence)."""
    s = s.strip()
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    s = re.sub(r"\\left|\\right", "", s)
    s = re.sub(r"\s+", "", s)
    s = s.rstrip(".")
    s = re.sub(r"\{(\w)\}", r"\1", s)
    return s.lower()


def extract_boxed(text: str) -> Optional[str]:
    """Content of the last ``\\boxed{...}`` (nested braces supported)."""
    starts = [m.start() for m in re.finditer(r"\\boxed\s*\{", text)]
    if not starts:
        return None
    pos = text.index("{", starts[-1])
    depth, content = 0, []
    for ch in text[pos:]:
        if ch == "{":
            depth += 1
            if depth > 1:
                content.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
            content.append(ch)
        else:
            content.append(ch)
    return "".join(content).strip() if depth == 0 else None


def load(max_samples: Optional[int] = None, **_) -> List[dict]:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    n = len(ds) if max_samples is None else min(max_samples, len(ds))
    return [
        {"idx": i, "question": ds[i]["problem"], "gt": ds[i]["answer"],
         "subject": ds[i].get("subject", ""), "level": ds[i].get("level", None)}
        for i in range(n)
    ]


def messages(item: dict):
    content = (
        "Solve the following competition math problem.\n"
        "Think through the problem carefully, step by step.\n"
        "Show all your working clearly.\n"
        "At the very end, write your final answer inside \\boxed{} like this:\n"
        "\\boxed{<your answer>}\n\n"
        f"Problem:\n{item['question']}"
    )
    fallback = f"Problem:\n{item['question']}\n\nSolve step by step. Write the final answer as \\boxed{{<answer>}}.\n"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}], fallback


def score(text: str, item: dict) -> Tuple[Optional[str], int]:
    """Exact match of the normalised last boxed answer (fallback: last number)."""
    gt = normalize(item["gt"])
    boxed = extract_boxed(text)
    if boxed is not None:
        pred = normalize(boxed)
        return pred, int(pred == gt)
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if nums:
        pred = normalize(nums[-1])
        return pred, int(pred == gt)
    return None, 0
