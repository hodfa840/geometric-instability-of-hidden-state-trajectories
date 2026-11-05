"""BIG-Bench Hard (Suzgun et al., 2023), 12-subtask subset (2,865 items in total)."""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

NAME = "bbh"

SUBTASKS = (
    "causal_judgement", "date_understanding", "disambiguation_qa",
    "formal_fallacies", "logical_deduction_five_objects", "navigate",
    "reasoning_about_colored_objects", "sports_understanding",
    "temporal_sequences", "tracking_shuffled_objects_five_objects",
    "web_of_lies", "snarks",
)
SYSTEM = "You are a careful reasoner. Think step by step and end with `Answer: <answer>`."


def load(max_samples: Optional[int] = None, per_task: Optional[int] = None, **_) -> List[dict]:
    """All items of the 12 subtasks (optionally the first ``per_task`` of each)."""
    from datasets import load_dataset

    items: List[dict] = []
    for task in SUBTASKS:
        try:
            ds = load_dataset("lukaemon/bbh", task, split="test")
        except Exception:
            ds = load_dataset("maveriq/bigbenchhard", task, split="train")
        n = len(ds) if per_task is None else min(per_task, len(ds))
        for i in range(n):
            items.append({"idx": len(items), "subtask": task, "question": ds[i]["input"], "gt": ds[i]["target"]})
    return items if max_samples is None else items[:max_samples]


def messages(item: dict):
    content = (
        "Solve the following problem by reasoning step by step. "
        "At the end, write a single line `Answer: <your answer>` with a "
        "concise final answer (e.g. a letter like (A), or yes/no, or a short phrase).\n\n"
        f"{item['question']}"
    )
    fallback = f"{item['question']}\n\nReason step by step. End with `Answer: <answer>`.\n"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}], fallback


def _normalize(s) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"""^[*_()\[\]"'`.\s]+|[*_()\[\]"'`.\s]+$""", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _letter_prefix(s) -> Optional[str]:
    """``A)``, ``(A)`` or ``A.`` at the start of a prediction -> ``a``."""
    if not s:
        return None
    m = re.match(r"^\s*\(?\s*([a-zA-Z])\s*[\)\.\]]\s", str(s))
    return m.group(1).lower() if m else None


def score(text: str, item: dict) -> Tuple[Optional[str], int]:
    """Last ``Answer: x`` (fallback: last non-empty line), normalised exact match."""
    gt_norm = _normalize(item["gt"])
    gt_stripped = re.sub(r"[()]", "", gt_norm)
    gt_letter = gt_stripped if (len(gt_stripped) == 1 and gt_stripped.isalpha()) else None

    def _match(raw: str):
        pred = _normalize(raw)
        if pred == gt_norm or re.sub(r"[()]", "", pred) == gt_stripped:
            return pred, 1
        if gt_letter is not None and _letter_prefix(raw) == gt_letter:
            return pred, 1
        return pred, 0

    matches = list(re.finditer(r"[Aa]nswer\s*(?:is|:)\s*(.+?)(?:\.|$|\n)", text))
    if matches:
        return _match(matches[-1].group(1).strip())
    for line in reversed(text.strip().splitlines()):
        if line.strip():
            return _match(line.strip())
    return None, 0
