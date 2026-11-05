"""CRUXEval-O (Gu et al., 2024): predict the output of a Python function call (800 items)."""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

NAME = "cruxeval"

SYSTEM = (
    "You are an expert Python programmer. "
    "Trace through code execution carefully, step by step, "
    "tracking every variable. Always end with 'Output: <result>'."
)


def load(max_samples: Optional[int] = None, **_) -> List[dict]:
    from datasets import load_dataset

    ds = load_dataset("cruxeval-org/cruxeval", split="test")
    n = len(ds) if max_samples is None else min(max_samples, len(ds))
    return [{"idx": i, "code": ds[i]["code"], "input": ds[i]["input"], "gt": ds[i]["output"]} for i in range(n)]


def messages(item: dict):
    content = (
        "Predict the output of the following Python function call.\n"
        "Trace through the code step by step, tracking the value of each variable.\n"
        "At the very end, write the exact output on a new line as:\n"
        "Output: <exact Python output>\n\n"
        "```python\n"
        f"{item['code']}\n"
        "```\n\n"
        f"Function call: f({item['input']})"
    )
    fallback = (
        f"Python code:\n```python\n{item['code']}\n```\n\n"
        f"Function call: f({item['input']})\n\n"
        "Trace step by step. End with:\nOutput: <result>"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}], fallback


def normalize_output(s: str) -> str:
    """Normalise quoting/whitespace/backticks of a Python literal."""
    s = s.strip()
    s = re.sub(r"^`+(.+)`+$", r"\1", s).strip()
    s = re.sub(r'"([^"]*)"', lambda m: "'" + m.group(1) + "'", s)
    s = re.sub(r"\s*,\s*", ", ", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"\[\s+", "[", s)
    s = re.sub(r"\s+\]", "]", s)
    s = re.sub(r"\{\s+", "{", s)
    s = re.sub(r"\s+\}", "}", s)
    return s


def extract_answer(text: str) -> Optional[str]:
    """``Output: x`` (same line or following line/code block), ``Result:``/``Return:``,
    else the last non-empty line."""
    lines = [l.strip() for l in text.strip().splitlines()]
    nonempty = [l for l in lines if l]
    for line in reversed(nonempty):
        m = re.match(r"(?i)^output\s*:\s*(.+)$", line)
        if m:
            return normalize_output(m.group(1))
    for i, line in enumerate(lines):
        if re.match(r"(?i)^output\s*:\s*$", line):
            for j in range(i + 1, min(i + 6, len(lines))):
                nxt = lines[j]
                if not nxt:
                    continue
                if nxt.startswith("```"):
                    for k in range(j + 1, min(j + 4, len(lines))):
                        inner = lines[k]
                        if inner and inner != "```" and not inner.startswith("```"):
                            return normalize_output(inner)
                    break
                return normalize_output(nxt)
    for line in reversed(nonempty):
        m = re.match(r"(?i)^(?:result|return(?:\s+value)?)\s*:\s*(.+)$", line)
        if m:
            return normalize_output(m.group(1))
    for line in reversed(nonempty):
        if not line.startswith("`"):
            return normalize_output(line)
    return normalize_output(nonempty[-1]) if nonempty else None


def is_correct(pred: Optional[str], gt: str) -> int:
    if pred is None:
        return 0
    p, g = normalize_output(pred), normalize_output(gt)
    if p == g:
        return 1
    if len(g) >= 2 and g[0] == g[-1] and g[0] in ("'", '"') and p == g[1:-1]:
        return 1
    return 0


def score(text: str, item: dict) -> Tuple[Optional[str], int]:
    pred = extract_answer(text)
    return pred, is_correct(pred, item["gt"])
