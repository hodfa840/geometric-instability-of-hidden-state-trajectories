"""HumanEval (Chen et al., 2021). Scoring executes model-generated code: run in a sandbox."""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from typing import List, Optional, Tuple

NAME = "humaneval"

SYSTEM = (
    "You are an expert Python programmer. "
    "Reason carefully through each algorithm step by step, "
    "then write clean, correct, efficient code."
)
TIMEOUT_S = 10
ALLOW_EXECUTION = False  # set by scripts/extract.py --allow_code_execution


def load(max_samples: Optional[int] = None, **_) -> List[dict]:
    from datasets import load_dataset

    ds = load_dataset("openai/openai_humaneval", split="test")
    n = len(ds) if max_samples is None else min(max_samples, len(ds))
    return [
        {"idx": i, "task_id": ds[i]["task_id"], "prompt": ds[i]["prompt"], "test": ds[i]["test"],
         "entry_point": ds[i]["entry_point"], "gt": ds[i]["canonical_solution"]}
        for i in range(n)
    ]


def messages(item: dict):
    content = (
        "Implement the following Python function.\n"
        "Think through the algorithm step by step, considering edge cases.\n"
        "Then write the COMPLETE implementation "
        "(including the function signature) inside a ```python...``` code block.\n\n"
        "```python\n"
        f"{item['prompt']}"
        "```"
    )
    fallback = (
        "Implement this Python function. Think step by step, then write "
        "the complete function inside a ```python...``` code block.\n\n"
        f"```python\n{item['prompt']}\n```"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}], fallback


def extract_code(text: str, prompt: str) -> str:
    """Last fenced block containing ``def``; else indented body; else ``pass``."""
    for pattern in (r"```python\s*\n(.*?)```", r"```\s*\n(.*?)```"):
        for m in reversed(list(re.finditer(pattern, text, re.DOTALL))):
            code = m.group(1).strip()
            if "def " in code:
                return code
    body: List[str] = []
    for line in text.split("\n"):
        if line.startswith("    ") or line.startswith("\t"):
            body.append(line)
        elif body and line.strip() == "":
            body.append("")
        elif body:
            break
    if body:
        while body and not body[-1].strip():
            body.pop()
        return prompt + "\n".join(body) + "\n"
    return prompt + "    pass\n"


def _limit_resources(timeout: int):
    def apply():
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout))
        resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
        resource.setrlimit(resource.RLIMIT_FSIZE, (10 << 20, 10 << 20))
    return apply if os.name == "posix" else None


def run_tests(code: str, test: str, entry_point: str, timeout: int = TIMEOUT_S) -> bool:
    if not ALLOW_EXECUTION:
        raise RuntimeError("HumanEval scoring executes model-generated code; "
                           "enable it explicitly with --allow_code_execution (preferably in a sandbox).")
    program = code + "\n\n" + test + f"\ncheck({entry_point})\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "candidate.py")
        with open(path, "w") as f:
            f.write(program)
        try:
            res = subprocess.run([sys.executable, "-I", path], capture_output=True, timeout=timeout, cwd=tmp,
                                 preexec_fn=_limit_resources(timeout))
            return res.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False


def score(text: str, item: dict) -> Tuple[Optional[str], int]:
    code = extract_code(text, item["prompt"])
    return code, int(run_tests(code, item["test"], item["entry_point"]))
