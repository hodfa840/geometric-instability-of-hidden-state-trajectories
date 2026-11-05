"""Benchmarks: loading, prompts and answer checking."""
from __future__ import annotations

from importlib import import_module
from types import ModuleType

TASKS = ("gsm8k", "math500", "humaneval", "cruxeval", "bbh")


def get_task(name: str) -> ModuleType:
    if name not in TASKS:
        raise ValueError(f"unknown benchmark '{name}'; choose from {TASKS}")
    return import_module(f"vane.tasks.{name}")


def render_prompt(tokenizer, messages, fallback: str) -> str:
    """Chat-template prompt (Qwen3 in non-thinking mode); ``fallback`` without a template."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        pass
    except Exception:
        return fallback
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return fallback
