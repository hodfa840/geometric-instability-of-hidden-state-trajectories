#!/usr/bin/env python3
"""Generate responses and extract VANE profiles for one model and benchmark (resumable).

    python scripts/extract.py --model Qwen/Qwen2.5-1.5B-Instruct --benchmark gsm8k --output_dir results
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vane.extraction import generate, load_model, num_layers, score_response  # noqa: E402
from vane.tasks import TASKS, get_task, render_prompt  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Hugging Face model id or local path")
    p.add_argument("--benchmark", required=True, choices=TASKS)
    p.add_argument("--output_dir", default="results")
    p.add_argument("--name", default=None, help="run name (default: model basename)")
    p.add_argument("--max_samples", type=int, default=None, help="use only the first N items")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--answer_window", type=int, default=20)
    p.add_argument("--quantize_4bit", action="store_true", help="NF4 4-bit weights (used for Qwen2.5-72B)")
    p.add_argument("--attn_implementation", default=None, help="e.g. eager (Gemma-3)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--allow_code_execution", action="store_true",
                   help="required for HumanEval: runs model-generated code (use a sandbox)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_name(model: str, quantized: bool) -> str:
    name = re.sub(r"[^a-zA-Z0-9\-]", "-", os.path.basename(model.rstrip("/"))).lower()
    return f"{name}-4bit" if quantized else name


def atomic_pickle(obj, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def main():
    args = parse_args()
    from transformers import set_seed

    set_seed(args.seed)
    task = get_task(args.benchmark)
    if args.benchmark == "humaneval":
        if not args.allow_code_execution:
            sys.exit("HumanEval executes model-generated code: rerun with --allow_code_execution (in a sandbox).")
        task.ALLOW_EXECUTION = True
    name = args.name or run_name(args.model, args.quantize_4bit)
    run_dir = Path(args.output_dir) / f"{name}_{args.benchmark}"
    run_dir.mkdir(parents=True, exist_ok=True)
    rec_path = run_dir / "records.pkl"

    items = task.load(max_samples=args.max_samples)
    records = []
    if rec_path.exists():
        with open(rec_path, "rb") as f:
            records = pickle.load(f)
    done = {r["sample_idx"] for r in records}
    todo = [it for it in items if it["idx"] not in done]
    print(f"[{name} / {args.benchmark}] {len(items)} items, {len(done)} done, {len(todo)} to go -> {run_dir}",
          flush=True)

    model, tok = load_model(args.model, args.quantize_4bit, args.attn_implementation, args.device)
    meta = {"model": args.model, "benchmark": args.benchmark, "n_items": len(items),
            "num_layers": num_layers(model), **{k: v for k, v in vars(args).items() if k not in ("model", "benchmark")}}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    t0 = time.time()
    with open(run_dir / "generations.jsonl", "a") as gen_log:
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start:start + args.batch_size]
            prompts = [render_prompt(tok, *task.messages(it)) for it in batch]
            for it, o in zip(batch, generate(model, tok, prompts, max_new_tokens=args.max_new_tokens)):
                text = tok.decode(o["response_ids"], skip_special_tokens=True)
                pred, correct = task.score(text, it)
                rec = score_response(model, o["prompt_ids"], o["response_ids"], args.answer_window)
                rec.update({"sample_idx": it["idx"], "is_correct": int(correct), "pred": pred, "gt": it["gt"],
                            "truncated": int(len(o["response_ids"]) >= args.max_new_tokens)})
                records.append(rec)
                gen_log.write(json.dumps({"idx": it["idx"], "response": text, "pred": pred, "gt": it["gt"],
                                          "is_correct": int(correct)}) + "\n")
            gen_log.flush()
            atomic_pickle(records, rec_path)
            acc = np.mean([r["is_correct"] for r in records])
            rate = (time.time() - t0) / (start + len(batch))
            print(f"  {len(records)}/{len(items)}  acc={acc:.3f}  {rate:.1f}s/sample", flush=True)

    print(f"done: {len(records)} samples, accuracy {np.mean([r['is_correct'] for r in records]):.4f} -> {rec_path}")


if __name__ == "__main__":
    main()
