"""Greedy generation and the single scoring forward pass."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from .metrics import DEFAULT_ANSWER_WINDOW, compute_profiles


def load_model(model_id: str, quantize_4bit: bool = False, attn_implementation: Optional[str] = None,
               device: str = "cuda:0"):
    """Causal LM in bfloat16 (or 4-bit NF4) and its tokenizer with left padding."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    kwargs = {}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    if quantize_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
        kwargs["device_map"] = {"": device}
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    else:
        kwargs["torch_dtype"] = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).to(device)
    model.eval()
    return model, tok


def num_layers(model) -> int:
    cfg = getattr(model.config, "text_config", None) or model.config
    return int(cfg.num_hidden_layers)


def _stop_ids(model, tokenizer) -> set:
    ids = set()
    eos = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(eos, int):
        ids.add(eos)
    elif eos is not None:
        ids.update(int(e) for e in eos)
    for t in (tokenizer.eos_token_id, tokenizer.pad_token_id):
        if t is not None:
            ids.add(int(t))
    return ids


@torch.no_grad()
def generate(model, tokenizer, prompts: Sequence[str], max_new_tokens: int = 1024) -> List[Dict[str, torch.Tensor]]:
    """Batched greedy decoding. Returns unpadded ``prompt_ids`` and ``response_ids``
    (cut after the first stop token) for each prompt."""
    batch = tokenizer(list(prompts), return_tensors="pt", padding=True, truncation=False).to(model.device)
    out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                         top_p=None, top_k=None, pad_token_id=tokenizer.pad_token_id)
    padded_len = batch["input_ids"].shape[1]
    stops = _stop_ids(model, tokenizer)
    results = []
    for b in range(out.shape[0]):
        prompt_ids = batch["input_ids"][b][batch["attention_mask"][b].bool()]
        resp = out[b, padded_len:]
        cut = len(resp)
        for j, t in enumerate(resp.tolist()):
            if t in stops:
                cut = j + 1
                break
        results.append({"prompt_ids": prompt_ids, "response_ids": resp[:cut]})
    return results


@torch.no_grad()
def score_response(model, prompt_ids: torch.Tensor, response_ids: torch.Tensor,
                   answer_window: int = DEFAULT_ANSWER_WINDOW) -> Dict[str, object]:
    """One forward pass over prompt + response: VANE profiles and mean token log-probability."""
    ids = torch.cat([prompt_ids, response_ids]).unsqueeze(0).to(model.device)
    p = int(prompt_ids.shape[0])
    out = model(ids, attention_mask=torch.ones_like(ids), output_hidden_states=True)
    H = torch.stack([h[0, p:, :] for h in out.hidden_states[1:]], dim=0)
    rec = compute_profiles(H, answer_window=answer_window)

    T = int(response_ids.shape[0])
    if T > 0:
        lp = torch.log_softmax(out.logits[0, p - 1:-1, :].float(), dim=-1)
        rec["mean_log_prob"] = float(lp[torch.arange(T, device=lp.device), ids[0, p:]].mean().item())
    else:
        rec["mean_log_prob"] = 0.0
    return rec
