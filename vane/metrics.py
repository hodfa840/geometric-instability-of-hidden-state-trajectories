"""VANE trajectory metrics (Section 3.2) and token aggregations (Section 3.3)."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-8

METRIC_PREFIXES: Dict[str, str] = {
    "curvature": "curv",
    "jerk": "jerk",
    "velocity": "vel",
    "geodesic_dev": "geodev",
    "token_coherence": "tokc",
}
AGGREGATIONS = ("max", "mean", "ans")
DEFAULT_ANSWER_WINDOW = 20


def per_token_metrics(H: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Per-(layer, token) values of the five metrics.

    H: hidden states of the generated tokens, shape (L, T, d).
    Returns a dict mapping metric prefix to a (n_layers, T) tensor.
    """
    if H.ndim != 3:
        raise ValueError(f"expected (L, T, d) hidden states, got shape {tuple(H.shape)}")
    L = H.shape[0]
    if L < 3:
        raise ValueError(f"need at least 3 layers, got {L}")
    H = H.float()

    delta = H[1:] - H[:-1]
    speed = torch.linalg.norm(delta, dim=-1)
    tangent = delta / (speed.unsqueeze(-1) + EPS)

    out: Dict[str, torch.Tensor] = {}
    out["vel"] = speed
    out["jerk"] = torch.linalg.norm(delta[1:] - delta[:-1], dim=-1)
    out["curv"] = torch.linalg.norm(tangent[1:] - tangent[:-1], dim=-1)

    first, last = H[0], H[-1]
    chord = last - first
    chord_len = torch.linalg.norm(chord, dim=-1)
    t = torch.arange(L, dtype=H.dtype, device=H.device) / (L - 1)
    line = first.unsqueeze(0) + t.view(L, 1, 1) * chord.unsqueeze(0)
    dev = torch.linalg.norm(H - line, dim=-1)
    out["geodev"] = torch.where(chord_len > EPS, dev / (chord_len + EPS), torch.zeros_like(dev))

    mean_t = tangent.mean(dim=1, keepdim=True)
    mean_t = mean_t / (torch.linalg.norm(mean_t, dim=-1, keepdim=True) + EPS)
    out["tokc"] = 1.0 - F.cosine_similarity(tangent, mean_t.expand_as(tangent), dim=-1)
    return out


def aggregate(values: torch.Tensor, answer_window: int = DEFAULT_ANSWER_WINDOW) -> Dict[str, np.ndarray]:
    """Max, mean and answer-window max over tokens of a (n_layers, T) tensor."""
    values = values.float()
    mx = values.max(dim=-1).values
    mn = values.mean(dim=-1)
    ans = values[:, -answer_window:].max(dim=-1).values if values.shape[-1] > answer_window else mx
    return {"max": mx.cpu().numpy(), "mean": mn.cpu().numpy(), "ans": ans.cpu().numpy()}


def profile_lengths(num_layers: int) -> Dict[str, int]:
    L = num_layers
    return {"vel": L - 1, "jerk": L - 2, "curv": L - 2, "geodev": L, "tokc": L - 1}


def compute_profiles(H: torch.Tensor, answer_window: int = DEFAULT_ANSWER_WINDOW) -> Dict[str, object]:
    """Per-layer profiles ``{prefix}_{aggregation}`` for one response.

    H: (L, T, d) hidden states of the generated tokens at the L transformer blocks.
    """
    L, T = int(H.shape[0]), int(H.shape[1])
    result: Dict[str, object] = {}
    if T == 0:
        for prefix, n in profile_lengths(L).items():
            for agg in AGGREGATIONS:
                result[f"{prefix}_{agg}"] = np.zeros(n, dtype=np.float32)
    else:
        for prefix, values in per_token_metrics(H).items():
            for agg, prof in aggregate(values, answer_window).items():
                result[f"{prefix}_{agg}"] = prof
    result["gen_length"] = T
    result["num_layers"] = L
    return result
