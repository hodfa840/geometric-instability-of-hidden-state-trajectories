import numpy as np
import pytest
import torch

from vane.features import build_features
from vane.metrics import compute_profiles, per_token_metrics, profile_lengths


def straight_line(L=12, T=5, d=16, speed=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    start = torch.randn(T, d, generator=g)
    direction = torch.randn(T, d, generator=g)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    steps = torch.arange(L, dtype=torch.float32).view(L, 1, 1) * speed
    return start.unsqueeze(0) + steps * direction.unsqueeze(0)


def test_shapes():
    L, T = 10, 7
    m = per_token_metrics(torch.randn(L, T, 8))
    lengths = profile_lengths(L)
    assert set(m) == set(lengths)
    for k, v in m.items():
        assert v.shape == (lengths[k], T), k


def test_straight_line_constant_speed():
    m = per_token_metrics(straight_line(speed=2.5))
    assert torch.allclose(m["vel"], torch.full_like(m["vel"], 2.5), atol=1e-4)
    assert m["jerk"].abs().max() < 1e-4          # constant velocity
    assert m["curv"].abs().max() < 1e-3          # no turning
    assert m["geodev"].abs().max() < 1e-4        # on the chord


def test_token_coherence_zero_for_shared_direction():
    L, T, d = 8, 6, 4
    direction = torch.randn(d)
    H = torch.randn(1, T, d) + torch.arange(L, dtype=torch.float32).view(L, 1, 1) * direction
    assert per_token_metrics(H)["tokc"].abs().max() < 1e-5


def test_curvature_of_right_angle():
    # e1, then e2, then e2: one 90-degree turn -> ||T_{l+1} - T_l|| = sqrt(2), then 0
    pts = torch.tensor([[0., 0.], [1., 0.], [1., 1.], [1., 2.]]).unsqueeze(1)   # (L=4, T=1, d=2)
    curv = per_token_metrics(pts)["curv"][:, 0]
    assert torch.allclose(curv, torch.tensor([2 ** 0.5, 0.0]), atol=1e-5)


def test_geodesic_deviation_degenerate_chord_is_zero():
    H = torch.zeros(6, 3, 5)
    H[2] = 1.0                                   # leaves and returns: h_L == h_1
    assert torch.all(per_token_metrics(H)["geodev"] == 0)


def test_answer_window():
    H = torch.randn(6, 30, 4)
    vel = per_token_metrics(H)["vel"]
    rec = compute_profiles(H, answer_window=5)
    np.testing.assert_allclose(rec["vel_ans"], vel[:, -5:].max(-1).values.numpy(), rtol=1e-6)
    np.testing.assert_allclose(rec["vel_max"], vel.max(-1).values.numpy(), rtol=1e-6)
    np.testing.assert_allclose(rec["vel_mean"], vel.mean(-1).numpy(), rtol=1e-6)
    short = compute_profiles(H[:, :3], answer_window=5)
    np.testing.assert_allclose(short["vel_ans"], short["vel_max"])


def test_feature_dim():
    # 3 aggregations x (30 + 30 + 31 + 32 + 31) per-layer values for L = 32 (Llama-3-8B)
    assert build_features(compute_profiles(torch.randn(32, 9, 8))).shape == (462,)
    assert build_features(compute_profiles(torch.randn(32, 9, 8)), ("velocity",)).shape == (93,)


def test_empty_generation_keeps_shapes():
    empty = build_features(compute_profiles(torch.randn(16, 0, 4)))
    assert empty.shape == build_features(compute_profiles(torch.randn(16, 3, 4))).shape
    assert not empty.any()


def test_unknown_metric():
    with pytest.raises(ValueError):
        build_features(compute_profiles(torch.randn(8, 3, 4)), ("nope",))
