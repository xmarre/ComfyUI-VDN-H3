from __future__ import annotations

import pytest
import torch

from vdn_h3.adapters import convert_adapter


def _pair(module, a, b):
    return {
        f"{module}.lora_A.weight": a,
        f"{module}.lora_B.weight": b,
    }


def test_qkv_fusion_preserves_individual_delta_matrices():
    torch.manual_seed(41)
    rank, inp, out = 3, 7, 5
    state = {}
    individual = []
    for suffix in ("to_q", "to_k", "to_v"):
        module = f"transformer_blocks.2.attn.orig.{suffix}"
        a = torch.randn(rank, inp)
        b = torch.randn(out, rank)
        state.update(_pair(module, a, b))
        individual.append(b.float() @ a.float())

    converted = convert_adapter(state, {"rank": rank, "alpha": rank})
    a, b, scale = converted["blocks.2.attn.qkv_proj"]
    got = (b.float() @ a.float()) * scale
    want = torch.cat(individual, dim=0)
    assert got.shape == want.shape
    assert torch.equal(got, want)


def test_qkv_fusion_preserves_common_alpha_scale():
    torch.manual_seed(42)
    rank, inp, out = 2, 4, 3
    state = {}
    individual = []
    for suffix in ("to_q", "to_k", "to_v"):
        module = f"transformer_blocks.0.attn.orig.{suffix}"
        a = torch.randn(rank, inp)
        b = torch.randn(out, rank)
        state.update(_pair(module, a, b))
        individual.append((b.float() @ a.float()) * 0.5)

    converted = convert_adapter(state, {"rank": rank, "alpha": 1})
    a, b, scale = converted["blocks.0.attn.qkv_proj"]
    assert scale == 0.5
    assert torch.equal((b.float() @ a.float()) * scale, torch.cat(individual, dim=0))


def test_incomplete_qkv_triplet_fails_closed():
    a = torch.randn(2, 4)
    b = torch.randn(3, 2)
    state = {}
    for suffix in ("to_q", "to_k"):
        state.update(_pair(f"transformer_blocks.0.attn.orig.{suffix}", a, b))
    with pytest.raises(ValueError, match="incomplete Q/K/V"):
        convert_adapter(state, {"rank": 2, "alpha": 2})


def test_mixed_qkv_rank_fails_before_model_patching():
    state = {}
    for rank, suffix in zip((2, 3, 2), ("to_q", "to_k", "to_v")):
        state.update(_pair(
            f"transformer_blocks.0.attn.orig.{suffix}",
            torch.randn(rank, 4), torch.randn(3, rank)))
    config = {
        "rank": 2,
        "alpha": 2,
        "rank_pattern": {"to_k": 3},
        "alpha_pattern": {"to_k": 3},
    }
    with pytest.raises(ValueError, match="mixed LoRA ranks"):
        convert_adapter(state, config)


def test_swiglu_b_half_swap_matches_comfy_gate_value_order():
    torch.manual_seed(43)
    a = torch.randn(2, 4)
    b = torch.randn(6, 2)
    module = "transformer_blocks.0.ff.net.0.proj"
    converted = convert_adapter(_pair(module, a, b), {"rank": 2, "alpha": 2})
    got_a, got_b, scale = converted["blocks.0.mlp.fc1"]
    value, gate = b.chunk(2, dim=0)
    assert torch.equal(got_a, a.float())
    assert torch.equal(got_b, torch.cat([gate, value], dim=0).float())
    assert scale == 1.0


def test_missing_lora_side_fails_with_context():
    state = {
        "transformer_blocks.0.attn.orig.to_out.0.lora_A.weight": torch.randn(2, 4),
    }
    with pytest.raises(ValueError, match="exactly A and B"):
        convert_adapter(state, {"rank": 2, "alpha": 2})
