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
    # Block-diagonal GEMM is mathematically exact but may choose a different FP32
    # reduction schedule than three separate GEMMs.
    assert torch.allclose(got, want, atol=2e-6, rtol=2e-6)


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
    assert torch.allclose(
        (b.float() @ a.float()) * scale,
        torch.cat(individual, dim=0),
        atol=2e-6,
        rtol=2e-6,
    )


def test_incomplete_qkv_triplet_fails_closed():
    a = torch.randn(2, 4)
    b = torch.randn(3, 2)
    state = {}
    for suffix in ("to_q", "to_k"):
        state.update(_pair(f"transformer_blocks.0.attn.orig.{suffix}", a, b))
    with pytest.raises(ValueError, match="incomplete Q/K/V"):
        convert_adapter(state, {"rank": 2, "alpha": 2})


def test_mixed_qkv_rank_and_scale_are_preserved_exactly():
    torch.manual_seed(420)
    inp, out = 4, 3
    ranks = (2, 3, 4)
    suffixes = ("to_q", "to_k", "to_v")
    state = {}
    want = []
    for rank, suffix in zip(ranks, suffixes):
        module = f"transformer_blocks.0.attn.orig.{suffix}"
        a = torch.randn(rank, inp)
        b = torch.randn(out, rank)
        state.update(_pair(module, a, b))
        scale = {"to_q": 0.5, "to_k": 2.0 / 3.0, "to_v": 1.25}[suffix]
        want.append((b.float() @ a.float()) * scale)

    config = {
        "rank": 2,
        "alpha": 1,
        "rank_pattern": {"to_k": 3, "to_v": 4},
        "alpha_pattern": {"to_k": 2, "to_v": 5},
    }
    a, b, scale = convert_adapter(state, config)["blocks.0.attn.qkv_proj"]
    assert a.shape == (sum(ranks), inp)
    assert b.shape == (3 * out, sum(ranks))
    assert scale == 1.0
    assert torch.allclose(
        (b.float() @ a.float()) * scale,
        torch.cat(want, dim=0),
        atol=2e-6,
        rtol=2e-6,
    )


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


def test_turbo_block_and_final_adaln_targets_map_independently():
    torch.manual_seed(44)
    a = torch.randn(2, 8)
    block_b = torch.randn(12, 2)
    final_b = torch.randn(6, 2)
    state = {}
    state.update(_pair("transformer_blocks.3.adaln_proj.linear", a, block_b))
    state.update(_pair("norm_out.linear", a, final_b))

    converted = convert_adapter(state, {"rank": 2, "alpha": 2})
    assert set(converted) == {
        "blocks.3.adaln_proj.linear",
        "final_layer.adaln_proj.linear",
    }
    assert torch.equal(converted["blocks.3.adaln_proj.linear"][0], a.float())
    assert torch.equal(converted["blocks.3.adaln_proj.linear"][1], block_b.float())
    assert torch.equal(converted["final_layer.adaln_proj.linear"][0], a.float())
    assert torch.equal(converted["final_layer.adaln_proj.linear"][1], final_b.float())