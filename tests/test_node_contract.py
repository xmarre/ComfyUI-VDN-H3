from __future__ import annotations

import pytest
import torch

from vdn_h3.nodes import (
    ApplyVDNH3,
    ApplyVDNH3Advanced,
    _effective_free_vram,
    _validate_branch_shapes,
)


def _cfg(*, gate=True, short_conv=("k", "v")):
    return {
        "linear_head_dim": 4,
        "enable_softmax_gate": gate,
        "short_conv": short_conv,
    }


def _weights(*, hidden=8, heads=2, dim=4, gate=True, short_conv=("k", "v")):
    out = {
        "to_out_linear.weight": torch.zeros(hidden, heads * dim),
        "beta_proj.weight": torch.zeros(heads, hidden),
        "norm.weight": torch.zeros(dim),
        "alpha.A_log": torch.zeros(heads),
        "alpha.dt_bias": torch.zeros(heads * dim),
        "alpha.down.weight": torch.zeros(dim, hidden),
        "alpha.up.weight": torch.zeros(heads * dim, dim),
        "output_gate.down.weight": torch.zeros(dim, hidden),
        "output_gate.up.weight": torch.zeros(heads * dim, dim),
        "output_gate.up.bias": torch.zeros(heads * dim),
    }
    if gate:
        out["softmax_gate.up.weight"] = torch.zeros(heads, hidden)
        out["softmax_gate.up.bias"] = torch.zeros(heads)
    channels = heads * dim
    for target in short_conv:
        out[f"short_conv.{target}_sp.weight"] = torch.zeros(channels, 1, 5, 5)
        out[f"short_conv.{target}_tm.weight"] = torch.zeros(channels, 1, 5)
    return out


def test_all_block_shapes_match_official_constructor_contract():
    branches = [_weights(), _weights(), _weights()]
    _validate_branch_shapes("synthetic", branches, _cfg(), 8, 2, 4)


def test_shape_validation_checks_later_blocks_not_only_block_zero():
    branches = [_weights(), _weights(), _weights()]
    branches[2]["alpha.up.weight"] = torch.zeros(7, 4)
    with pytest.raises(RuntimeError, match=r"block 2: alpha\.up\.weight"):
        _validate_branch_shapes("synthetic", branches, _cfg(), 8, 2, 4)


def test_feature_disabled_tensors_are_not_required():
    branches = [_weights(gate=False, short_conv=())]
    _validate_branch_shapes(
        "synthetic", branches, _cfg(gate=False, short_conv=()), 8, 2, 4)


def test_linear_head_dim_incompatible_with_shared_qkv_fails_closed():
    cfg = _cfg()
    cfg["linear_head_dim"] = 3
    with pytest.raises(RuntimeError, match="linear_head_dim=3"):
        _validate_branch_shapes("synthetic", [_weights()], cfg, 8, 2, 4)


def test_effective_free_vram_reserves_unloaded_base_bytes(monkeypatch):
    gib = 1 << 30

    class Model:
        @staticmethod
        def model_size():
            return 10 * gib

        @staticmethod
        def loaded_size():
            return 3 * gib

    monkeypatch.setattr("vdn_h3.nodes._free_vram", lambda: 12 * gib)
    assert _effective_free_vram(Model()) == 5 * gib


def test_effective_free_vram_never_goes_negative(monkeypatch):
    gib = 1 << 30

    class Model:
        @staticmethod
        def model_size():
            return 12 * gib

        @staticmethod
        def loaded_size():
            return 0

    monkeypatch.setattr("vdn_h3.nodes._free_vram", lambda: 8 * gib)
    assert _effective_free_vram(Model()) == 0


def test_base_node_exposes_safe_runtime_and_v14_auto_controls(monkeypatch):
    monkeypatch.setattr("vdn_h3.spec.list_vdn_checkpoints", lambda: ["stage"])
    required = ApplyVDNH3.INPUT_TYPES()["required"]
    assert required["lora_mode"][0] == ["merge", "bypass"]
    assert required["lora_mode"][1]["default"] == "merge"
    assert required["branch_weights"][0] == ["auto", "stream", "resident"]
    assert required["branch_weights"][1]["default"] == "auto"
    assert required["retain_buffers"][0] == ["auto", "on", "off"]
    assert required["retain_buffers"][1]["default"] == "auto"


def test_advanced_node_keeps_checkpoint_architecture_as_default(monkeypatch):
    monkeypatch.setattr("vdn_h3.spec.list_vdn_checkpoints", lambda: ["stage"])
    required = ApplyVDNH3Advanced.INPUT_TYPES()["required"]
    assert required["architecture_mode"][0] == ["checkpoint", "override"]
    assert required["architecture_mode"][1]["default"] == "checkpoint"
    assert required["branch_weights"][1]["default"] == "auto"
    assert required["retain_buffers"][1]["default"] == "auto"
