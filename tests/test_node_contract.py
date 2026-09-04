from __future__ import annotations

import pytest
import torch

from vdn_h3.nodes import _validate_branch_shapes


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
