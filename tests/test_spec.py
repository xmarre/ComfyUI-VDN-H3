from __future__ import annotations

import json
import os

import pytest
import torch
from safetensors.torch import save_file

import vdn_h3.spec as spec


def _model_spec(*, softmax_gate=False, short_conv=(), extra_linear=None):
    linear = {
        "delta_rule": "vdn_solve",
        "bridge": "alpha",
        "a_fp32": True,
        "linear_head_dim": 4,
        "short_conv": {"targets": list(short_conv)},
        "enable_text_state": True,
    }
    if extra_linear:
        linear.update(extra_linear)
    return {
        "format_version": 2,
        "base": {
            "library": "diffusers",
            "class_name": "MiniMaxH3Transformer2DModel",
            "source": "example/base",
            "subfolder": "transformer",
            "resolved_config": {
                "hidden_size": 8,
                "num_layers": 1,
                "num_attention_heads": 2,
                "attention_head_dim": 4,
            },
        },
        "transforms": [{
            "type": "hybrid_attention",
            "version": 2,
            "config": {
                "enable_softmax_gate": softmax_gate,
                "anchor_frames": "both",
                "softmax_attention": {"radius": 1, "chunk": 5},
                "linear_attention": linear,
            },
        }],
        "adapters": [],
    }


def _required_branch_state(*, softmax_gate=False, short_conv=(), fill=1.0):
    cfg = spec.transform_config(
        _model_spec(softmax_gate=softmax_gate, short_conv=short_conv))
    tensors = {}
    for name in sorted(spec._required_branch_tensors(cfg)):
        key = f"transformer_blocks.0.attn.linear_attention.{name}"
        tensors[key] = torch.full((1,), float(fill))
    return tensors


def _write_stage(root, *, fill=1.0, softmax_gate=False, short_conv=()):
    stage = root / "stage"
    branch_dir = stage / "linear_branch"
    branch_dir.mkdir(parents=True, exist_ok=True)
    (stage / "model_spec.json").write_text(
        json.dumps(_model_spec(
            softmax_gate=softmax_gate, short_conv=short_conv)),
        encoding="utf-8",
    )
    save_file(
        _required_branch_state(
            softmax_gate=softmax_gate, short_conv=short_conv, fill=fill),
        branch_dir / spec.BRANCH_FILE,
    )
    return stage


def test_valid_model_spec_round_trips_checkpoint_architecture():
    cfg = spec.transform_config(_model_spec(softmax_gate=True, short_conv=("k", "v")))
    assert cfg == {
        "enable_softmax_gate": True,
        "anchor_frames": "both",
        "radius": 1,
        "chunk": 5,
        "delta_rule": "vdn_solve",
        "bridge": "alpha",
        "a_fp32": True,
        "linear_head_dim": 4,
        "short_conv": ("k", "v"),
        "enable_text_state": True,
    }


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda p: p.__setitem__("format_version", 1), "format_version"),
        (lambda p: p["transforms"][0].__setitem__("version", 1), "transform version"),
        (lambda p: p["transforms"][0]["config"]["linear_attention"].__setitem__(
            "delta_rule", "unknown"), "delta_rule"),
        (lambda p: p["transforms"][0]["config"]["linear_attention"].__setitem__(
            "bridge", "unknown"), "bridge"),
        (lambda p: p["transforms"][0]["config"].__setitem__(
            "anchor_frames", "future"), "anchor_frames"),
        (lambda p: p["transforms"][0]["config"]["softmax_attention"].__setitem__(
            "radius", None), "unresolved"),
        (lambda p: p["transforms"][0]["config"].__setitem__(
            "softmax_backend", "flash"), "runtime keys"),
    ],
)
def test_malformed_model_specs_fail_at_load_contract(mutator, match):
    payload = _model_spec()
    mutator(payload)
    with pytest.raises(ValueError, match=match):
        spec.validate_model_spec(payload)


def test_exactly_one_hybrid_transform_is_required():
    payload = _model_spec()
    payload["transforms"].append(payload["transforms"][0].copy())
    with pytest.raises(ValueError, match="exactly one hybrid_attention"):
        spec.validate_model_spec(payload)


def test_required_branch_tensors_follow_enabled_features():
    plain = spec._required_branch_tensors(spec.transform_config(_model_spec()))
    enabled = spec._required_branch_tensors(spec.transform_config(
        _model_spec(softmax_gate=True, short_conv=("k", "v"))))
    assert "softmax_gate.up.weight" not in plain
    assert "short_conv.k_sp.weight" not in plain
    assert {
        "softmax_gate.up.weight",
        "softmax_gate.up.bias",
        "short_conv.k_sp.weight",
        "short_conv.k_tm.weight",
        "short_conv.v_sp.weight",
        "short_conv.v_tm.weight",
    } <= enabled


def test_split_branches_reports_missing_feature_tensor_with_block(tmp_path):
    cfg = spec.transform_config(_model_spec(softmax_gate=True))
    state = _required_branch_state(softmax_gate=True)
    state.pop("transformer_blocks.0.attn.linear_attention.softmax_gate.up.bias")
    with pytest.raises(ValueError, match=r"block 0.*softmax_gate\.up\.bias"):
        spec._split_branches(str(tmp_path), state, cfg)


def test_branch_block_indices_must_be_contiguous(tmp_path):
    cfg = spec.transform_config(_model_spec())
    state = _required_branch_state()
    second = {
        key.replace("transformer_blocks.0", "transformer_blocks.2"): value
        for key, value in state.items()
    }
    state.update(second)
    with pytest.raises(ValueError, match="not contiguous"):
        spec._split_branches(str(tmp_path), state, cfg)


def test_lazy_descriptor_observes_replacement_as_stale(tmp_path):
    branch = tmp_path / "branch.safetensors"
    key = "transformer_blocks.0.attn.linear_attention.alpha.A_log"
    save_file({key: torch.tensor([1.0])}, branch)
    lazy = spec._lazy_branch_sd(str(branch))
    descriptor = lazy[key]
    assert isinstance(descriptor, spec.LazyBranchTensor)
    resolved = spec.resolve_branch_weights({"alpha.A_log": descriptor}, "cpu")
    assert torch.equal(resolved["alpha.A_log"], torch.tensor([1.0]))

    replacement = tmp_path / "replacement.safetensors"
    save_file({key: torch.tensor([2.0])}, replacement)
    os.replace(replacement, branch)

    with pytest.raises(RuntimeError, match="changed after it was loaded"):
        spec.resolve_branch_weights({"alpha.A_log": descriptor}, "cpu")

    fresh = spec._lazy_branch_sd(str(branch))[key]
    resolved = spec.resolve_branch_weights({"alpha.A_log": fresh}, "cpu")
    assert torch.equal(resolved["alpha.A_log"], torch.tensor([2.0]))


def test_checkpoint_cache_invalidates_when_branch_replaced(tmp_path):
    spec._CACHE.clear()
    stage = _write_stage(tmp_path, fill=1.0)
    cfg1, branches1, _ = spec.load_vdn_checkpoint(str(stage))
    first = spec.resolve_branch_weights(branches1[0], "cpu")
    assert torch.equal(first["alpha.A_log"], torch.tensor([1.0]))

    replacement = tmp_path / "new_branch.safetensors"
    save_file(_required_branch_state(fill=3.0), replacement)
    os.replace(replacement, stage / "linear_branch" / spec.BRANCH_FILE)

    cfg2, branches2, _ = spec.load_vdn_checkpoint(str(stage))
    second = spec.resolve_branch_weights(branches2[0], "cpu")
    assert cfg2 == cfg1
    assert torch.equal(second["alpha.A_log"], torch.tensor([3.0]))
    assert branches2[0]["alpha.A_log"].identity != branches1[0]["alpha.A_log"].identity


def test_truncated_branch_header_fails_closed(tmp_path):
    broken = tmp_path / "broken.safetensors"
    broken.write_bytes((256).to_bytes(8, "little") + b"{}")
    with pytest.raises(ValueError, match="truncated safetensors JSON header"):
        spec._read_header(str(broken))


def test_adapter_requires_complete_pairs(tmp_path):
    state = {
        "transformer_blocks.0.attn.orig.to_q.lora_A.weight": torch.randn(2, 4),
    }
    with pytest.raises(ValueError, match="both A and B"):
        spec._validate_adapter_weights(
            "default", state, {"rank": 2, "alpha": 2}, str(tmp_path))


@pytest.mark.parametrize("missing", ["branch", "spec"])
def test_missing_stage_file_has_contextual_error(tmp_path, missing):
    stage = _write_stage(tmp_path)
    target = stage / "linear_branch" / spec.BRANCH_FILE if missing == "branch" else stage / "model_spec.json"
    target.unlink()
    with pytest.raises(FileNotFoundError, match="missing " + ("linear_branch/" if missing == "branch" else "model_spec.json")):
        spec.load_vdn_checkpoint(str(stage))
