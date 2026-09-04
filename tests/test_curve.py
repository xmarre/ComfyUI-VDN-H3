from __future__ import annotations

import os

import pytest
import torch
from torch import nn
from safetensors.torch import save_file

import vdn_h3.curve as curve


def _dense_tensors(fill: float = 0.0):
    return {
        "time_embedder.proj_in.weight": torch.full((4, 256), fill),
        "time_embedder.proj_in.bias": torch.full((4,), fill),
        "time_embedder.proj_out.weight": torch.full((8, 4), fill),
        "time_embedder.proj_out.bias": torch.full((8,), fill),
    }


def test_curve_state_nested_reset_restores_outer_value():
    state = curve.CurveAdalnState()
    outer = torch.tensor([[1.0]])
    inner = torch.tensor([[2.0]])

    outer_token = state.push(outer)
    assert state.get() is outer
    inner_token = state.push(inner)
    assert state.get() is inner

    state.reset(inner_token)
    assert state.get() is outer
    state.reset(outer_token)
    assert state.get() is None


def test_dense_curve_wrapper_is_reentrant(monkeypatch):
    state = curve.CurveAdalnState()

    class Embedder:
        def silu_grid(self, t):
            return t.reshape(-1, 1)

    monkeypatch.setattr(
        curve,
        "minimax_unique_timesteps",
        lambda dm, x, timestep, context, *args, **kwargs: [float(timestep.item())],
    )
    wrapper = curve.make_dense_curve_wrapper(object(), Embedder(), state)
    context = torch.zeros(1, 1, 1)
    x = (torch.zeros(1), torch.zeros(1))

    def inner_executor(*args, **kwargs):
        assert torch.equal(state.get(), torch.tensor([[0.75]]))
        return "inner"

    def outer_executor(*args, **kwargs):
        assert torch.equal(state.get(), torch.tensor([[0.25]]))
        result = wrapper(inner_executor, x, torch.tensor([0.75]), context, {})
        assert result == "inner"
        assert torch.equal(state.get(), torch.tensor([[0.25]]))
        return "outer"

    assert wrapper(outer_executor, x, torch.tensor([0.25]), context, {}) == "outer"
    assert state.get() is None


class _CurveAdaln(nn.Module):
    def __init__(self):
        super().__init__()
        self.apply_silu = False
        self.modalities = 1
        self.expand = 2
        self.hidden = 3
        self.linear = nn.Linear(2, self.modalities * self.expand * self.hidden, bias=True)


def _flatten_chunks(chunks):
    return torch.cat(chunks, dim=-1)


def test_curve_adaln_adds_original_dense_low_rank_delta_exactly():
    torch.manual_seed(301)
    base = _CurveAdaln()
    state = curve.CurveAdalnState()
    curve_input = torch.randn(4, 2)
    dense_input = torch.randn(4, 5)
    a = torch.randn(3, 5)
    b = torch.randn(6, 3)
    scale = 0.625

    forward = curve.make_curve_adaln_forward(base, [(a, b, scale)], state)
    token = state.push(dense_input)
    try:
        got = _flatten_chunks(forward(curve_input))
    finally:
        state.reset(token)

    base_out = base.linear(curve_input)
    expected = base_out + torch.nn.functional.linear(
        torch.nn.functional.linear(dense_input, a), b) * scale
    assert torch.allclose(got, expected, atol=2e-6, rtol=2e-6)


def test_curve_adaln_strength_scaling_is_linear_and_does_not_touch_base_curve():
    torch.manual_seed(302)
    base = _CurveAdaln()
    base_weight = base.linear.weight.detach().clone()
    base_bias = base.linear.bias.detach().clone()
    state = curve.CurveAdalnState()
    curve_input = torch.randn(3, 2)
    dense_input = torch.randn(3, 5)
    a = torch.randn(2, 5)
    b = torch.randn(6, 2)

    full = curve.make_curve_adaln_forward(base, [(a, b, 1.0)], state)
    half = curve.make_curve_adaln_forward(base, [(a, b, 0.5)], state)
    token = state.push(dense_input)
    try:
        y_full = _flatten_chunks(full(curve_input))
        y_half = _flatten_chunks(half(curve_input))
    finally:
        state.reset(token)
    y_base = base.linear(curve_input)

    assert torch.allclose(y_half - y_base, 0.5 * (y_full - y_base), atol=2e-6, rtol=2e-6)
    assert torch.equal(base.linear.weight, base_weight)
    assert torch.equal(base.linear.bias, base_bias)


def test_curve_adaln_requires_execution_local_dense_state():
    base = _CurveAdaln()
    state = curve.CurveAdalnState()
    a = torch.randn(2, 5)
    b = torch.randn(6, 2)
    forward = curve.make_curve_adaln_forward(base, [(a, b, 1.0)], state)
    with pytest.raises(RuntimeError, match="state is unavailable"):
        forward(torch.randn(2, 2))


def test_root_prefix_is_valid_dense_checkpoint_candidate(tmp_path, monkeypatch):
    candidate = tmp_path / "dense.safetensors"
    save_file(_dense_tensors(0.125), candidate)
    monkeypatch.setattr(curve, "_candidate_dense_checkpoints", lambda: [str(candidate)])
    monkeypatch.setattr(curve, "_curve_fit_residual", lambda table, dense: 0.0)
    curve._EMBEDDER_CACHE.clear()

    header = curve._safetensors_header(str(candidate))
    assert curve._find_dense_prefix(header) == ""

    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    embedder, residual = curve.find_dense_time_embedder(
        str(stage_dir), torch.zeros(3, 2))
    assert os.path.realpath(embedder.source) == os.path.realpath(candidate)
    assert residual == 0.0
    assert torch.all(embedder.proj_in_weight == 0.125)


def test_local_companion_is_authoritative(tmp_path, monkeypatch):
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    local = stage_dir / curve.TIME_EMBEDDER_FILENAME
    installed = tmp_path / "installed.safetensors"
    save_file(_dense_tensors(0.25), local)
    save_file(_dense_tensors(0.75), installed)
    monkeypatch.setattr(curve, "_candidate_dense_checkpoints", lambda: [str(installed)])
    monkeypatch.setattr(curve, "_curve_fit_residual", lambda table, dense: 0.0)
    curve._EMBEDDER_CACHE.clear()

    embedder, _ = curve.find_dense_time_embedder(str(stage_dir), torch.zeros(3, 2))
    assert os.path.realpath(embedder.source) == os.path.realpath(local)
    assert torch.all(embedder.proj_in_weight == 0.25)


def test_replacing_candidate_file_invalidates_embedder_cache(tmp_path, monkeypatch):
    candidate = tmp_path / "dense.safetensors"
    save_file(_dense_tensors(0.125), candidate)
    monkeypatch.setattr(curve, "_candidate_dense_checkpoints", lambda: [str(candidate)])
    monkeypatch.setattr(curve, "_curve_fit_residual", lambda table, dense: 0.0)
    curve._EMBEDDER_CACHE.clear()

    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    table = torch.zeros(3, 2)
    first, _ = curve.find_dense_time_embedder(str(stage_dir), table)
    first_identity = first.identity
    assert torch.all(first.proj_in_weight == 0.125)

    replacement = tmp_path / "replacement.safetensors"
    save_file(_dense_tensors(0.875), replacement)
    os.replace(replacement, candidate)

    second, _ = curve.find_dense_time_embedder(str(stage_dir), table)
    assert second.identity != first_identity
    assert torch.all(second.proj_in_weight == 0.875)


def test_invalid_curve_table_shape_fails_closed(tmp_path):
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    try:
        curve.find_dense_time_embedder(str(stage_dir), torch.zeros(1, 2))
    except RuntimeError as exc:
        assert "invalid shape" in str(exc)
    else:
        raise AssertionError("invalid one-row AdaLN table was accepted")


def test_truncated_safetensors_header_fails_closed(tmp_path):
    broken = tmp_path / "broken.safetensors"
    broken.write_bytes((128).to_bytes(8, "little") + b"{}")
    try:
        curve._safetensors_header(str(broken))
    except ValueError as exc:
        assert "truncated safetensors JSON header" in str(exc)
    else:
        raise AssertionError("truncated safetensors header was accepted")
