from __future__ import annotations

import os

import torch
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
