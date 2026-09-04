from __future__ import annotations

import os

import pytest
import torch
from safetensors.torch import save_file

import vdn_h3.curve_affine as affine


def _save_affine(path, basis, mean, table=None, metadata=None):
    tensors = {"adaln_basis": basis, "adaln_mean": mean}
    if table is not None:
        tensors["adaln_t_table"] = table
    save_file(tensors, path, metadata=metadata or {})


def test_projected_curve_lora_matches_affine_dense_formula():
    torch.manual_seed(410)
    rows, dense, curve, rank, out = 7, 13, 3, 4, 9
    coords = torch.randn(rows, curve, dtype=torch.float64)
    basis = torch.randn(curve, dense, dtype=torch.float64)
    mean = torch.randn(dense, dtype=torch.float64)
    a = torch.randn(rank, dense, dtype=torch.float32)
    b = torch.randn(out, rank, dtype=torch.bfloat16)
    scale = 0.625

    mapping = affine.CurveAffine(
        basis=basis.float(), mean=mean.float(), source="test",
        identity=("test",), table_hash="test")
    projected, offsets = affine.project_curve_terms(
        {"blocks.0.adaln_proj.linear": [(a, b, scale)]}, mapping)
    ap, bp, sp = projected["blocks.0.adaln_proj.linear"][0]
    offset, sb = offsets["blocks.0.adaln_proj.linear"][0]

    # The projection is defined on the stored float32 affine and stored adapter
    # factors. Compare the decomposed pruned expression to the same dense affine
    # expression in float64, then allow only the intentional float32 storage of
    # projected A/constant.
    dense_x = mapping.mean.double() + coords @ mapping.basis.double()
    expected = (dense_x @ a.double().T) @ b.double().T * scale
    got = ((coords.float() @ ap.T) @ bp.float().T + offset) * scale

    assert sp == scale and sb == scale
    assert ap.shape == (rank, curve)
    assert bp.dtype == torch.bfloat16
    assert offset.dtype == torch.float32
    assert torch.allclose(got.double(), expected, atol=4e-4, rtol=4e-4)


def test_local_affine_sidecar_is_authoritative_and_shape_checked(tmp_path, monkeypatch):
    table = torch.randn(9, 3)
    basis = torch.randn(3, 11)
    mean = torch.randn(11)
    stage = tmp_path / "stage"
    stage.mkdir()
    local = stage / affine.AFFINE_FILENAME
    _save_affine(local, basis, mean)
    monkeypatch.setattr(affine, "_candidate_affine_checkpoints", lambda base=None: [])
    affine._AFFINE_CACHE.clear()

    got = affine.find_curve_affine(str(stage), table)
    assert os.path.realpath(got.source) == os.path.realpath(local)
    assert torch.equal(got.basis, basis.float())
    assert torch.equal(got.mean, mean.float())


def test_installed_affine_candidate_must_match_curve_table(tmp_path, monkeypatch):
    table = torch.randn(9, 3)
    wrong = table.clone()
    wrong[0, 0] += 1.0
    candidate = tmp_path / "candidate.safetensors"
    _save_affine(candidate, torch.randn(3, 11), torch.randn(11), wrong)
    stage = tmp_path / "stage"
    stage.mkdir()
    monkeypatch.setattr(
        affine, "_candidate_affine_checkpoints", lambda base=None: [str(candidate)])
    affine._AFFINE_CACHE.clear()

    with pytest.raises(RuntimeError, match="different curve table"):
        affine.find_curve_affine(str(stage), table)


def test_installed_matching_pruned_checkpoint_supplies_affine(tmp_path, monkeypatch):
    table = torch.randn(9, 3)
    basis = torch.randn(3, 11)
    mean = torch.randn(11)
    candidate = tmp_path / "matching-bf16.safetensors"
    _save_affine(candidate, basis, mean, table)
    stage = tmp_path / "stage"
    stage.mkdir()
    monkeypatch.setattr(
        affine, "_candidate_affine_checkpoints", lambda base=None: [str(candidate)])
    affine._AFFINE_CACHE.clear()

    got = affine.find_curve_affine(str(stage), table)
    assert os.path.realpath(got.source) == os.path.realpath(candidate)
    assert torch.equal(got.basis, basis.float())


def test_affine_cache_invalidates_when_candidate_is_replaced(tmp_path, monkeypatch):
    table = torch.randn(9, 3)
    candidate = tmp_path / "matching-bf16.safetensors"
    _save_affine(candidate, torch.ones(3, 11), torch.ones(11), table)
    stage = tmp_path / "stage"
    stage.mkdir()
    monkeypatch.setattr(
        affine, "_candidate_affine_checkpoints", lambda base=None: [str(candidate)])
    affine._AFFINE_CACHE.clear()

    first = affine.find_curve_affine(str(stage), table)
    replacement = tmp_path / "replacement.safetensors"
    _save_affine(replacement, torch.full((3, 11), 2.0), torch.ones(11), table)
    os.replace(replacement, candidate)
    second = affine.find_curve_affine(str(stage), table)

    assert first.identity != second.identity
    assert torch.all(first.basis == 1.0)
    assert torch.all(second.basis == 2.0)


def test_production_width_and_51_curve_modules_project_to_rank8():
    torch.manual_seed(411)
    dense, curve, rank, out = 2688, 8, 2, 12
    mapping = affine.CurveAffine(
        basis=torch.randn(curve, dense), mean=torch.randn(dense),
        source="production-shape", identity=("shape",), table_hash="shape")
    a = torch.randn(rank, dense, dtype=torch.bfloat16)
    b = torch.randn(out, rank, dtype=torch.bfloat16)
    terms = {
        f"blocks.{i}.adaln_proj.linear": [(a, b, 1.0)]
        for i in range(50)
    }
    terms["final_layer.adaln_proj.linear"] = [(a, b, 1.0)]

    projected, offsets = affine.project_curve_terms(terms, mapping)
    assert len(projected) == 51
    assert len(offsets) == 51
    assert all(group[0][0].shape == (rank, curve) for group in projected.values())
    assert all(group[0][0].shape == (out,) for group in offsets.values())
