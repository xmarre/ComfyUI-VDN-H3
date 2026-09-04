from __future__ import annotations

import os

import pytest
import torch
from safetensors.torch import save_file

import vdn_h3.curve_affine as affine


def _stage(tmp_path):
    path = tmp_path / "stage"
    path.mkdir()
    return path


def _save(path, basis, mean, table=None, metadata=None):
    tensors = {"adaln_basis": basis, "adaln_mean": mean}
    if table is not None:
        tensors["adaln_t_table"] = table
    save_file(tensors, path, metadata=metadata or {})


def test_local_affine_rejects_nonmatrix_basis(tmp_path, monkeypatch):
    table = torch.randn(9, 3)
    stage = _stage(tmp_path)
    _save(stage / affine.AFFINE_FILENAME, torch.randn(3), torch.randn(3))
    monkeypatch.setattr(affine, "_candidate_affine_checkpoints", lambda base=None: [])
    affine._AFFINE_CACHE.clear()
    with pytest.raises(RuntimeError, match="expected adaln_basis"):
        affine.find_curve_affine(str(stage), table)


def test_local_affine_rejects_curve_rank_mismatch(tmp_path, monkeypatch):
    table = torch.randn(9, 3)
    stage = _stage(tmp_path)
    _save(stage / affine.AFFINE_FILENAME, torch.randn(4, 11), torch.randn(11))
    monkeypatch.setattr(affine, "_candidate_affine_checkpoints", lambda base=None: [])
    affine._AFFINE_CACHE.clear()
    with pytest.raises(RuntimeError, match="does not match adaln_t_table"):
        affine.find_curve_affine(str(stage), table)


def test_installed_affine_without_table_identity_is_rejected(tmp_path, monkeypatch):
    table = torch.randn(9, 3)
    candidate = tmp_path / "unverified.safetensors"
    _save(candidate, torch.randn(3, 11), torch.randn(11))
    stage = _stage(tmp_path)
    monkeypatch.setattr(
        affine, "_candidate_affine_checkpoints", lambda base=None: [str(candidate)])
    affine._AFFINE_CACHE.clear()
    with pytest.raises(RuntimeError, match="no curve-table identity"):
        affine.find_curve_affine(str(stage), table)


def test_local_affine_bad_declared_table_hash_is_rejected(tmp_path, monkeypatch):
    table = torch.randn(9, 3)
    stage = _stage(tmp_path)
    _save(
        stage / affine.AFFINE_FILENAME,
        torch.randn(3, 11), torch.randn(11),
        metadata={"adaln_table_sha256": "0" * 64},
    )
    monkeypatch.setattr(affine, "_candidate_affine_checkpoints", lambda base=None: [])
    affine._AFFINE_CACHE.clear()
    with pytest.raises(RuntimeError, match="adaln_table_sha256"):
        affine.find_curve_affine(str(stage), table)


def test_missing_affine_fails_closed_with_actionable_message(tmp_path, monkeypatch):
    stage = _stage(tmp_path)
    monkeypatch.setattr(affine, "_candidate_affine_checkpoints", lambda base=None: [])
    affine._AFFINE_CACHE.clear()
    with pytest.raises(RuntimeError, match="51 AdaLN updates"):
        affine.find_curve_affine(str(stage), torch.randn(9, 3))


def test_project_curve_terms_rejects_malformed_low_rank_pair():
    mapping = affine.CurveAffine(
        basis=torch.randn(3, 11), mean=torch.randn(11), source="test",
        identity=("test",), table_hash="test")
    with pytest.raises(RuntimeError, match="incompatible"):
        affine.project_curve_terms(
            {"blocks.0.adaln_proj.linear": [
                (torch.randn(2, 11), torch.randn(7, 3), 1.0)]},
            mapping,
        )


def test_project_curve_terms_rejects_dense_width_mismatch():
    mapping = affine.CurveAffine(
        basis=torch.randn(3, 11), mean=torch.randn(11), source="test",
        identity=("test",), table_hash="test")
    with pytest.raises(RuntimeError, match="expects dense AdaLN width 12"):
        affine.project_curve_terms(
            {"blocks.0.adaln_proj.linear": [
                (torch.randn(2, 12), torch.randn(7, 2), 1.0)]},
            mapping,
        )


def test_project_curve_terms_preserves_multiple_adapter_terms():
    torch.manual_seed(431)
    mapping = affine.CurveAffine(
        basis=torch.randn(3, 11), mean=torch.randn(11), source="test",
        identity=("test",), table_hash="test")
    terms = {"blocks.0.adaln_proj.linear": [
        (torch.randn(2, 11), torch.randn(7, 2), 0.25),
        (torch.randn(4, 11), torch.randn(7, 4), 0.75),
    ]}
    projected, offsets = affine.project_curve_terms(terms, mapping)
    assert len(projected["blocks.0.adaln_proj.linear"]) == 2
    assert len(offsets["blocks.0.adaln_proj.linear"]) == 2
    assert [term[2] for term in projected["blocks.0.adaln_proj.linear"]] == [0.25, 0.75]
    assert [term[1] for term in offsets["blocks.0.adaln_proj.linear"]] == [0.25, 0.75]


def test_kj_cached_patcher_init_exposes_selected_checkpoint_path(tmp_path):
    selected = tmp_path / "selected-int8-convrot.safetensors"
    save_file({"dummy": torch.zeros(1)}, selected)

    class Patcher:
        cached_patcher_init = (object(), (str(selected), {"weight_dtype": "int8"}, None))

    assert os.path.realpath(affine._base_checkpoint_path(Patcher())) == os.path.realpath(selected)

    class Missing:
        cached_patcher_init = (object(), (str(tmp_path / "missing.safetensors"), {}, None))

    assert affine._base_checkpoint_path(Missing()) is None
