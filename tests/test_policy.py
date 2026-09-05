from __future__ import annotations

import os

from vdn_h3 import policy
from vdn_h3 import spec


def _files(tmp_path, plain_size=100, quant_size=40):
    stage = tmp_path / "stage"
    branch = stage / "linear_branch"
    branch.mkdir(parents=True)
    plain = branch / spec.BRANCH_FILE
    quant = branch / spec.BRANCH_FILE_INT8
    with open(plain, "wb") as fh:
        fh.truncate(plain_size)
    with open(quant, "wb") as fh:
        fh.truncate(quant_size)
    return stage, plain, quant


def test_select_branch_file_prefers_plain_unless_int8_requested(tmp_path):
    stage, plain, quant = _files(tmp_path)
    assert policy.select_branch_file(str(stage), prefer_int8=False) == str(plain)
    assert policy.select_branch_file(str(stage), prefer_int8=True) == str(quant)


def test_auto_branch_policy_uses_resident_plain_with_headroom(tmp_path, monkeypatch):
    stage, plain, _ = _files(tmp_path)
    gib = 1 << 30
    monkeypatch.setattr(os.path, "getsize", lambda path: gib if path == str(plain) else gib // 2)
    mode, prefer_int8 = policy.auto_branch_policy(str(stage), 8 * gib)
    assert mode == "resident"
    assert prefer_int8 is False


def test_auto_branch_policy_streams_int8_under_pressure(tmp_path, monkeypatch):
    stage, plain, quant = _files(tmp_path)
    gib = 1 << 30
    sizes = {str(plain): 4 * gib, str(quant): 2 * gib}
    monkeypatch.setattr(os.path, "getsize", lambda path: sizes[path])
    mode, prefer_int8 = policy.auto_branch_policy(str(stage), 5 * gib)
    assert mode == "stream"
    assert prefer_int8 is True


def test_auto_retain_policy_tracks_selected_branch_size(tmp_path, monkeypatch):
    stage, plain, quant = _files(tmp_path)
    gib = 1 << 30
    sizes = {str(plain): 4 * gib, str(quant): 2 * gib}
    monkeypatch.setattr(os.path, "getsize", lambda path: sizes[path])
    assert policy.auto_retain_policy(str(stage), False, 14 * gib)
    assert not policy.auto_retain_policy(str(stage), False, 13 * gib)
    assert policy.auto_retain_policy(str(stage), True, 12 * gib)
    assert not policy.auto_retain_policy(str(stage), True, 11 * gib)
