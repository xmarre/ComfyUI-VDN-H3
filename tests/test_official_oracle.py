"""Direct numerical oracle against the checked-out OpenVDN source tree.

Unlike ``test_vdn_math.py`` this module executes the official implementation files.
The CI pins OpenVDN by commit and exposes it through ``OPENVDN_ROOT``.  Small CPU
shapes keep the comparison cheap while preserving the real operations and parameters.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import types

import pytest
import torch

from vdn_h3 import branch as port_branch
from vdn_h3 import window as port_window

ROOT = Path(os.environ.get("OPENVDN_ROOT", ""))
pytestmark = pytest.mark.skipif(
    not ROOT.is_dir(), reason="OPENVDN_ROOT checkout is required for official oracle")


def _pkg(name):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__path__ = []
        sys.modules[name] = mod
    return sys.modules[name]


def _load(name, relative):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _official_modules():
    for name in (
        "src", "src.models", "src.models.ops", "src.models.linear_attention",
        "src.models.softmax_attention", "src.checkpoints",
    ):
        _pkg(name)

    sequence = _load("src.models.sequence_layout", "src/models/sequence_layout.py")
    window = _load("official_vdn_window", "src/models/softmax_attention/window.py")
    delta = _load("src.models.linear_attention.delta_rule",
                  "src/models/linear_attention/delta_rule.py")
    scan = _load("src.models.linear_attention.scan", "src/models/linear_attention/scan.py")

    # features.py imports the CUDA inference helper even though its CPU/eager path does
    # not call it.  Stub only that unavailable optional kernel module; the feature math
    # being compared still executes directly from official features.py.
    temporal = types.ModuleType("src.models.ops.temporal_conv")
    temporal.temporal_conv_activate = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("CUDA temporal kernel must not run in CPU oracle"))
    sys.modules[temporal.__name__] = temporal
    features = _load("src.models.linear_attention.features",
                     "src/models/linear_attention/features.py")

    rms = _load("src.models.ops.rms_norm", "src/models/ops/rms_norm.py")
    gates = _load("src.models.attention_gates", "src/models/attention_gates.py")
    kernels = _load("src.models.linear_attention.kernels",
                    "src/models/linear_attention/kernels.py")
    layers = _load("src.models.linear_attention.layers",
                   "src/models/linear_attention/layers.py")

    key_mapping = types.ModuleType("src.checkpoints.key_mapping")
    key_mapping.SHORT_CONV_TARGETS = ("q", "k", "v")
    sys.modules[key_mapping.__name__] = key_mapping

    linear_pkg = sys.modules["src.models.linear_attention"]
    linear_pkg.DELTA_BACKENDS = delta.DELTA_BACKENDS
    linear_pkg.BidirectionalLinearBranch = None
    branch = _load("official_vdn_branch", "src/models/linear_attention/branch.py")
    return types.SimpleNamespace(
        sequence=sequence, window=window, delta=delta, scan=scan,
        features=features, rms=rms, gates=gates, kernels=kernels,
        layers=layers, branch=branch)


@pytest.fixture(scope="module")
def official():
    return _official_modules()


def test_window_bounds_frame_and_chunk_modes(official):
    for frames in (1, 2, 7, 23):
        for radius in (0, 1, 3):
            for chunk in (0, 1, 5, 8):
                got = port_window.window_bounds(frames, radius, chunk)
                want = official.window.window_bounds(frames, radius, chunk)
                assert got == want
                assert port_window.full_coverage(got, frames) == all(
                    lo <= 0 and hi >= frames - 1 for lo, hi in want)


def test_window_softmax_all_anchor_modes_against_official(official):
    torch.manual_seed(100)
    frames, per_frame, heads, dim = 9, 3, 2, 8
    video_start = 4
    total = video_start + frames * per_frame + 5
    layout = official.sequence.SequenceLayout(
        seq_len=total, video_start=video_start, num_frames=frames,
        tokens_per_frame=per_frame)
    q, k, v = [torch.randn(total, heads, dim) for _ in range(3)]
    bounds = official.window.window_bounds(frames, 1, 3)
    scale = dim ** -0.5
    for anchor in ("none", "rows", "columns", "both"):
        want = official.window.window_softmax_reference(
            q, k, v, layout, bounds, scale, anchor_frames=anchor)
        got = port_window.window_softmax_grouped(
            q, k, v, layout.video_start, layout.video_end,
            frames, per_frame, bounds, scale, anchor_frames=anchor)
        assert torch.allclose(got, want, atol=2e-6, rtol=2e-6), anchor


def _spd_stats(frames=5, heads=2, dim=8):
    torch.manual_seed(101)
    alpha = torch.rand(frames, heads, dim) * 0.4 + 0.55
    x = torch.randn(frames, heads, dim, dim) * 0.02
    a = x @ x.transpose(-1, -2)
    b = torch.randn(frames, heads, dim, dim) * 0.1
    return alpha, a, b


@pytest.mark.parametrize("rule", ["vdn_solve", "sana_scaled", "vdn_scaled"])
def test_delta_rules_direct_official(rule, official):
    alpha, a, b = _spd_stats()
    tokens = 17
    port_cls = port_branch.DELTA_BACKENDS[rule]
    off_cls = official.delta.DELTA_BACKENDS[rule]
    got_t, got_i = port_cls(tokens).factor_apply(alpha, a, b)
    want_t, want_i = off_cls(tokens).factor_apply(alpha, a, b)
    assert torch.allclose(got_t, want_t, atol=2e-6, rtol=2e-6)
    assert torch.allclose(got_i, want_i, atol=2e-6, rtol=2e-6)


def test_frame_statistics_direct_official(official):
    torch.manual_seed(102)
    frames, heads, rows, dim = 4, 3, 7, 8
    k = torch.randn(frames, heads, rows, dim)
    v = torch.randn(frames, heads, rows, dim)
    beta = torch.sigmoid(torch.randn(frames, heads, rows))
    got_a, got_b = port_branch.frame_statistics(k, v, beta, a_fp32=True)
    want_a, want_b = official.scan.frame_statistics(k, v, beta, a_fp32=True, inference=False)
    assert torch.allclose(got_a, want_a, atol=1e-6, rtol=1e-6)
    assert torch.allclose(got_b, want_b, atol=1e-6, rtol=1e-6)


def test_forward_reverse_scans_and_alpha_bridge_direct_official(official):
    alpha, a, b = _spd_stats(frames=7)
    text = torch.randn(2, 8, 8) * 0.03
    port_backend = port_branch.VdnDelta(11)
    official_backend = official.delta.VdnDelta(11)
    got_prefix, got_suffix = port_branch.run_scans(
        port_backend, alpha, a, b, text_state=text)
    want_prefix, want_suffix = official.scan._run_scans_inference(
        official_backend, alpha, a, b, text_state=text)
    assert torch.allclose(got_prefix, want_prefix, atol=2e-6, rtol=2e-6)
    assert torch.allclose(got_suffix, want_suffix, atol=2e-6, rtol=2e-6)

    bounds = official.window.window_bounds(7, 1, 3)
    got = port_branch.gather_linear_state(
        got_prefix, got_suffix, alpha, bounds, bridge="alpha", text_state=text)
    want = official.scan.gather_linear_state(
        want_prefix, want_suffix, alpha, bounds, bridge="alpha", text_state=text,
        inference=False)
    assert torch.allclose(got, want, atol=3e-6, rtol=3e-6)


def test_activation_and_short_conv_feature_path_direct_official(official):
    torch.manual_seed(103)
    frames, gh, gw, heads, dim = 3, 2, 3, 2, 4
    rows = frames * gh * gw
    tokens = torch.randn(rows, heads, dim)
    channels = heads * dim

    for l2norm in (False, True):
        got = port_branch._activate(tokens, l2norm)
        want = official.features._activate_body(tokens, l2norm)
        assert torch.allclose(got, want, atol=1e-7, rtol=1e-7)

    module = official.features.LinearAttentionSepConv(channels, ("k", "v"))
    sp = torch.randn(channels, 1, 5, 5) * 0.05
    tm = torch.randn(channels, 1, 5) * 0.05
    with torch.no_grad():
        module.k_sp.weight.copy_(sp)
        module.k_tm.weight.copy_(tm)
    want = official.features.prepare_linear_features(
        tokens, True, conv=module, proj="k", num_frames=frames,
        frame_size=(gh, gw))
    got = port_branch.conv_features(
        tokens, sp, tm, frames, (gh, gw), l2norm=True)
    assert torch.allclose(got, want, atol=2e-6, rtol=2e-6)


def _branch_weights(module):
    return {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}


@pytest.mark.parametrize("short_conv", [(), ("k", "v")])
def test_complete_bidirectional_linear_branch_direct_official(short_conv, official):
    torch.manual_seed(104 if not short_conv else 105)
    hidden, heads, dim = 16, 2, 4
    frames, gh, gw = 5, 2, 2
    per_frame = gh * gw
    rows = frames * per_frame
    text_len = 6

    off = official.branch.BidirectionalLinearBranch(
        hidden, heads, dim, delta_rule="vdn_solve", bridge="alpha",
        a_fp32=True, short_conv=short_conv)
    weights = _branch_weights(off)
    port = port_branch.LinearBranch(
        weights, heads, dim, delta_rule="vdn_solve", bridge="alpha",
        a_fp32=True, short_conv=short_conv, enable_text_state=True)

    xv = torch.randn(rows, hidden)
    qkv = tuple(torch.randn(rows, heads, dim) for _ in range(3))
    text_x = torch.randn(text_len, hidden)
    text_qkv = tuple(torch.randn(text_len, heads, dim) for _ in range(3))
    bounds = official.window.window_bounds(frames, 1, 2)

    want = off(
        xv, frames, per_frame, bounds, qkv,
        frame_size=(gh, gw) if short_conv else None,
        skip_ends=True, text_x=text_x, text_qkv_raw=text_qkv, inference=False)
    got = port.readout(
        weights, xv, *qkv, frames, per_frame, bounds,
        frame_size=(gh, gw) if short_conv else None,
        text_x=text_x, text_k_raw=text_qkv[1], text_v_raw=text_qkv[2],
        skip_ends=True)
    assert torch.allclose(got, want, atol=2e-5, rtol=2e-5)


def test_released_stage_dmd_configuration_is_eight_step(official):
    # Directly inspect the official repository artifact rather than duplicating this
    # release fact in a local fixture.
    import yaml
    cfg = yaml.safe_load((ROOT / "configs/training/stage_dmd_vdn.yaml").read_text())
    assert cfg["turbo"]["num_steps"] == 8
    assert cfg["turbo"]["video_shift"] == 12.0
    assert cfg["turbo"]["audio_shift"] == 3.0
