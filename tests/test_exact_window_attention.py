from __future__ import annotations

import torch

from vdn_h3 import window


def test_vdn_window_sdpa_ignores_model_attention_override(monkeypatch):
    """Local VDN windows must stay on the exact v1.3.1/OpenVDN SDPA path.

    KJ's MiniMax loader can install a Sage ``optimized_attention_override`` for the
    base model. Passing transformer_options into VDN's grouped windows used to route
    those local windows through that approximate override. The released VDN hybrid
    was validated with exact local attention, so this must remain isolated from model
    dense-attention overrides.
    """
    from comfy.ldm.modules import attention as comfy_attention

    def forbidden_override(*_args, **_kwargs):
        raise AssertionError("VDN local window reached optimized_attention override")

    monkeypatch.setattr(comfy_attention, "optimized_attention", forbidden_override)

    torch.manual_seed(23)
    q = torch.randn(5, 2, 4)
    k = torch.randn(7, 2, 4)
    v = torch.randn(7, 2, 4)
    options = {"optimized_attention_override": forbidden_override}

    got = window._sdpa(q, k, v, 4 ** -0.5, options)
    want = window._sdpa(q, k, v, 4 ** -0.5, None)

    assert torch.equal(got, want)


def test_grouped_window_is_override_invariant(monkeypatch):
    """The full grouped-window helper must preserve the same isolation contract."""
    from comfy.ldm.modules import attention as comfy_attention

    def forbidden_override(*_args, **_kwargs):
        raise AssertionError("grouped VDN window reached optimized_attention override")

    monkeypatch.setattr(comfy_attention, "optimized_attention", forbidden_override)

    torch.manual_seed(29)
    frames = 4
    tokens_per_frame = 2
    video_start = 2
    video_end = video_start + frames * tokens_per_frame
    seq = video_end + 1
    q = torch.randn(seq, 2, 4)
    k = torch.randn(seq, 2, 4)
    v = torch.randn(seq, 2, 4)
    bounds = window.window_bounds(frames, radius=1, chunk=2)
    options = {"optimized_attention_override": forbidden_override}

    got = window.window_softmax_grouped(
        q,
        k,
        v,
        video_start,
        video_end,
        frames,
        tokens_per_frame,
        bounds,
        4 ** -0.5,
        anchor_frames="both",
        transformer_options=options,
    )
    want = window.window_softmax_grouped(
        q,
        k,
        v,
        video_start,
        video_end,
        frames,
        tokens_per_frame,
        bounds,
        4 ** -0.5,
        anchor_frames="both",
        transformer_options=None,
    )

    assert torch.equal(got, want)
