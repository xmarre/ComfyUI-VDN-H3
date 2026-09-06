from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from vdn_h3 import hybrid
from vdn_h3.mixed_passthrough import install


class _TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 2
        self.head_dim = 2
        self.qkv_proj = nn.Linear(4, 12, bias=False)
        self.q_norm = nn.RMSNorm(2, eps=1e-6)
        self.k_norm = nn.RMSNorm(2, eps=1e-6)
        self.out_proj = nn.Identity()


def _state():
    cfg = {
        "radius": 1,
        "chunk": 1,
        "anchor_frames": "both",
        "enable_softmax_gate": True,
        "linear_enabled": True,
    }
    branch = SimpleNamespace(w={}, enable_text_state=False)
    state = hybrid.VDNState("mixed-grid-native-test", cfg, [branch], 2, 2)
    layout = hybrid.VDNLayout(
        video_start=2,
        video_end=8,
        num_frames=2,
        tokens_per_frame=3,
        frame_size=(1, 3),
        text_start=0,
        text_len=2,
        seq_len=8,
        radius=1,
        chunk=1,
        anchor_frames="both",
    )
    return state, layout


def _contract():
    return {
        hybrid.VDN_EXTERNAL_SEQUENCE_KEY: {
            "api": 2,
            "mode": hybrid.VDN_EXTERNAL_SEQUENCE_MODE,
            "topology": "mixed_grid_low_suffix",
            "native_sequence_rows": 8,
            "sequence_rows": 11,
            "video_start": 2,
            "temporal": 2,
            "prefix_t": 1,
            "source_rows_per_frame": 3,
            "prefix_rows_per_frame": 6,
        }
    }


def test_api2_mixed_grid_uses_native_attention_and_skips_vdn_gate(monkeypatch):
    install()
    torch.manual_seed(731)
    attn = _TinyAttention().requires_grad_(False)
    state, layout = _state()
    x = torch.randn(11, 4)
    rope = torch.eye(2).reshape(1, 1, 1, 1, 2, 2).expand(1, 11, 1, 1, 2, 2).contiguous()
    options = _contract()

    # Any attempt to execute the old dense_gate_no_linear compatibility path would
    # request VDN branch weights for softmax_gate. API-2 passthrough must not.
    monkeypatch.setattr(
        state,
        "weights_on",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("mixed-grid passthrough must not execute VDN gate/branch weights")
        ),
    )

    token = state._layout.set(layout)
    try:
        with torch.no_grad():
            got = hybrid.make_vdn_forward(attn, state, 0)(
                x, rope_freqs=rope, transformer_options=options
            )
            base_options = dict(options)
            base_options.pop(hybrid.VDN_EXTERNAL_SEQUENCE_KEY)
            want = hybrid._base_attention(attn, x, rope, base_options)
    finally:
        state._layout.reset(token)

    assert torch.allclose(got, want, atol=1e-6, rtol=1e-6)


def test_uniform_grid_keeps_normal_vdn_forward():
    install()
    attn = _TinyAttention()
    state, _ = _state()
    forward = hybrid.make_vdn_forward(attn, state, 0)
    assert forward._vdn_forward is True
    assert forward._vdn_external_sequence_api == 2
    assert forward._vdn_mixed_grid_native_passthrough is True
