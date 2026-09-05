"""CPU-only regression for VDN local-window attention isolation.

VDN's grouped local windows are part of the trained hybrid operator. The known-good
v1.3.1 path and upstream v1.4.x deliberately keep those windows on exact SDPA even
when the surrounding MiniMax model installs an ``optimized_attention_override``
(such as Sage/Kitchen quantized attention). The model-level override remains valid for
base/dense attention paths, but it must not enter the VDN local windows.

This test installs a recording optimized_attention stub and proves that supplying
transformer_options to the grouped helper neither dispatches through the stub nor
changes the result.
"""
import sys
import types

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

calls = []


def _stub_optimized_attention(q, k, v, heads, mask=None, **kwargs):
    calls.append({
        "heads": heads,
        "transformer_options": kwargs.get("transformer_options"),
        "q_shape": tuple(q.shape),
        "k_shape": tuple(k.shape),
    })
    raise AssertionError("VDN local window routed through optimized_attention")


def _install_stub():
    """Shadow comfy.ldm.modules.attention with the recording stub, then restore it."""
    for name in ("comfy", "comfy.ldm", "comfy.ldm.modules"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    stub = types.ModuleType("comfy.ldm.modules.attention")
    stub.optimized_attention = _stub_optimized_attention
    prev_mod = sys.modules.get("comfy.ldm.modules.attention")
    parent = sys.modules["comfy.ldm.modules"]
    prev_attr = getattr(parent, "attention", None)
    sys.modules["comfy.ldm.modules.attention"] = stub
    parent.attention = stub

    def restore():
        if prev_mod is None:
            sys.modules.pop("comfy.ldm.modules.attention", None)
        else:
            sys.modules["comfy.ldm.modules.attention"] = prev_mod
        if prev_attr is None:
            parent.__dict__.pop("attention", None)
        else:
            parent.attention = prev_attr

    return restore


def test_dispatch():
    from vdn_h3.window import window_bounds, window_softmax_grouped

    restore = _install_stub()
    try:
        calls.clear()

        torch.manual_seed(0)
        video_start, tokens, frames, heads, dim = 5, 8, 12, 2, 16
        seq = video_start + frames * tokens + 3
        q = torch.randn(seq, heads, dim)
        k = torch.randn(seq, heads, dim)
        v = torch.randn(seq, heads, dim)
        transformer_options = {
            "optimized_attention_override": _stub_optimized_attention,
            "anything": True,
        }

        got = window_softmax_grouped(
            q,
            k,
            v,
            video_start,
            seq - 3,
            frames,
            tokens,
            window_bounds(frames, 1, 5),
            dim ** -0.5,
            anchor_frames="both",
            transformer_options=transformer_options,
        )
        assert calls == [], "VDN local windows must not consume optimized_attention overrides"

        want = window_softmax_grouped(
            q,
            k,
            v,
            video_start,
            seq - 3,
            frames,
            tokens,
            window_bounds(frames, 1, 5),
            dim ** -0.5,
            anchor_frames="both",
            transformer_options=None,
        )
        assert torch.equal(got, want), "transformer_options changed exact VDN window math"
    finally:
        restore()


if __name__ == "__main__":
    test_dispatch()
    print("ALL PASS")
