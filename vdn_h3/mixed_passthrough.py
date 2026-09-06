"""Safe VDN behavior for Flow's heterogeneous mixed-grid transformer calls.

VDN's released hybrid-attention branch is trained on one uniform video-token grid.
Flow's Continuum mixed-grid handoff deliberately presents a heterogeneous sequence:
a target-grid protected prefix followed by a lower-grid generated suffix. PR #2 added
an experimental ``dense_gate_no_linear`` compatibility mode for this topology, but
that mode was never GPU-quality validated and still applied VDN's learned softmax
gate to the heterogeneous rows.

For API-2 mixed-grid calls, keep the VDN adapter-modified QKV/out projections but use
the native MiniMax-H3 attention operator. This preserves the exact mixed-grid RoPE
and reference/conditioning semantics while declining to run untrained VDN hybrid
geometry on an unsupported topology. Normal uniform-grid VDN calls and API-1 sparse
compatibility are unchanged.
"""
from __future__ import annotations

from vdn_h3 import hybrid as H

_ORIGINAL_MAKE_VDN_FORWARD = H.make_vdn_forward
_INSTALLED = False


def make_vdn_forward_mixed_passthrough(attn, state, block_index):
    """Wrap the normal VDN forward with an API-2 mixed-grid native-attention path."""
    ordinary_vdn = _ORIGINAL_MAKE_VDN_FORWARD(attn, state, block_index)

    def vdn_forward(x, rope_freqs=None, transformer_options=None):
        options = transformer_options or {}
        contract = options.get(H.VDN_EXTERNAL_SEQUENCE_KEY)
        if isinstance(contract, dict) and contract.get("api") == 2:
            layout = state.layout
            if layout is None:
                raise RuntimeError("VDN mixed-grid call arrived without an active native layout")

            # Reuse the strict API-2 geometry/RoPE validator. A malformed or stale
            # contract must fail closed rather than silently falling back.
            if not H._external_reduced_sequence_active(
                options, layout, int(x.shape[0]), rope_freqs
            ):
                raise RuntimeError("VDN API-2 mixed-grid contract did not activate")

            H._once(
                ("mixed-grid-native-passthrough", layout.seq_len, int(x.shape[0])),
                f"external mixed-grid sequence {int(x.shape[0])}/{layout.seq_len}: "
                "using native MiniMax-H3 attention; VDN hybrid window/linear/gate "
                "geometry is disabled for this untrained heterogeneous topology",
            )

            # The qkv/out modules are still the model's live modules, so VDN's
            # adapter patches/hooks remain effective. Only the VDN hybrid attention
            # architecture is bypassed for this one heterogeneous transformer call.
            base_options = dict(options)
            base_options.pop(H.VDN_EXTERNAL_SEQUENCE_KEY, None)
            return H._base_attention(attn, x, rope_freqs, base_options)

        return ordinary_vdn(x, rope_freqs=rope_freqs, transformer_options=options)

    vdn_forward._vdn_forward = True
    vdn_forward._vdn_external_sequence_api = H.VDN_EXTERNAL_SEQUENCE_API_VERSION
    vdn_forward._vdn_mixed_grid_native_passthrough = True
    return vdn_forward


def install():
    """Install once before Apply-VDN creates attention object patches."""
    global _INSTALLED
    if _INSTALLED:
        return
    H.make_vdn_forward = make_vdn_forward_mixed_passthrough
    _INSTALLED = True
