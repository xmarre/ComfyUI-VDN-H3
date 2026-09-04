"""Execution-owned retained helpers layered on the released VDN branch math.

This module intentionally subclasses :class:`vdn_h3.branch.LinearBranch` instead of
turning the reference-math module into a cache owner.  The arithmetic remains the
same; only prefix/suffix bank storage is borrowed from the execution-local
``RuntimeBuffers`` lease when retention is enabled.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from vdn_h3 import branch as B
from vdn_h3.runtime import current_runtime_buffers


def run_scans_runtime(backend, alpha, a_raw, b_raw, text_state=None):
    """Reference recurrence with execution-local bank reuse when available."""
    with torch.autocast(device_type=a_raw.device.type, enabled=False):
        transitions, injections = backend.factor_apply(alpha, a_raw, b_raw)
        num_frames = transitions.shape[0]
        start = (
            torch.zeros_like(injections[0])
            if text_state is None else text_state.to(injections.dtype)
        )
        resources = current_runtime_buffers()
        if resources is None:
            prefix = torch.empty(
                (num_frames, *start.shape), dtype=injections.dtype,
                device=injections.device)
            suffix = torch.empty_like(prefix)
        else:
            prefix, suffix = resources.scan_banks(
                num_frames, start.shape, injections.dtype, injections.device)

        state = start
        for frame in range(num_frames):
            torch.baddbmm(
                injections[frame], state, transitions[frame], out=prefix[frame])
            state = prefix[frame]
        state = start
        for frame in range(num_frames - 1, -1, -1):
            torch.baddbmm(
                injections[frame], state, transitions[frame], out=suffix[frame])
            state = suffix[frame]
        return prefix, suffix


class RuntimeLinearBranch(B.LinearBranch):
    """LinearBranch whose reusable state banks belong to the current VDN execution."""

    def _readout(self, w, xv, qkv_raw, num_frames, tokens_per_frame, bounds,
                 frame_size, text_x, text_k_raw, text_v_raw):
        n_heads, head_dim = self.num_heads, self.head_dim
        backend = self._delta_backend(tokens_per_frame)
        shape = (num_frames, tokens_per_frame, n_heads, head_dim)

        query, key, value = self._features(
            w, *qkv_raw, num_frames, frame_size,
            q_fhsd=self.fuse_epilogue)
        key_by_frame = key.view(shape).permute(0, 2, 1, 3)
        value_by_frame = value.view(shape).permute(0, 2, 1, 3)
        beta = torch.sigmoid(F.linear(xv, w["beta_proj.weight"]))
        beta = beta.view(
            num_frames, tokens_per_frame, n_heads).permute(0, 2, 1)

        a, b = B.frame_statistics(
            key_by_frame, value_by_frame, beta, a_fp32=self.a_fp32)
        frame_mean = xv.view(num_frames, tokens_per_frame, -1).mean(
            dim=1, dtype=torch.float32)
        alpha = B.alpha_gate(
            frame_mean,
            w["alpha.down.weight"],
            w["alpha.up.weight"],
            w["alpha.dt_bias"],
            w["alpha.A_log"],
            n_heads,
            head_dim,
        )

        text_state = self._text_state(w, text_x, text_k_raw, text_v_raw)
        prefix_states, suffix_states = run_scans_runtime(
            backend, alpha, a, b, text_state=text_state)
        gate = torch.sigmoid(
            F.linear(xv, w["output_gate.down.weight"])
            @ w["output_gate.up.weight"].T
            + w["output_gate.up.bias"]
        )
        linear_state = B.gather_linear_state(
            prefix_states,
            suffix_states,
            alpha,
            bounds,
            bridge=self.bridge,
            text_state=text_state,
            out_dtype=gate.dtype,
            fuse=self.fuse_epilogue,
        )

        if query.dim() == 4:
            query_fhsd = query
        else:
            query_fhsd = query.view(shape).permute(0, 2, 1, 3)
        readout = torch.matmul(
            query_fhsd, linear_state.transpose(-1, -2))
        return B.linear_epilogue(
            readout,
            w["norm.weight"],
            gate,
            w["norm.weight"].new_tensor(1e-6).item(),
            fuse=self.fuse_epilogue,
        )
