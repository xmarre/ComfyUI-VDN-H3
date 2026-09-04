"""Exact full-width AdaLN adapter support for Comfy H3 curve/pruned bases.

Comfy's curve H3 stores low-dimensional coordinates in ``adaln_t_table`` and removes
the dense TimeEmbedder. A released full-width AdaLN LoRA was trained on the dense
time embedding, so projecting its A/B factors into the curve basis is only an
approximation. This module instead reconstructs the matching dense embedding at the
exact timesteps used by current Comfy and adds the original low-rank delta.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import struct
from collections import OrderedDict
from contextvars import ContextVar
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from safetensors import safe_open

import comfy.ldm.common_dit
import comfy.ldm.minimax.model as mm
import comfy.model_management

_log = logging.getLogger("comfy.vdn")

DENSE_TIME_KEYS = (
    "time_embedder.proj_in.weight",
    "time_embedder.proj_in.bias",
    "time_embedder.proj_out.weight",
    "time_embedder.proj_out.bias",
)
MAX_CURVE_MATCH_RESIDUAL = 5e-4
_MAX_EMBEDDER_CACHE = 4
_EMBEDDER_CACHE = OrderedDict()


@dataclass(frozen=True)
class DenseFileIdentity:
    realpath: str
    mtime_ns: int
    size: int
    inode: int | None


def _identity(path):
    real = os.path.realpath(path)
    st = os.stat(real)
    return DenseFileIdentity(real, st.st_mtime_ns, st.st_size, getattr(st, "st_ino", None))


def _read_header(path):
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: truncated safetensors header")
        size = struct.unpack("<Q", raw)[0]
        if size <= 0 or size > (64 << 20):
            raise ValueError(f"{path}: invalid safetensors header size {size}")
        payload = fh.read(size)
        if len(payload) != size:
            raise ValueError(f"{path}: truncated safetensors JSON header")
        parsed = json.loads(payload)
        if not isinstance(parsed, dict):
            raise ValueError(f"{path}: safetensors header must contain a JSON object")
        return parsed


def _find_dense_prefix(header):
    for key in header:
        if not key.endswith(DENSE_TIME_KEYS[2]):
            continue
        prefix = key[: -len(DENSE_TIME_KEYS[2])]
        if all(prefix + suffix in header for suffix in DENSE_TIME_KEYS):
            return prefix
    return None


def _curve_hash(table):
    raw = table.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


@dataclass
class DenseTimeEmbedder:
    proj_in_weight: torch.Tensor
    proj_in_bias: torch.Tensor
    proj_out_weight: torch.Tensor
    proj_out_bias: torch.Tensor
    source: str

    def time_embedding(self, t):
        freq_dim = self.proj_in_weight.shape[1]
        half = freq_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t.to(torch.float32)[:, None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def silu_grid(self, t):
        emb = self.time_embedding(t)
        wi = self.proj_in_weight.to(device=t.device, dtype=torch.float32)
        bi = self.proj_in_bias.to(device=t.device, dtype=torch.float32)
        wo = self.proj_out_weight.to(device=t.device, dtype=torch.float32)
        bo = self.proj_out_bias.to(device=t.device, dtype=torch.float32)
        hidden = F.silu(F.linear(emb, wi, bi))
        return F.silu(F.linear(hidden, wo, bo))


def _load_dense_embedder(path, prefix):
    identity = _identity(path)
    with safe_open(identity.realpath, framework="pt", device="cpu") as handle:
        tensors = {
            suffix: handle.get_tensor(prefix + suffix).clone()
            for suffix in DENSE_TIME_KEYS
        }
    if _identity(path) != identity:
        raise RuntimeError(f"dense H3 checkpoint changed while reading: {path}")
    return DenseTimeEmbedder(
        tensors[DENSE_TIME_KEYS[0]],
        tensors[DENSE_TIME_KEYS[1]],
        tensors[DENSE_TIME_KEYS[2]],
        tensors[DENSE_TIME_KEYS[3]],
        identity.realpath,
    ), identity


def _candidate_files(stage_path):
    local = os.path.join(stage_path, "dense_time_embedder.safetensors")
    if os.path.isfile(local):
        # A stage-local companion is an explicit user assertion of base identity.
        # Do not silently fall through to an unrelated installed model if it is stale.
        return [local], True
    try:
        import folder_paths
        roots = folder_paths.get_folder_paths("diffusion_models")
    except Exception:
        roots = []
    found = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, files in os.walk(root):
            for filename in files:
                if filename.endswith(".safetensors"):
                    found.append(os.path.join(dirpath, filename))
    return sorted(found), False


def _fit_residual(table, embedder):
    if table.ndim != 2 or table.shape[0] < 2:
        raise ValueError("adaln_t_table must be a two-dimensional grid with at least two rows")
    t = torch.linspace(0.0, 1.0, table.shape[0], dtype=torch.float32)
    dense = embedder.silu_grid(t).cpu()
    curve = table.detach().to(device="cpu", dtype=torch.float32)
    # The curve coordinates should linearly reproduce the matching dense time grid.
    solution = torch.linalg.lstsq(curve, dense).solution
    reconstructed = curve @ solution
    denom = dense.norm().clamp_min(1e-12)
    return float((reconstructed - dense).norm() / denom)


def find_dense_time_embedder(stage_path, curve_table):
    """Resolve the exact matching dense time embedder for a curve base.

    The least-squares residual is only a base-match fingerprint. It is never used to
    transform LoRA factors or to approximate the adapter.
    """
    curve_key = _curve_hash(curve_table)
    candidates, authoritative = _candidate_files(stage_path)
    errors = []
    for path in candidates:
        try:
            identity = _identity(path)
            cache_key = (curve_key, identity)
            hit = _EMBEDDER_CACHE.get(cache_key)
            if hit is not None:
                _EMBEDDER_CACHE.move_to_end(cache_key)
                return hit
            header = _read_header(path)
            if any(key.endswith("adaln_t_table") for key in header):
                raise ValueError("candidate is itself a curve/pruned H3 checkpoint")
            prefix = _find_dense_prefix(header)
            if prefix is None:
                raise ValueError("candidate has no complete dense H3 time_embedder")
            embedder, checked_identity = _load_dense_embedder(path, prefix)
            if checked_identity != identity:
                raise RuntimeError("candidate changed while reading")
            residual = _fit_residual(curve_table, embedder)
            if residual > MAX_CURVE_MATCH_RESIDUAL:
                raise ValueError(
                    f"curve/dense fingerprint residual {residual:.3e} exceeds "
                    f"{MAX_CURVE_MATCH_RESIDUAL:.1e}")
            result = (embedder, residual)
            _EMBEDDER_CACHE[cache_key] = result
            _EMBEDDER_CACHE.move_to_end(cache_key)
            while len(_EMBEDDER_CACHE) > _MAX_EMBEDDER_CACHE:
                _EMBEDDER_CACHE.popitem(last=False)
            return result
        except Exception as exc:
            errors.append(f"{path}: {exc}")
            if authoritative:
                break
    suffix = "\n".join(errors[:8])
    raise RuntimeError(
        "VDN needs the matching dense MiniMax-H3 time embedder to apply full-width "
        "AdaLN adapters exactly on this curve/pruned base. Place a companion "
        "dense_time_embedder.safetensors in the selected VDN stage directory using "
        "tools/extract_h3_time_embedder.py, or install the matching dense H3 base."
        + ("\nCandidates rejected:\n" + suffix if suffix else ""))


class CurveAdalnState:
    """Execution-local dense AdaLN rows, safe for nested/reentrant model calls."""
    def __init__(self):
        self._value = ContextVar("vdn_curve_dense_adaln", default=None)

    def push(self, value):
        return self._value.set(value)

    def reset(self, token):
        self._value.reset(token)

    def get(self):
        return self._value.get()


def _mask_row_values(mask, latent_t, lat_h, lat_w):
    m = F.pad(mask, (0, lat_w - mask.shape[-1], 0, lat_h - mask.shape[-2]), mode="replicate")
    m = m.reshape(latent_t, lat_h // 2, 2, lat_w // 2, 2).amax(dim=(2, 4))
    values = m.reshape(-1)
    if bool((values >= 1.0 - 1e-3).all()):
        return None
    return values


def minimax_unique_timesteps(dm, x, timestep, context, transformer_options=None,
                             minimax_payload=None, denoise_mask=None,
                             audio_denoise_mask=None):
    """Mirror current Comfy MiniMax-H3's exact ``unique_t`` construction."""
    payload = minimax_payload or {}
    transformer_options = transformer_options or {}
    video_x, audio_x = x[0], x[1]
    padded = comfy.ldm.common_dit.pad_to_patch_size(video_x, dm.patch_size)
    latent_t, lat_h, lat_w = padded.shape[2], padded.shape[3], padded.shape[4]
    audio_t = audio_x.shape[-1]
    text_len = context.shape[1]
    layout = payload.get("layout")
    signature = (text_len, latent_t, lat_h, lat_w, audio_t)
    if layout is None or layout.signature != signature:
        layout = mm.PackedLayout(text_len, latent_t, lat_h, lat_w, audio_t,
                                 keyframes=payload.get("keyframes"),
                                 refs=payload.get("refs"))

    shift_v = float(transformer_options.get(
        "minimax_h3_sigma_shift_video", dm.sigma_shift_video))
    shift_a = float(transformer_options.get(
        "minimax_h3_sigma_shift_audio", dm.sigma_shift_audio))
    sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    t_v = float(1.0 - sigma_v)
    t_a = float(1.0 - mm.time_shift_sigma(sigma_v, shift_v, shift_a))

    vis_aug = float(payload.get("visual_cond_noise_aug", mm.VISUAL_COND_TIMESTEP))
    aud_aug = float(payload.get("audio_cond_noise_aug", mm.AUDIO_COND_TIMESTEP))
    seg_t = {
        "text": t_v, "video": t_v, "audio": t_a,
        "cond": max(t_v, vis_aug), "ref_img": max(t_v, vis_aug),
        "cond_audio": max(t_a, aud_aug), "ref_audio": max(t_a, aud_aug),
    }

    video_rows_t = None
    audio_rows_t = None
    if denoise_mask is not None:
        m = _mask_row_values(denoise_mask[0, 0].to(torch.float32),
                             latent_t, lat_h, lat_w)
        if m is not None:
            rows_t = (1.0 - m * sigma_v.to(m.device)).clamp(
                max=max(t_v, mm.VISUAL_COND_TIMESTEP))
            if rows_t.unique().numel() == 1:
                seg_t["video"] = float(rows_t[0])
            else:
                video_rows_t = rows_t
    if audio_denoise_mask is not None:
        m = audio_denoise_mask[0, 0].to(torch.float32).reshape(-1)
        if not bool((m >= 1.0 - 1e-3).all()):
            sigma_a = 1.0 - t_a
            rows_t = (1.0 - m * sigma_a).clamp(
                max=max(t_a, mm.AUDIO_COND_TIMESTEP))
            if rows_t.unique().numel() == 1:
                seg_t["audio"] = float(rows_t[0])
            else:
                audio_rows_t = rows_t

    values = {t_v, t_a} | {seg_t[k] for _, _, k in layout.segments}
    if video_rows_t is not None:
        values |= set(video_rows_t.unique().tolist())
    if audio_rows_t is not None:
        values |= set(audio_rows_t.unique().tolist())
    return sorted(values)


def make_dense_curve_wrapper(dm, embedder: DenseTimeEmbedder, state: CurveAdalnState):
    """Publish exact dense AdaLN inputs for the duration of one model execution."""
    def wrap(executor, *args, **kwargs):
        x = args[0] if args else kwargs["x"]
        timestep = args[1] if len(args) > 1 else kwargs["timestep"]
        context = args[2] if len(args) > 2 else kwargs["context"]
        transformer_options = (args[3] if len(args) > 3
                               else kwargs.get("transformer_options", {}))
        unique_t = minimax_unique_timesteps(
            dm, x, timestep, context, transformer_options,
            kwargs.get("minimax_payload"), kwargs.get("denoise_mask"),
            kwargs.get("audio_denoise_mask"))
        dense = embedder.silu_grid(torch.tensor(unique_t, dtype=torch.float32))
        token = state.push(dense.to(context.device, context.dtype))
        try:
            return executor(*args, **kwargs)
        finally:
            state.reset(token)
    return wrap


def make_curve_adaln_forward(base, terms, state: CurveAdalnState,
                             managed_module: str | None = None):
    """Patch one curve ``AdalnProj.forward`` without a mutable hook chain.

    ``terms`` is either the historical direct list of ``(A, B, scale)`` tensors used
    by focused unit tests, or a managed runtime-term model when ``managed_module`` is
    supplied. Production uses the managed form so the large AdaLN B factors follow
    Comfy load/offload ownership instead of being copied from a closure every block.
    """
    if getattr(base, "apply_silu", True):
        raise RuntimeError("curve AdaLN patch received a full-width AdalnProj")

    def forward(t_emb):
        x = base.linear(t_emb)
        dense = state.get()
        if dense is None:
            raise RuntimeError("VDN curve AdaLN state is unavailable outside model forward")
        if dense.shape[0] != x.shape[0]:
            raise RuntimeError(
                f"VDN curve AdaLN row mismatch: dense={dense.shape[0]}, curve={x.shape[0]}")

        if managed_module is None:
            active_terms = tuple(
                (a.to(device=x.device, dtype=x.dtype),
                 b.to(device=x.device, dtype=x.dtype), scale)
                for a, b, scale in terms
            )
        else:
            active_terms = terms.terms_on(managed_module, x.device, x.dtype)

        dense_cast = dense.to(x.dtype)
        for av, bv, scale in active_terms:
            delta = F.linear(F.linear(dense_cast, av), bv)
            x = x + delta * scale
        x = x.view(x.shape[0] * base.modalities, base.expand * base.hidden)
        return x.chunk(base.expand, dim=-1)

    forward._vdn_curve_adaln = True
    return forward
