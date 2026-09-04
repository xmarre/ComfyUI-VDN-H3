"""Exact dense-AdaLN adapter support for ComfyUI MiniMax-H3 curve bases.

Curve/pruned H3 checkpoints replace the dense time embedder with ``adaln_t_table``
and make every AdaLN projection consume the table coordinates directly. A dense
Turbo LoRA therefore cannot be reshaped or projected into the small curve basis
without approximation.

This module takes the exact route instead: recover the matching *dense* H3 time
embedder from either a checkpoint-local companion file or an installed non-curve H3
checkpoint, evaluate it at the exact timesteps used by current ComfyUI, and add the
original low-rank AdaLN delta at runtime. The base curve projection itself remains
untouched.

The least-squares curve residual used below is only a compatibility fingerprint for
choosing the matching dense time embedder. It is never used to project adapter
weights and therefore never changes the adapter mathematics.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import struct
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from safetensors import safe_open

_log = logging.getLogger("comfy.vdn")

TIME_EMBEDDER_FILENAME = "dense_time_embedder.safetensors"
TIME_KEYS = (
    "time_embedder.proj_in.weight",
    "time_embedder.proj_in.bias",
    "time_embedder.proj_out.weight",
    "time_embedder.proj_out.bias",
)
DEFAULT_FREQ_DIM = 256
# A matching H3 dense/curve pair is a low-rank *base-model* approximation, so its
# residual is non-zero. Keep this deliberately strict: wrong-build grids observed in
# the wild are an order of magnitude worse. The chosen value is a compatibility
# guard, not an error tolerance for VDN math.
MAX_CURVE_MATCH_RESIDUAL = 5.0e-4


class CurveAdalnState:
    """Per-execution dense AdaLN state with correct nested/reentrant restoration."""

    def __init__(self):
        self._value: ContextVar[torch.Tensor | None] = ContextVar(
            "vdn_curve_dense_silu_temb", default=None)

    def push(self, value: torch.Tensor) -> Token:
        return self._value.set(value)

    def reset(self, token: Token) -> None:
        self._value.reset(token)

    def get(self) -> torch.Tensor | None:
        return self._value.get()


def _file_identity(path: str) -> tuple[str, int, int, int | None]:
    real = os.path.realpath(path)
    st = os.stat(real)
    return (real, st.st_mtime_ns, st.st_size, getattr(st, "st_ino", None))


def _tensor_hash(t: torch.Tensor) -> str:
    x = t.detach().to(torch.float32, device="cpu").contiguous()
    h = hashlib.sha256()
    h.update(str(tuple(x.shape)).encode())
    h.update(x.numpy().tobytes())
    return h.hexdigest()


def _safetensors_header(path: str) -> dict:
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
        header = json.loads(payload)
        if not isinstance(header, dict):
            raise ValueError(f"{path}: safetensors header is not an object")
        return header


def _find_dense_prefix(header: dict) -> str | None:
    if any(k.endswith("adaln_t_table") for k in header):
        return None
    suffixes = TIME_KEYS
    for key in header:
        if key.endswith(suffixes[2]):
            prefix = key[: -len(suffixes[2])]
            if all(prefix + suffix in header for suffix in suffixes):
                return prefix
    return None


def _owned_tensor(handle, key: str) -> torch.Tensor:
    # safe_open tensors may reference the file mapping. Always detach ownership
    # before leaving the context, including when the source tensor is already fp32.
    return handle.get_tensor(key).to(torch.float32).clone()


@dataclass(frozen=True)
class DenseTimeEmbedder:
    proj_in_weight: torch.Tensor
    proj_in_bias: torch.Tensor
    proj_out_weight: torch.Tensor
    proj_out_bias: torch.Tensor
    source: str
    identity: tuple | None = None
    freq_dim: int = DEFAULT_FREQ_DIM

    def silu_grid(self, t: torch.Tensor) -> torch.Tensor:
        """Return ``silu(TimeEmbedder(t))`` with current Comfy H3 arithmetic."""
        t = t.detach().to(torch.float32, device="cpu").reshape(-1)
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
        args = t[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        h = F.silu(F.linear(emb, self.proj_in_weight, self.proj_in_bias))
        return F.silu(F.linear(h, self.proj_out_weight, self.proj_out_bias))


def _load_embedder(path: str, prefix: str = "") -> DenseTimeEmbedder:
    identity = _file_identity(path)
    header = _safetensors_header(path)
    if not prefix:
        detected = _find_dense_prefix(header)
        if detected is not None:
            prefix = detected
    keys = [prefix + k for k in TIME_KEYS]
    missing = [k for k in keys if k not in header]
    if missing:
        raise ValueError(f"{path}: missing dense time-embedder tensors {missing}")
    with safe_open(path, framework="pt", device="cpu") as handle:
        tensors = [_owned_tensor(handle, key) for key in keys]
    return DenseTimeEmbedder(*tensors, source=path, identity=identity)


def _curve_fit_residual(table: torch.Tensor, dense_grid: torch.Tensor) -> float:
    """Compatibility fingerprint only; does not transform any adapter tensor."""
    table = table.detach().to(torch.float32, device="cpu")
    grid = dense_grid.detach().to(torch.float32, device="cpu")
    if table.shape[0] != grid.shape[0]:
        return float("inf")
    x = torch.cat([torch.ones(table.shape[0], 1), table], dim=1)
    solution = torch.linalg.lstsq(x, grid).solution
    residual = x @ solution - grid
    denom = grid.norm()
    return float(residual.norm() / denom) if float(denom) else float("inf")


def _candidate_dense_checkpoints() -> Iterable[str]:
    try:
        import folder_paths
    except Exception:
        return ()
    out = []
    for name in folder_paths.get_filename_list("diffusion_models"):
        if not name.lower().endswith(".safetensors"):
            continue
        path = folder_paths.get_full_path("diffusion_models", name)
        if path:
            out.append(path)
    return out


# Small bounded process cache. Keys include both the curve table hash and concrete
# file identity, so replacing a candidate checkpoint cannot reuse stale weights.
_EMBEDDER_CACHE: dict[tuple[str, tuple], tuple[DenseTimeEmbedder, float]] = {}
_MAX_EMBEDDER_CACHE = 4


def _cache_put(key, value):
    if key in _EMBEDDER_CACHE:
        _EMBEDDER_CACHE.pop(key)
    _EMBEDDER_CACHE[key] = value
    while len(_EMBEDDER_CACHE) > _MAX_EMBEDDER_CACHE:
        _EMBEDDER_CACHE.pop(next(iter(_EMBEDDER_CACHE)))


def find_dense_time_embedder(stage_path: str, table: torch.Tensor) -> tuple[DenseTimeEmbedder, float]:
    """Resolve the exact dense time embedder corresponding to a curve H3 base.

    A checkpoint-local ``dense_time_embedder.safetensors`` is authoritative. If it
    is absent, installed non-curve H3 checkpoints are scored against the curve table
    and the best compatible source is used. If no source meets the strict guard,
    fail closed rather than dropping or approximating the learned AdaLN delta.
    """
    table_cpu = table.detach().to(torch.float32, device="cpu").clone()
    if table_cpu.ndim != 2 or table_cpu.shape[0] < 2:
        raise RuntimeError(
            f"MiniMax-H3 adaln_t_table has invalid shape {tuple(table_cpu.shape)}")
    table_key = _tensor_hash(table_cpu)
    rows = table_cpu.shape[0]
    t_grid = torch.arange(rows, dtype=torch.float32) / float(rows - 1)

    local = os.path.join(stage_path, TIME_EMBEDDER_FILENAME)
    local_exists = os.path.isfile(local)
    if local_exists:
        # A deliberately colocated companion is an explicit operator choice. Never
        # mask a stale/wrong companion by silently falling through to another model.
        candidates = [local]
    else:
        candidates = []
        seen = set()
        for path in _candidate_dense_checkpoints():
            real = os.path.realpath(path)
            if real not in seen:
                candidates.append(path)
                seen.add(real)

    best = None
    errors = []
    local_real = os.path.realpath(local)
    for path in candidates:
        try:
            identity = _file_identity(path)
            cache_key = (table_key, identity)
            cached = _EMBEDDER_CACHE.get(cache_key)
            if cached is not None:
                embedder, residual = cached
            else:
                header = _safetensors_header(path)
                if local_exists and os.path.realpath(path) == local_real:
                    prefix = ""
                else:
                    prefix = _find_dense_prefix(header)
                    # Empty string is a valid root prefix. Only None means no dense
                    # time embedder was found (or a curve checkpoint was detected).
                    if prefix is None:
                        continue
                embedder = _load_embedder(path, prefix)
                dense_grid = embedder.silu_grid(t_grid)
                residual = _curve_fit_residual(table_cpu, dense_grid)
                _cache_put(cache_key, (embedder, residual))
            if best is None or residual < best[1]:
                best = (embedder, residual)
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    if best is None or best[1] > MAX_CURVE_MATCH_RESIDUAL:
        detail = ""
        if best is not None:
            detail = (f" Best candidate was {best[0].source!r} with curve residual "
                      f"{best[1]:.3e}, above {MAX_CURVE_MATCH_RESIDUAL:.1e}.")
        elif errors:
            detail = " Candidate errors: " + "; ".join(errors[:3])
        raise RuntimeError(
            "The loaded MiniMax-H3 base uses a collapsed AdaLN curve, while the VDN "
            "Turbo adapter contains full-width learned AdaLN deltas. Applying those "
            "exactly requires the matching dense H3 time embedder. Place "
            f"{TIME_EMBEDDER_FILENAME!r} in the VDN stage directory (use "
            "tools/extract_h3_time_embedder.py on the matching dense H3 checkpoint), "
            "or keep that dense checkpoint installed under models/diffusion_models. "
            "VDN will not silently drop or project these weights." + detail)
    return best


def _mask_row_values(mask, latent_t, lat_h, lat_w):
    # Prefer the exact current Comfy helper; the local fallback exists only so the
    # pure unit tests do not need to import all of ComfyUI.
    try:
        from comfy.ldm.minimax.model import mask_row_values
        return mask_row_values(mask, latent_t, lat_h, lat_w)
    except Exception:
        m = F.pad(mask, (0, lat_w - mask.shape[-1], 0, lat_h - mask.shape[-2]),
                  mode="replicate")
        m = m.reshape(latent_t, lat_h // 2, 2, lat_w // 2, 2).amax(dim=(2, 4))
        values = m.reshape(-1)
        return None if bool((values >= 1.0 - 1e-3).all()) else values


def minimax_unique_timesteps(dm, x, timestep, context, transformer_options=None,
                             minimax_payload=None, denoise_mask=None,
                             audio_denoise_mask=None) -> list[float]:
    """Mirror current ComfyUI MiniMaxH3Model._forward's ``unique_t`` construction.

    This includes reference/condition rows and per-row masked timesteps. Keeping this
    as a separately tested function is deliberate: a Comfy upstream layout change
    must fail the compatibility CI rather than quietly misalign AdaLN rows.
    """
    import comfy.ldm.common_dit
    import comfy.ldm.minimax.model as mm

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


def make_curve_adaln_forward(base, terms: list[tuple[torch.Tensor, torch.Tensor, float]],
                             state: CurveAdalnState):
    """Patch one curve ``AdalnProj.forward`` without a mutable forward-hook chain."""
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
        for a, b, scale in terms:
            av = a.to(device=x.device, dtype=x.dtype)
            bv = b.to(device=x.device, dtype=x.dtype)
            delta = F.linear(F.linear(dense.to(x.dtype), av), bv)
            x = x + delta * scale
        x = x.view(x.shape[0] * base.modalities, base.expand * base.hidden)
        return x.chunk(base.expand, dim=-1)

    forward._vdn_curve_adaln = True
    return forward
