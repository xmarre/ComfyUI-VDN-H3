"""Affine projection of released full-width AdaLN LoRAs onto pruned H3 curves.

MiniMax-H3's pruned AdaLN representation is constructed from

    dense(t) ~= mean + curve(t) @ basis

where ``curve(t)`` is the small coordinate vector stored in ``adaln_t_table``.
A released full-width LoRA ``B @ A`` therefore maps onto the pruned coordinates as

    A_pruned = A @ basis.T
    bias_offset = B @ (A @ mean)

The constant term is essential.  This module resolves the exact affine map used by
an installed pruned checkpoint (or a tiny checkpoint-local companion file), performs
the projection once in float64, then returns ordinary low-rank weight terms plus
float32 bias offsets.  No dense timestep MLP is needed at sampling time.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass
from typing import Iterable

import torch
from safetensors import safe_open

AFFINE_FILENAME = "adaln_affine.safetensors"
_BASIS_KEY = "adaln_basis"
_MEAN_KEY = "adaln_mean"
_TABLE_KEYS = ("adaln_t_table", "time_embedder.table")
_MAX_AFFINE_CACHE = 4


def _file_identity(path: str) -> tuple[str, int, int, int | None]:
    real = os.path.realpath(path)
    st = os.stat(real)
    return real, st.st_mtime_ns, st.st_size, getattr(st, "st_ino", None)


def _tensor_hash(t: torch.Tensor) -> str:
    x = t.detach().to(device="cpu", dtype=torch.float32).contiguous()
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


@dataclass(frozen=True)
class CurveAffine:
    basis: torch.Tensor
    mean: torch.Tensor
    source: str
    identity: tuple
    table_hash: str


def _validate_affine(basis: torch.Tensor, mean: torch.Tensor, table: torch.Tensor, source: str):
    if basis.ndim != 2 or mean.ndim != 1:
        raise ValueError(
            f"{source}: expected adaln_basis [curve,dense] and adaln_mean [dense], got "
            f"{tuple(basis.shape)} and {tuple(mean.shape)}")
    if basis.shape[1] != mean.shape[0]:
        raise ValueError(
            f"{source}: affine dense width mismatch: basis={tuple(basis.shape)} "
            f"mean={tuple(mean.shape)}")
    if table.ndim != 2 or basis.shape[0] != table.shape[1]:
        raise ValueError(
            f"{source}: affine curve rank {basis.shape[0]} does not match "
            f"adaln_t_table {tuple(table.shape)}")


def _load_affine(path: str, table: torch.Tensor, explicit_local: bool) -> CurveAffine:
    identity = _file_identity(path)
    header = _safetensors_header(path)
    if _BASIS_KEY not in header or _MEAN_KEY not in header:
        raise ValueError(f"{path}: missing {_BASIS_KEY}/{_MEAN_KEY}")

    with safe_open(path, framework="pt", device="cpu") as handle:
        basis = handle.get_tensor(_BASIS_KEY).to(torch.float32).clone()
        mean = handle.get_tensor(_MEAN_KEY).to(torch.float32).clone()
        candidate_table = None
        for key in _TABLE_KEYS:
            if key in header:
                candidate_table = handle.get_tensor(key).to(torch.float32).clone()
                break
        metadata = handle.metadata() or {}

    table_cpu = table.detach().to(device="cpu", dtype=torch.float32).contiguous()
    _validate_affine(basis, mean, table_cpu, path)
    table_hash = _tensor_hash(table_cpu)

    if candidate_table is not None:
        if tuple(candidate_table.shape) != tuple(table_cpu.shape) or not torch.equal(
                candidate_table, table_cpu):
            raise ValueError(f"{path}: AdaLN affine belongs to a different curve table")
    else:
        declared = metadata.get("adaln_table_sha256")
        if declared is not None and declared != table_hash:
            raise ValueError(f"{path}: adaln_table_sha256 does not match the loaded base")
        if not explicit_local and declared is None:
            # An installed model without its table cannot prove that the affine map is
            # for this basis.  A deliberately placed stage-local sidecar is allowed to
            # be authoritative, matching normal checkpoint companion-file semantics.
            raise ValueError(f"{path}: affine map has no curve-table identity")

    return CurveAffine(
        basis=basis.contiguous(), mean=mean.contiguous(), source=path,
        identity=identity, table_hash=table_hash)


def _base_checkpoint_path(base_patcher) -> str | None:
    cached = getattr(base_patcher, "cached_patcher_init", None)
    try:
        args = cached[1]
        path = args[0]
    except Exception:
        return None
    return path if isinstance(path, str) and os.path.isfile(path) else None


def _candidate_affine_checkpoints(base_patcher=None) -> Iterable[str]:
    out = []
    seen = set()

    current = _base_checkpoint_path(base_patcher) if base_patcher is not None else None
    if current is not None:
        directory = os.path.dirname(os.path.realpath(current))
        try:
            siblings = sorted(os.listdir(directory))
        except OSError:
            siblings = []
        # Prefer the selected model and its siblings.  In particular, the repaired
        # BF16 Comfy artifact can retain adaln_basis/adaln_mean while its INT8
        # derivatives intentionally omit them.
        for path in [current] + [os.path.join(directory, name) for name in siblings]:
            if not path.lower().endswith(".safetensors") or not os.path.isfile(path):
                continue
            real = os.path.realpath(path)
            if real not in seen:
                seen.add(real)
                out.append(path)

    try:
        import folder_paths
        names = folder_paths.get_filename_list("diffusion_models")
    except Exception:
        names = ()
    for name in names:
        if not name.lower().endswith(".safetensors"):
            continue
        try:
            path = folder_paths.get_full_path("diffusion_models", name)
        except Exception:
            path = None
        if not path:
            continue
        real = os.path.realpath(path)
        if real not in seen:
            seen.add(real)
            out.append(path)
    return out


_AFFINE_CACHE: dict[tuple[str, tuple], CurveAffine] = {}


def _cache_put(key, value):
    if key in _AFFINE_CACHE:
        _AFFINE_CACHE.pop(key)
    _AFFINE_CACHE[key] = value
    while len(_AFFINE_CACHE) > _MAX_AFFINE_CACHE:
        _AFFINE_CACHE.pop(next(iter(_AFFINE_CACHE)))


def find_curve_affine(stage_path: str, table: torch.Tensor, base_patcher=None) -> CurveAffine:
    """Resolve the exact basis/mean pair corresponding to the loaded curve base."""
    table_cpu = table.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if table_cpu.ndim != 2 or table_cpu.shape[0] < 2:
        raise RuntimeError(f"MiniMax-H3 adaln_t_table has invalid shape {tuple(table_cpu.shape)}")
    table_hash = _tensor_hash(table_cpu)

    local = os.path.join(stage_path, AFFINE_FILENAME)
    candidates = []
    if os.path.isfile(local):
        candidates.append((local, True))
    candidates.extend((path, False) for path in _candidate_affine_checkpoints(base_patcher))

    errors = []
    for path, explicit_local in candidates:
        try:
            identity = _file_identity(path)
            cache_key = (table_hash, identity)
            cached = _AFFINE_CACHE.get(cache_key)
            if cached is not None:
                return cached
            header = _safetensors_header(path)
            if _BASIS_KEY not in header or _MEAN_KEY not in header:
                continue
            affine = _load_affine(path, table_cpu, explicit_local=explicit_local)
            _cache_put(cache_key, affine)
            return affine
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    detail = ""
    if errors:
        detail = " Candidate errors: " + "; ".join(errors[:3])
    raise RuntimeError(
        "The loaded MiniMax-H3 base uses a collapsed AdaLN curve, while the VDN "
        "Turbo adapter contains full-width learned AdaLN deltas. Projecting those "
        "onto the pruned coordinates requires the exact pruning affine "
        "('adaln_basis' + 'adaln_mean'). Place 'adaln_affine.safetensors' in the "
        "VDN stage directory, or keep a matching pruned BF16 checkpoint containing "
        "those two auxiliaries under models/diffusion_models. Quantized derivatives "
        "may intentionally omit them; the companion is only about 97 KB. VDN will "
        "not silently drop the 51 AdaLN updates or use an unverified basis." + detail)


def project_curve_terms(terms_by_module, affine: CurveAffine):
    """Project full-width LoRA factors and return weight terms + constant bias terms."""
    basis = affine.basis.to(torch.float64)
    mean = affine.mean.to(torch.float64)
    projected = {}
    bias_terms = {}

    for module, terms in terms_by_module.items():
        out_terms = []
        out_bias = []
        for a, b, scale in terms:
            if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
                raise RuntimeError(
                    f"VDN curve LoRA {module} has incompatible A{tuple(a.shape)} "
                    f"B{tuple(b.shape)}")
            if a.shape[1] != mean.shape[0]:
                raise RuntimeError(
                    f"VDN curve LoRA {module} expects dense AdaLN width {a.shape[1]}, "
                    f"but affine source {affine.source!r} has {mean.shape[0]}")
            a64 = a.detach().to(device="cpu", dtype=torch.float64)
            b64 = b.detach().to(device="cpu", dtype=torch.float64)
            a_pruned = (a64 @ basis.T).to(torch.float32).contiguous()
            offset = (b64 @ (a64 @ mean)).to(torch.float32).contiguous()
            # Keep B in its checkpoint dtype for persistent low-rank storage.  The
            # affine itself and constant are computed from the stored values in
            # float64 exactly once, matching the published pruned-H3 projection.
            out_terms.append((a_pruned, b.detach().contiguous(), float(scale)))
            out_bias.append((offset, float(scale)))
        projected[module] = out_terms
        bias_terms[module] = out_bias
    return projected, bias_terms
