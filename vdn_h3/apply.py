"""Apply released VDN adapters through ComfyUI-owned patch mechanisms.

VDN intentionally does *not* participate in ``BypassForwardHook`` chains. ``merge``
uses normal ``ModelPatcher.add_patches`` weight ownership. ``bypass`` is retained as
the low-VRAM runtime mode for workflow compatibility, but its implementation uses
ComfyUI ``weight_function``/``add_weight_wrapper`` plus a managed additional model
for the low-rank A/B tensors. VDN never traverses, replaces, splices or restores
LoRA-target ``module.forward`` methods.

Full-width AdaLN LoRAs on a curve/pruned H3 base are reconstructed at runtime by
:mod:`vdn_h3.curve`; their A/B factors share the same Comfy-managed low-rank model.
"""
from __future__ import annotations

import logging

import torch

import comfy.float
import comfy.lora
import comfy.model_management
import comfy.patcher_extension
import comfy.utils

from vdn_h3.curve import (
    CurveAdalnState,
    find_dense_time_embedder,
    make_curve_adaln_forward,
    make_dense_curve_wrapper,
)
from vdn_h3.managed import make_managed_runtime_lora_patcher

_log = logging.getLogger("comfy.vdn")
_RUNTIME_DELTA_BUFFER_BYTES = 8 << 20


def _is_adaln(module: str) -> bool:
    return module.endswith(".adaln_proj.linear")


def _is_pruned_base(dm) -> bool:
    if getattr(dm, "use_adaln_curves", False):
        return True
    try:
        w = comfy.utils.get_attr(dm, "blocks.0.adaln_proj.linear.weight")
        return w.dim() == 2 and w.shape[-1] < 64
    except Exception:
        return False


def _comfy_lora(converted):
    lora = {}
    for path, (a, b, scale) in converted.items():
        rank = a.shape[0]
        lora[path + ".lora_A.weight"] = a.contiguous()
        lora[path + ".lora_B.weight"] = b.contiguous()
        lora[path + ".alpha"] = torch.tensor(scale * rank)
    return lora


def _native_patch_adapter(new_model, converted, strength: float) -> int:
    """Register one adapter entirely through ``ModelPatcher.add_patches``."""
    if not converted:
        return 0
    modules = sorted(converted)
    lora = _comfy_lora(converted)
    key_map = {module: f"diffusion_model.{module}.weight" for module in modules}
    state_keys = set(new_model.model.state_dict().keys())
    missing_base = sorted(key_map[m] for m in modules if key_map[m] not in state_keys)
    if missing_base:
        raise RuntimeError(
            "VDN adapter targets are absent from the loaded MiniMax-H3 base: "
            + ", ".join(missing_base[:8])
            + (" ..." if len(missing_base) > 8 else ""))

    loaded = comfy.lora.load_lora(lora, key_map, log_missing=False)
    expected = {key_map[m] for m in modules}
    missing_loaded = sorted(expected - set(loaded))
    if missing_loaded:
        raise RuntimeError(
            "ComfyUI could not materialize VDN LoRA targets: "
            + ", ".join(missing_loaded[:8])
            + (" ..." if len(missing_loaded) > 8 else ""))
    accepted = set(new_model.add_patches(
        {key: loaded[key] for key in expected}, strength))
    missing_patch = sorted(expected - accepted)
    if missing_patch:
        raise RuntimeError(
            "ModelPatcher rejected VDN adapter targets: "
            + ", ".join(missing_patch[:8])
            + (" ..." if len(missing_patch) > 8 else ""))
    return len(accepted)


class _RuntimeLoRAWeight:
    """Stateless runtime weight wrapper with bounded delta temporary memory."""

    __slots__ = ("key", "module", "source", "_vdn_runtime_lora")

    def __init__(self, key: str, module: str, source):
        self.key = key
        self.module = module
        self.source = source
        self._vdn_runtime_lora = True

    @staticmethod
    def _rows_per_chunk(out, input_dim):
        row_bytes = max(1, int(input_dim) * out.element_size())
        return max(1, _RUNTIME_DELTA_BUFFER_BYTES // row_bytes)

    def __call__(self, weight):
        if not isinstance(weight, torch.Tensor) or not weight.is_floating_point():
            raise RuntimeError(
                f"VDN runtime LoRA for {self.key} expected a floating compute weight; "
                f"got {type(weight).__name__} / {getattr(weight, 'dtype', None)}")

        compute_dtype = comfy.model_management.lora_compute_dtype(weight.device)
        if compute_dtype is None:
            compute_dtype = weight.dtype
        out = weight.to(dtype=compute_dtype, copy=True)
        terms = self.source.terms_on(self.module, out.device, compute_dtype)

        for av, bv, scale in terms:
            if scale == 0.0:
                continue
            rows_per_chunk = self._rows_per_chunk(out, av.shape[1])
            for start in range(0, bv.shape[0], rows_per_chunk):
                stop = min(start + rows_per_chunk, bv.shape[0])
                delta = torch.mm(bv[start:stop], av)
                if scale != 1.0:
                    delta.mul_(scale)
                out[start:stop].add_(delta)

        if out.dtype != weight.dtype:
            out = comfy.float.stochastic_rounding(
                out, weight.dtype, seed=comfy.utils.string_to_seed(self.key))
        return out


def _validate_runtime_weight_targets(new_model, terms_by_module):
    state_keys = set(new_model.model.state_dict().keys())
    for module in sorted(terms_by_module):
        key = f"diffusion_model.{module}.weight"
        if key not in state_keys:
            raise RuntimeError(f"VDN runtime adapter target is absent from the base: {key}")

        owner_key = f"diffusion_model.{module}"
        owner = comfy.utils.get_attr(new_model.model, owner_key)
        if not hasattr(owner, "weight_function"):
            raise RuntimeError(
                f"VDN runtime low-VRAM mode requires Comfy's weight_function contract, "
                f"but {owner_key} ({type(owner).__name__}) does not expose it. Use "
                "lora_mode='merge' for this unsupported module implementation.")

        weight = comfy.utils.get_attr(new_model.model, key)
        expected_shape = tuple(weight.shape)
        for a, b, _scale in terms_by_module[module]:
            if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
                raise RuntimeError(
                    f"VDN runtime LoRA {key} has incompatible A{tuple(a.shape)} "
                    f"B{tuple(b.shape)}")
            delta_shape = (b.shape[0], a.shape[1])
            if delta_shape != expected_shape:
                raise RuntimeError(
                    f"VDN runtime LoRA {key} produces {delta_shape}, but the base "
                    f"weight is {expected_shape}")


def _install_runtime_weight_adapters(new_model, terms_by_module, managed):
    if not terms_by_module:
        return 0
    _validate_runtime_weight_targets(new_model, terms_by_module)
    installed = 0
    for module in sorted(terms_by_module):
        key = f"diffusion_model.{module}.weight"
        new_model.add_weight_wrapper(
            key, _RuntimeLoRAWeight(key, module, managed))
        installed += 1
    return installed


def _install_curve_adaln(new_model, dm, stage_path, terms_by_module, managed):
    """Install exact full-width AdaLN deltas on a curve H3 base."""
    if not terms_by_module:
        return None
    if stage_path is None:
        raise RuntimeError("VDN curve AdaLN reconstruction requires the stage path")
    if managed is None:
        raise RuntimeError("VDN curve AdaLN terms have no managed runtime owner")

    table = getattr(dm, "adaln_t_table", None)
    if table is None:
        raise RuntimeError(
            "MiniMax-H3 was detected as a curve/pruned base but has no adaln_t_table")
    if getattr(table, "device", None) is not None and table.device.type == "meta":
        raise RuntimeError(
            "MiniMax-H3 adaln_t_table is still on the meta device; load the base "
            "checkpoint before applying VDN")
    table_cpu = table.detach().to(device="cpu", dtype=torch.float32).clone()
    embedder, residual = find_dense_time_embedder(stage_path, table_cpu)
    _log.info("[vdn] curve AdaLN source: %s (base-curve residual %.3e)",
              embedder.source, residual)

    state = CurveAdalnState()
    new_model.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        "vdn_curve_adaln",
        make_dense_curve_wrapper(dm, embedder, state),
    )

    for module in terms_by_module:
        parent = module.rsplit(".linear", 1)[0]
        object_key = f"diffusion_model.{parent}.forward"
        existing = new_model.object_patches.get(object_key)
        if existing is not None and not getattr(existing, "_vdn_curve_adaln", False):
            raise RuntimeError(
                f"VDN needs to patch {object_key} for exact curve AdaLN, but another "
                "object patch already owns that forward. Apply VDN before the "
                "conflicting provider or remove that incompatible patch.")
        base = new_model.get_model_object(f"diffusion_model.{parent}")
        new_model.add_object_patch(
            object_key,
            make_curve_adaln_forward(
                base, managed, state, managed_module=module),
        )
    return embedder.source, residual


def _merge_runtime_terms(*groups):
    combined = {}
    for group in groups:
        for module, terms in group.items():
            combined.setdefault(module, []).extend(terms)
    return combined


def apply_adapters(new_model, converted_by_name, strength, mode="merge",
                   stage_path=None, verbose=False):
    """Apply released adapters through merge or safe runtime-low-VRAM mode."""
    if mode not in ("merge", "bypass"):
        raise ValueError(f"VDN lora_mode must be 'merge' or 'bypass', got {mode!r}")

    per_name = strength if isinstance(strength, dict) else None
    dm = new_model.get_model_object("diffusion_model")
    pruned = _is_pruned_base(dm)
    report = {}
    curve_terms = {}
    runtime_terms = {}

    for name, converted in converted_by_name.items():
        s = float(per_name.get(name, 1.0) if per_name is not None else strength)
        ordinary = {}
        runtime_count = 0
        curve_count = 0
        for path, (a, b, scale) in converted.items():
            effective_scale = float(scale) * s
            if pruned and _is_adaln(path):
                curve_terms.setdefault(path, []).append((a, b, effective_scale))
                curve_count += 1
            elif mode == "bypass":
                runtime_terms.setdefault(path, []).append((a, b, effective_scale))
                runtime_count += 1
            else:
                ordinary[path] = (a, b, scale)

        patched = _native_patch_adapter(new_model, ordinary, s) if ordinary else 0
        report[name] = {
            "native_weight_patches": patched,
            "runtime_weight_targets": runtime_count,
            "curve_adaln": curve_count,
            "strength": s,
        }
        if verbose:
            _log.info(
                "[vdn] adapter %s: %d native patches, %d runtime-lowvram targets, "
                "%d curve AdaLN",
                name, patched, runtime_count, curve_count)

    managed_terms = _merge_runtime_terms(runtime_terms, curve_terms)
    managed = None
    managed_bytes = 0
    if managed_terms:
        managed, managed_patcher = make_managed_runtime_lora_patcher(
            managed_terms, new_model)
        new_model.set_additional_models("vdn_runtime_lora", [managed_patcher])
        managed_bytes = sum(
            p.numel() * p.element_size() for p in managed.parameters())

    runtime_wrappers = _install_runtime_weight_adapters(
        new_model, runtime_terms, managed) if runtime_terms else 0
    if runtime_wrappers:
        report["runtime_lowvram"] = {
            "weight_wrappers": runtime_wrappers,
            "forward_hooks": 0,
            "managed_adapter_bytes": managed_bytes,
            "delta_buffer_limit_bytes": _RUNTIME_DELTA_BUFFER_BYTES,
        }

    curve_source = _install_curve_adaln(
        new_model, dm, stage_path, curve_terms, managed) if curve_terms else None
    if curve_source is not None:
        report["curve_adaln_source"] = {
            "source": curve_source[0],
            "residual": curve_source[1],
            "managed_adapter_bytes": managed_bytes,
        }
    return report
