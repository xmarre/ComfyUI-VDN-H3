"""Apply released VDN adapters through ComfyUI-owned patch mechanisms.

VDN intentionally does *not* participate in ``BypassForwardHook`` chains. ``merge``
uses normal ``ModelPatcher.add_patches`` weight ownership. ``bypass`` is retained as
the low-VRAM runtime mode for workflow compatibility, but its implementation uses
ComfyUI ``weight_function``/``add_weight_wrapper`` plus a managed additional model
for the low-rank A/B tensors. VDN never traverses, replaces, splices or restores
LoRA-target ``module.forward`` methods.

Full-width AdaLN LoRAs on a curve/pruned H3 base are projected once through the
exact pruning affine (basis + mean). Their weight term then targets the native
small AdaLN coordinate projection and their essential constant term targets its
float32 bias. No dense timestep MLP or AdaLN forward object patch is required.
"""
from __future__ import annotations

import logging
import uuid

import torch

import comfy.float
import comfy.lora
import comfy.model_management
import comfy.utils

from vdn_h3.curve_affine import find_curve_affine, project_curve_terms
from vdn_h3.managed import make_managed_runtime_lora_patcher

_log = logging.getLogger("comfy.vdn")
_RUNTIME_DELTA_BUFFER_BYTES = 8 << 20
_RUNTIME_MODEL_KEY_PREFIX = "vdn_runtime_lora_"


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


def _native_patch_projected_curve(new_model, terms_by_module, bias_terms_by_module):
    """Apply projected curve LoRAs + their constant terms through normal patches."""
    weight_patches = 0
    bias_patches = 0
    state_keys = set(new_model.model.state_dict().keys())

    # Keep projected weight updates low-rank and let Comfy own their normal LoRA
    # materialization lifecycle. Multiple adapters can target the same module; each
    # add_patches call appends rather than replacing the prior patch.
    for module in sorted(terms_by_module):
        for term in terms_by_module[module]:
            weight_patches += _native_patch_adapter(new_model, {module: term}, 1.0)

    for module in sorted(bias_terms_by_module):
        key = f"diffusion_model.{module}.bias"
        if key not in state_keys:
            raise RuntimeError(f"VDN projected AdaLN bias target is absent from the base: {key}")
        base_bias = comfy.utils.get_attr(new_model.model, key)
        for offset, scale in bias_terms_by_module[module]:
            if tuple(offset.shape) != tuple(base_bias.shape):
                raise RuntimeError(
                    f"VDN projected AdaLN bias {key} has {tuple(offset.shape)}, but "
                    f"the base bias is {tuple(base_bias.shape)}")
            if scale == 0.0:
                continue
            accepted = set(new_model.add_patches({key: (offset,)}, float(scale)))
            if key not in accepted:
                raise RuntimeError(f"ModelPatcher rejected VDN projected AdaLN bias: {key}")
            bias_patches += 1
    return weight_patches, bias_patches


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


class _RuntimeBiasWeight:
    """Stateless runtime wrapper for projected pruned-AdaLN constant terms."""

    __slots__ = ("key", "module", "source", "_vdn_runtime_bias")

    def __init__(self, key: str, module: str, source):
        self.key = key
        self.module = module
        self.source = source
        self._vdn_runtime_bias = True

    def __call__(self, bias):
        if not isinstance(bias, torch.Tensor) or not bias.is_floating_point():
            raise RuntimeError(
                f"VDN runtime AdaLN bias for {self.key} expected a floating tensor; "
                f"got {type(bias).__name__} / {getattr(bias, 'dtype', None)}")
        compute_dtype = comfy.model_management.lora_compute_dtype(bias.device)
        if compute_dtype is None:
            compute_dtype = bias.dtype
        out = bias.to(dtype=compute_dtype, copy=True)
        for offset, scale in self.source.bias_terms_on(
                self.module, out.device, compute_dtype):
            if scale != 0.0:
                out.add_(offset, alpha=float(scale))
        if out.dtype != bias.dtype:
            out = comfy.float.stochastic_rounding(
                out, bias.dtype, seed=comfy.utils.string_to_seed(self.key))
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


def _validate_runtime_bias_targets(new_model, terms_by_module):
    state_keys = set(new_model.model.state_dict().keys())
    for module in sorted(terms_by_module):
        key = f"diffusion_model.{module}.bias"
        if key not in state_keys:
            raise RuntimeError(f"VDN runtime AdaLN bias target is absent from the base: {key}")
        owner_key = f"diffusion_model.{module}"
        owner = comfy.utils.get_attr(new_model.model, owner_key)
        if not hasattr(owner, "bias_function"):
            raise RuntimeError(
                f"VDN runtime AdaLN bias requires Comfy's bias_function contract, but "
                f"{owner_key} ({type(owner).__name__}) does not expose it")
        bias = comfy.utils.get_attr(new_model.model, key)
        for offset, _scale in terms_by_module[module]:
            if tuple(offset.shape) != tuple(bias.shape):
                raise RuntimeError(
                    f"VDN runtime AdaLN bias {key} has {tuple(offset.shape)}, but "
                    f"the base bias is {tuple(bias.shape)}")


def _install_runtime_weight_adapters(new_model, terms_by_module, managed):
    if not terms_by_module:
        return 0
    _validate_runtime_weight_targets(new_model, terms_by_module)
    installed = 0
    for module in sorted(terms_by_module):
        key = f"diffusion_model.{module}.weight"
        new_model.add_weight_wrapper(key, _RuntimeLoRAWeight(key, module, managed))
        installed += 1
    return installed


def _install_runtime_bias_adapters(new_model, terms_by_module, managed):
    if not terms_by_module:
        return 0
    _validate_runtime_bias_targets(new_model, terms_by_module)
    installed = 0
    for module in sorted(terms_by_module):
        key = f"diffusion_model.{module}.bias"
        new_model.add_weight_wrapper(key, _RuntimeBiasWeight(key, module, managed))
        installed += 1
    return installed


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

    projected_curve = {}
    curve_bias_terms = {}
    affine = None
    if curve_terms:
        if stage_path is None:
            raise RuntimeError("VDN curve AdaLN projection requires the stage path")
        table = getattr(dm, "adaln_t_table", None)
        if table is None:
            raise RuntimeError(
                "MiniMax-H3 was detected as a curve/pruned base but has no adaln_t_table")
        if getattr(table, "device", None) is not None and table.device.type == "meta":
            raise RuntimeError(
                "MiniMax-H3 adaln_t_table is still on the meta device; load the base "
                "checkpoint before applying VDN")
        affine = find_curve_affine(stage_path, table, base_patcher=new_model)
        projected_curve, curve_bias_terms = project_curve_terms(curve_terms, affine)
        _log.info(
            "[vdn] curve AdaLN affine: %s (%d weight targets + %d float32 bias targets)",
            affine.source, len(projected_curve), len(curve_bias_terms))

    if mode == "merge" and projected_curve:
        curve_weight_patches, curve_bias_patches = _native_patch_projected_curve(
            new_model, projected_curve, curve_bias_terms)
        report["curve_adaln_projection"] = {
            "source": affine.source,
            "mode": "merge",
            "weight_patches": curve_weight_patches,
            "bias_patches": curve_bias_patches,
            "dense_width": int(affine.mean.shape[0]),
            "curve_width": int(affine.basis.shape[0]),
        }
        return report

    if projected_curve:
        runtime_terms = _merge_runtime_terms(runtime_terms, projected_curve)

    managed = None
    managed_bytes = 0
    runtime_owner_key = None
    if runtime_terms or curve_bias_terms:
        managed, managed_patcher = make_managed_runtime_lora_patcher(
            runtime_terms, new_model, bias_terms_by_module=curve_bias_terms)
        # Core currently does not compare weight_wrapper_patches in
        # ModelPatcher.clone_has_same_weights(), and it can return early when both
        # patchers have no ordinary weight patches. Give every Apply execution a
        # distinct additional-model key so changing strength/options cannot reuse a
        # still-loaded wrapper set from an older clone. Clones of *this* result keep
        # the same key and remain equivalent.
        runtime_owner_key = _RUNTIME_MODEL_KEY_PREFIX + uuid.uuid4().hex
        new_model.set_additional_models(runtime_owner_key, [managed_patcher])
        managed_bytes = sum(
            p.numel() * p.element_size() for p in managed.parameters())

    runtime_wrappers = _install_runtime_weight_adapters(
        new_model, runtime_terms, managed) if runtime_terms else 0
    runtime_bias_wrappers = _install_runtime_bias_adapters(
        new_model, curve_bias_terms, managed) if curve_bias_terms else 0
    if runtime_wrappers or runtime_bias_wrappers:
        report["runtime_lowvram"] = {
            "weight_wrappers": runtime_wrappers,
            "bias_wrappers": runtime_bias_wrappers,
            "forward_hooks": 0,
            "managed_adapter_bytes": managed_bytes,
            "delta_buffer_limit_bytes": _RUNTIME_DELTA_BUFFER_BYTES,
            "owner_key": runtime_owner_key,
        }

    if affine is not None:
        report["curve_adaln_projection"] = {
            "source": affine.source,
            "mode": "bypass",
            "weight_targets": len(projected_curve),
            "bias_targets": len(curve_bias_terms),
            "dense_width": int(affine.mean.shape[0]),
            "curve_width": int(affine.basis.shape[0]),
            "managed_adapter_bytes": managed_bytes,
            "owner_key": runtime_owner_key,
        }
    return report
