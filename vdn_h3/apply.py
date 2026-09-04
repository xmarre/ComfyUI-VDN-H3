"""Apply released VDN adapters through ComfyUI-owned patch mechanisms.

VDN intentionally does *not* participate in ``BypassForwardHook`` chains. Ordinary
LoRA targets, including quantized/fused weights, are registered with
``ModelPatcher.add_patches`` so Comfy owns backup/restore, low-VRAM application and
clone semantics. Full-width AdaLN LoRAs on a curve/pruned H3 base are reconstructed
at runtime by :mod:`vdn_h3.curve` and installed as ordinary ModelPatcher object
patches; they are not mutable injection hooks.
"""
from __future__ import annotations

import logging

import torch

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

_log = logging.getLogger("comfy.vdn")


def _is_adaln(module: str) -> bool:
    return module.endswith(".adaln_proj.linear")


def _is_pruned_base(dm) -> bool:
    """Return whether H3 consumes the collapsed AdaLN curve basis."""
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
    """Register one adapter entirely through ``ModelPatcher.add_patches``.

    This is the same path Comfy uses for normal LoRAs and is intentionally used for
    fused/quantized FC2 as well: the patcher knows how a quantized parameter must be
    converted/materialized, whereas a module-forward hook does not.
    """
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


def _install_curve_adaln(new_model, dm, stage_path, terms_by_module):
    """Install exact full-width AdaLN deltas on a curve H3 base.

    ``terms_by_module`` contains the original dense LoRA factors; no low-rank curve
    projection is performed. ModelPatcher owns every changed object and restores it
    on unload before/after other providers according to core lifecycle semantics.
    """
    if not terms_by_module:
        return None
    if stage_path is None:
        raise RuntimeError("VDN curve AdaLN reconstruction requires the stage path")

    table = getattr(dm, "adaln_t_table", None)
    if table is None:
        raise RuntimeError(
            "MiniMax-H3 was detected as a curve/pruned base but has no adaln_t_table")
    if getattr(table, "device", None) is not None and table.device.type == "meta":
        raise RuntimeError(
            "MiniMax-H3 adaln_t_table is still on the meta device; load the base "
            "checkpoint before applying VDN")
    table_cpu = table.detach().to(torch.float32, device="cpu").clone()
    embedder, residual = find_dense_time_embedder(stage_path, table_cpu)
    _log.info("[vdn] curve AdaLN source: %s (base-curve residual %.3e)",
              embedder.source, residual)

    state = CurveAdalnState()
    wrapper_key = "vdn_curve_adaln"
    new_model.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        wrapper_key,
        make_dense_curve_wrapper(dm, embedder, state),
    )

    for module, terms in terms_by_module.items():
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
            make_curve_adaln_forward(base, terms, state),
        )
    return embedder.source, residual


def apply_adapters(new_model, converted_by_name, strength, mode="merge",
                   stage_path=None, verbose=False):
    """Apply converted released adapters to a cloned MiniMax-H3 ModelPatcher.

    ``mode`` remains in the Python signature only so old serialized workflows fail
    with an explicit migration message. The node UI exposes only ``merge``.
    ``strength`` may be one float or ``{adapter_name: float}``.
    """
    if mode != "merge":
        raise RuntimeError(
            "VDN lora_mode='bypass' was removed because mutable module.forward "
            "BypassForwardHook chains are not lifecycle-safe across ModelPatcher "
            "clones/Continuum chunks. Use lora_mode='merge'; adapters now use "
            "ComfyUI's native patch ownership.")

    per_name = strength if isinstance(strength, dict) else None
    dm = new_model.get_model_object("diffusion_model")
    pruned = _is_pruned_base(dm)
    report = {}
    curve_terms = {}

    for name, converted in converted_by_name.items():
        s = float(per_name.get(name, 1.0) if per_name is not None else strength)
        ordinary = {}
        curve_count = 0
        for path, (a, b, scale) in converted.items():
            if pruned and _is_adaln(path):
                curve_terms.setdefault(path, []).append((a, b, float(scale) * s))
                curve_count += 1
            else:
                ordinary[path] = (a, b, scale)

        patched = _native_patch_adapter(new_model, ordinary, s)
        report[name] = {
            "native_weight_patches": patched,
            "curve_adaln": curve_count,
            "strength": s,
        }
        if verbose:
            _log.info("[vdn] adapter %s: %d native weight patches, %d curve AdaLN",
                      name, patched, curve_count)

    curve_source = _install_curve_adaln(
        new_model, dm, stage_path, curve_terms) if curve_terms else None
    if curve_source is not None:
        report["curve_adaln_source"] = {
            "source": curve_source[0], "residual": curve_source[1]}
    return report
