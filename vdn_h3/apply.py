"""Apply released VDN adapters through ComfyUI-owned patch mechanisms.

``merge`` uses normal ``ModelPatcher.add_patches`` weight ownership.

``bypass`` keeps the workflow-facing low-residency mode but deliberately uses
ComfyUI's activation-side ``BypassForwardHook`` contract for ordinary LoRA
targets. The hooks are installed through a stack-safe VDN injection that keeps
VDN inside any independently managed Comfy bypass chain and can splice VDN back
out without restoring stale ``module.forward`` objects.

This is intentional for quantized MiniMax-H3 bases: using a ModelPatcher
``weight_function`` on those layers forces Comfy's dequantized weight path. In
the real stacked VDN + external runtime-DoRA workflow that v1.5.0 path regressed
into a CUDA illegal-memory-access abort. The stack-safe hook path preserves the
native quantized base forward and is the path previously validated with the same
cross-provider Continuum lifecycle.

Full-width AdaLN LoRAs on a curve/pruned H3 base are projected once through the
exact pruning affine (basis + mean). Their projected native curve weight and
constant bias terms stay under ordinary Comfy weight-patch ownership even in
``bypass`` mode; no dense timestep MLP reconstruction is used.
"""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

import comfy.lora
import comfy.patcher_extension
import comfy.utils
import comfy.weight_adapter

from vdn_h3.curve_affine import find_curve_affine, project_curve_terms

_log = logging.getLogger("comfy.vdn")


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


def _load_comfy_adapter(new_model, converted):
    if not converted:
        return {}, {}
    modules = sorted(converted)
    lora = _comfy_lora(converted)
    key_map = {module: f"diffusion_model.{module}.weight" for module in modules}
    state_keys = set(new_model.model.state_dict().keys())
    missing_base = sorted(key_map[m] for m in modules if key_map[m] not in state_keys)
    if missing_base:
        raise RuntimeError(
            "VDN adapter targets are absent from the loaded MiniMax-H3 base: "
            + ", ".join(missing_base[:8])
            + (" ..." if len(missing_base) > 8 else "")
        )

    loaded = comfy.lora.load_lora(lora, key_map, log_missing=False)
    expected = {key_map[m] for m in modules}
    missing_loaded = sorted(expected - set(loaded))
    if missing_loaded:
        raise RuntimeError(
            "ComfyUI could not materialize VDN LoRA targets: "
            + ", ".join(missing_loaded[:8])
            + (" ..." if len(missing_loaded) > 8 else "")
        )
    return loaded, key_map


def _native_patch_adapter(new_model, converted, strength: float) -> int:
    """Register one adapter entirely through ``ModelPatcher.add_patches``."""
    if not converted:
        return 0
    loaded, key_map = _load_comfy_adapter(new_model, converted)
    expected = set(key_map.values())
    accepted = set(
        new_model.add_patches({key: loaded[key] for key in expected}, strength)
    )
    missing_patch = sorted(expected - accepted)
    if missing_patch:
        raise RuntimeError(
            "ModelPatcher rejected VDN adapter targets: "
            + ", ".join(missing_patch[:8])
            + (" ..." if len(missing_patch) > 8 else "")
        )
    return len(accepted)


def _native_patch_projected_curve(new_model, terms_by_module, bias_terms_by_module):
    """Apply projected curve LoRAs + their constant terms through normal patches."""
    weight_patches = 0
    bias_patches = 0
    state_keys = set(new_model.model.state_dict().keys())

    for module in sorted(terms_by_module):
        for term in terms_by_module[module]:
            weight_patches += _native_patch_adapter(new_model, {module: term}, 1.0)

    for module in sorted(bias_terms_by_module):
        key = f"diffusion_model.{module}.bias"
        if key not in state_keys:
            raise RuntimeError(
                f"VDN projected AdaLN bias target is absent from the base: {key}"
            )
        base_bias = comfy.utils.get_attr(new_model.model, key)
        for offset, scale in bias_terms_by_module[module]:
            if tuple(offset.shape) != tuple(base_bias.shape):
                raise RuntimeError(
                    f"VDN projected AdaLN bias {key} has {tuple(offset.shape)}, but "
                    f"the base bias is {tuple(base_bias.shape)}"
                )
            if scale == 0.0:
                continue
            accepted = set(new_model.add_patches({key: (offset,)}, float(scale)))
            if key not in accepted:
                raise RuntimeError(
                    f"ModelPatcher rejected VDN projected AdaLN bias: {key}"
                )
            bias_patches += 1
    return weight_patches, bias_patches


class _FrugalLoRA(comfy.weight_adapter.LoRAAdapter):
    """Activation-side LoRA used by VDN's stack-safe bypass path."""

    def bypass_forward(self, org_forward, x, *args, **kwargs):
        base_out = org_forward(x, *args, **kwargs)
        if getattr(self, "is_conv", False):
            return super().bypass_forward(org_forward, x, *args, **kwargs)

        up, down, alpha = self.weights[0], self.weights[1], self.weights[2]
        rank = down.shape[0]
        scale = (
            (alpha / rank if alpha is not None else 1.0)
            * getattr(self, "multiplier", 1.0)
        )
        down = down.to(dtype=x.dtype)
        up = up.to(dtype=x.dtype)
        return base_out.add_(F.linear(F.linear(x, down), up), alpha=scale)


def _int8_fused_fc2(dm, modules):
    """Return fc2 targets whose fused quantized forward bypasses module.forward.

    These targets cannot use an activation-side hook because H3's fused path reads
    the underlying linear weight directly. Keep them under normal Comfy weight
    patching, matching the previously validated PR #1 behavior.
    """
    fused = []
    for module in modules:
        if not module.endswith(".mlp.fc2"):
            continue
        try:
            weight = comfy.utils.get_attr(dm, module + ".weight")
        except Exception:
            continue
        if (
            getattr(weight, "_layout_cls", None) == "TensorWiseINT8Layout"
            and not getattr(getattr(weight, "_params", None), "transposed", False)
        ):
            fused.append(module)
    return fused


def _bypass(new_model, converted, modules, strength, hooks):
    if not modules:
        return 0
    subset = {module: converted[module] for module in modules}
    loaded, key_map = _load_comfy_adapter(new_model, subset)

    manager = comfy.weight_adapter.BypassInjectionManager()
    installed = 0
    for module in modules:
        key = key_map[module]
        adapter = loaded.get(key)
        if adapter is None:
            continue
        if isinstance(adapter, comfy.weight_adapter.LoRAAdapter):
            adapter = _FrugalLoRA(adapter.loaded_keys, adapter.weights)
        elif not isinstance(adapter, comfy.weight_adapter.WeightAdapterBase):
            raise RuntimeError(
                f"VDN bypass target {key} produced unsupported adapter "
                f"{type(adapter).__name__}"
            )
        manager.add_adapter(key, adapter, strength=float(strength))
        installed += 1

    manager.create_injections(new_model.model)
    hooks.extend(manager.hooks)
    if len(manager.hooks) != installed:
        raise RuntimeError(
            f"VDN bypass created {len(manager.hooks)} hooks for {installed} adapters"
        )
    return installed


def _same_bound_method(left, right):
    """Identity check stable across repeated bound-method attribute reads."""
    if left is right:
        return True
    left_self = getattr(left, "__self__", None)
    right_self = getattr(right, "__self__", None)
    left_func = getattr(left, "__func__", None)
    right_func = getattr(right, "__func__", None)
    return (
        left_self is right_self
        and left_func is not None
        and left_func is right_func
    )


def _bypass_hook_owner(forward):
    """Return the Comfy bypass hook owning a bound forward, if there is one."""
    hook_type = getattr(comfy.weight_adapter, "BypassForwardHook", None)
    owner = getattr(forward, "__self__", None)
    if isinstance(hook_type, type) and isinstance(owner, hook_type):
        return owner
    return None


def _inject_hook_stack_safe(hook):
    """Inject VDN below any already-active Comfy bypass-hook chain."""
    if getattr(hook, "original_forward", None) is not None:
        return

    module = hook.module
    previous_forward = module.forward
    hook.inject()  # lets Comfy set adapter metadata/device placement

    outer = _bypass_hook_owner(previous_forward)
    if outer is None:
        return

    current = outer
    seen = set()
    while True:
        marker = id(current)
        if marker in seen:
            module.forward = previous_forward
            hook.original_forward = None
            raise RuntimeError("VDN found a cyclic Comfy bypass-forward chain")
        seen.add(marker)

        inner_forward = getattr(current, "original_forward", None)
        inner = _bypass_hook_owner(inner_forward)
        if inner is None:
            break
        current = inner

    # hook.inject() temporarily made VDN outermost. Restore the existing chain,
    # then splice VDN immediately above its true/base forward instead.
    module.forward = previous_forward
    hook.original_forward = inner_forward
    current.original_forward = hook._bypass_forward


def _eject_hook_stack_safe(hook):
    """Remove a VDN hook even when another Comfy bypass hook still wraps it."""
    original_forward = getattr(hook, "original_forward", None)
    if original_forward is None:
        return

    module = hook.module
    target = hook._bypass_forward
    current_forward = module.forward

    if _same_bound_method(current_forward, target):
        module.forward = original_forward
        hook.original_forward = None
        return

    current = _bypass_hook_owner(current_forward)
    seen = set()
    while current is not None:
        marker = id(current)
        if marker in seen:
            raise RuntimeError(
                "VDN found a cyclic Comfy bypass-forward chain during eject"
            )
        seen.add(marker)

        inner_forward = getattr(current, "original_forward", None)
        if _same_bound_method(inner_forward, target):
            current.original_forward = original_forward
            hook.original_forward = None
            return
        current = _bypass_hook_owner(inner_forward)

    # Another provider may already have detached this hook. Never resurrect a
    # stale original over the currently valid live chain.
    hook.original_forward = None


def _install_injection(new_model, hooks):
    """Install one clone- and cross-provider-safe VDN bypass injection."""
    if not hooks:
        return

    owner = new_model.model  # clone-shared inner model

    def inject_all(model_patcher):
        old = getattr(owner, "_vdn_live_hooks", None)
        if old:
            for hook in reversed(old):
                _eject_hook_stack_safe(hook)
        try:
            for hook in hooks:
                _inject_hook_stack_safe(hook)
        except Exception:
            for hook in reversed(hooks):
                _eject_hook_stack_safe(hook)
            raise
        owner._vdn_live_hooks = hooks

    def eject_all(model_patcher):
        for hook in reversed(hooks):
            _eject_hook_stack_safe(hook)
        if getattr(owner, "_vdn_live_hooks", None) is hooks:
            owner._vdn_live_hooks = None

    injection = comfy.patcher_extension.PatcherInjection(
        inject=inject_all,
        eject=eject_all,
    )
    new_model.set_injections("vdn_lora", [injection])


def apply_adapters(
    new_model,
    converted_by_name,
    strength,
    mode="merge",
    stage_path=None,
    verbose=False,
):
    """Apply released VDN adapters through merge or stack-safe bypass mode."""
    if mode not in ("merge", "bypass"):
        raise ValueError(
            f"VDN lora_mode must be 'merge' or 'bypass', got {mode!r}"
        )

    per_name = strength if isinstance(strength, dict) else None
    dm = new_model.get_model_object("diffusion_model")
    pruned = _is_pruned_base(dm)
    report = {}
    curve_terms = {}
    all_hooks = []

    for name, converted in converted_by_name.items():
        s = float(per_name.get(name, 1.0) if per_name is not None else strength)
        ordinary = {}
        curve_count = 0
        for path, (a, b, scale) in converted.items():
            effective_scale = float(scale) * s
            if pruned and _is_adaln(path):
                curve_terms.setdefault(path, []).append((a, b, effective_scale))
                curve_count += 1
            else:
                ordinary[path] = (a, b, scale)

        native_patches = 0
        bypass_targets = 0
        fused_native_targets = 0

        if mode == "merge":
            native_patches = (
                _native_patch_adapter(new_model, ordinary, s) if ordinary else 0
            )
        elif ordinary:
            modules = sorted(ordinary)
            fused = set(_int8_fused_fc2(dm, modules))
            bypass_modules = [module for module in modules if module not in fused]
            fused_terms = {module: ordinary[module] for module in sorted(fused)}

            bypass_targets = _bypass(
                new_model,
                ordinary,
                bypass_modules,
                s,
                all_hooks,
            )
            if fused_terms:
                fused_native_targets = _native_patch_adapter(
                    new_model, fused_terms, s
                )
                native_patches += fused_native_targets

        report[name] = {
            "native_weight_patches": native_patches,
            "runtime_bypass_targets": bypass_targets,
            # Kept for one release as a compatibility/logging alias. It no longer
            # means ModelPatcher weight wrappers in v1.5.1.
            "runtime_weight_targets": bypass_targets,
            "fused_native_targets": fused_native_targets,
            "curve_adaln": curve_count,
            "strength": s,
        }
        if verbose:
            _log.info(
                "[vdn] adapter %s: %d native patches, %d stack-safe bypass targets, "
                "%d fused-native targets, %d curve AdaLN",
                name,
                native_patches,
                bypass_targets,
                fused_native_targets,
                curve_count,
            )

    projected_curve = {}
    curve_bias_terms = {}
    affine = None
    if curve_terms:
        if stage_path is None:
            raise RuntimeError("VDN curve AdaLN projection requires the stage path")
        table = getattr(dm, "adaln_t_table", None)
        if table is None:
            raise RuntimeError(
                "MiniMax-H3 was detected as a curve/pruned base but has no "
                "adaln_t_table"
            )
        if (
            getattr(table, "device", None) is not None
            and table.device.type == "meta"
        ):
            raise RuntimeError(
                "MiniMax-H3 adaln_t_table is still on the meta device; load the "
                "base checkpoint before applying VDN"
            )
        affine = find_curve_affine(stage_path, table, base_patcher=new_model)
        projected_curve, curve_bias_terms = project_curve_terms(
            curve_terms, affine
        )
        _log.info(
            "[vdn] curve AdaLN affine: %s (%d weight targets + %d float32 bias targets)",
            affine.source,
            len(projected_curve),
            len(curve_bias_terms),
        )

    curve_weight_patches = 0
    curve_bias_patches = 0
    if projected_curve or curve_bias_terms:
        curve_weight_patches, curve_bias_patches = _native_patch_projected_curve(
            new_model,
            projected_curve,
            curve_bias_terms,
        )

    if mode == "bypass":
        _install_injection(new_model, all_hooks)
        runtime_report = {
            "mode": "stack_safe_bypass",
            "forward_hooks": len(all_hooks),
            "weight_wrappers": 0,
            "bias_wrappers": 0,
            "managed_adapter_bytes": 0,
            "delta_buffer_limit_bytes": 0,
            "owner_key": None,
            "stack_safe_cross_provider": True,
            "projected_curve_weight_patches": curve_weight_patches,
            "projected_curve_bias_patches": curve_bias_patches,
        }
        report["runtime_bypass"] = runtime_report
        # Preserve the v1.5.0 report key for callers/log parsers while making the
        # changed ownership explicit in its values.
        report["runtime_lowvram"] = runtime_report

    if affine is not None:
        report["curve_adaln_projection"] = {
            "source": affine.source,
            "mode": (
                "merge" if mode == "merge" else "bypass_native_projected_patch"
            ),
            "weight_patches": curve_weight_patches,
            "bias_patches": curve_bias_patches,
            "dense_width": int(affine.mean.shape[0]),
            "curve_width": int(affine.basis.shape[0]),
        }

    return report
