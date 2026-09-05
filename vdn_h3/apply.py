"""Apply released VDN adapters through ComfyUI-owned patch mechanisms.

``merge`` uses normal ``ModelPatcher.add_patches`` weight ownership.

``bypass`` is the low-residency execution mode. Ordinary VDN LoRA terms are
implemented with PyTorch forward *post-hooks*, not Comfy ``BypassForwardHook``
and not ``ModelPatcher.weight_function``. VDN therefore never replaces or
splices ``module.forward`` and cannot participate in another provider's mutable
forward-wrapper linked list.

This split is deliberate for quantized MiniMax-H3. The v1.5.0 weight-wrapper
path forced custom quantized layers through a copied/dequantized weight path and
hard-aborted in the production stacked-adapter workflow. v1.5.1 restored the old
mutable-forward bypass chain, but the same real RTX PRO 6000 workflow still
hard-aborted at the first H3 evaluation. VDN now stays out of both mechanisms:
the native quantized base forward runs unchanged, an independently managed Comfy
runtime adapter may wrap that forward if it wants to, and VDN adds its exact
low-rank residual after the module returns.

Fused INT8 ``mlp.fc2`` targets whose H3 fast path bypasses ``module.forward``
remain ordinary Comfy weight patches. Full-width AdaLN LoRAs on curve/pruned H3
bases remain projected once through the exact pruning affine (basis + mean), with
the resulting native curve weight and constant-bias terms owned by normal Comfy
patches in both modes.
"""
from __future__ import annotations

import logging
import threading
import weakref
from collections import defaultdict

import torch
import torch.nn.functional as F

import comfy.lora
import comfy.patcher_extension
import comfy.utils

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


def _int8_fused_fc2(dm, modules):
    """Return fc2 targets whose fused quantized forward bypasses module.forward.

    A PyTorch forward hook cannot observe these H3 fused calls. Keep these terms
    under normal Comfy weight-patch ownership, matching the previously validated
    quantized path.
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


class _PostForwardLoRA:
    """Exact additive LoRA residual without replacing ``module.forward``.

    All terms for one module are fused into one down/up pair per active
    device/dtype. Source tensors stay in their checkpoint-owned representation;
    device copies are cached only while the PatcherInjection is active and are
    dropped on eject/replacement.
    """

    def __init__(self, terms):
        self.terms = tuple(
            (down.detach(), up.detach(), float(scale))
            for down, up, scale in terms
        )
        self._cache = {}

    def _compiled_weights(self, x: torch.Tensor):
        key = (x.device.type, x.device.index, x.dtype)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        downs = []
        ups = []
        for down, up, scale in self.terms:
            down_active = down.to(device=x.device, dtype=x.dtype, non_blocking=True)
            up_active = up.to(device=x.device, dtype=x.dtype, non_blocking=True)
            if scale != 1.0:
                up_active = up_active * scale
            downs.append(down_active)
            ups.append(up_active)

        if not downs:
            raise RuntimeError("VDN post-forward LoRA hook has no adapter terms")
        down_cat = downs[0] if len(downs) == 1 else torch.cat(downs, dim=0)
        up_cat = ups[0] if len(ups) == 1 else torch.cat(ups, dim=1)
        self._cache[key] = (down_cat, up_cat)
        return down_cat, up_cat

    def __call__(self, module, inputs, output):
        if not isinstance(output, torch.Tensor):
            raise RuntimeError(
                f"VDN post-forward bypass expected Tensor output from "
                f"{type(module).__name__}, got {type(output).__name__}"
            )
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError(
                f"VDN post-forward bypass expected the first positional input to "
                f"{type(module).__name__} to be a Tensor"
            )
        x = inputs[0]
        down, up = self._compiled_weights(x)
        delta = F.linear(F.linear(x, down), up)
        return output + delta

    def clear(self):
        self._cache.clear()


# ModelPatcher clones share the same inner model. Track the currently active VDN
# registration outside the model object so a newer clone can replace an older
# registration without private attributes on the model and so an old clone's
# later eject cannot tear down the newer generation.
_ACTIVE_POST_FORWARD = weakref.WeakKeyDictionary()
_ACTIVE_POST_FORWARD_LOCK = threading.RLock()


def _remove_post_forward_registration(registration):
    for handle in reversed(registration["handles"]):
        handle.remove()
    for plan in registration["plans"]:
        plan.clear()


def _install_post_forward_injection(new_model, dm, terms_by_module):
    if not terms_by_module:
        return 0

    plans = []
    for path in sorted(terms_by_module):
        module = comfy.utils.get_attr(dm, path)
        register = getattr(module, "register_forward_hook", None)
        if not callable(register):
            raise RuntimeError(
                f"VDN bypass target {path!r} ({type(module).__name__}) does not "
                "support PyTorch forward hooks"
            )
        plans.append((path, module, _PostForwardLoRA(terms_by_module[path])))

    owner = new_model.model
    token = object()

    def inject_all(model_patcher):
        del model_patcher
        with _ACTIVE_POST_FORWARD_LOCK:
            current = _ACTIVE_POST_FORWARD.get(owner)
            if current is not None and current["token"] is token:
                return
            if current is not None:
                _remove_post_forward_registration(current)

            handles = []
            hook_plans = [plan for _, _, plan in plans]
            try:
                for _, module, plan in plans:
                    handles.append(module.register_forward_hook(plan))
            except Exception:
                for handle in reversed(handles):
                    handle.remove()
                for plan in hook_plans:
                    plan.clear()
                raise

            _ACTIVE_POST_FORWARD[owner] = {
                "token": token,
                "handles": handles,
                "plans": hook_plans,
            }

    def eject_all(model_patcher):
        del model_patcher
        with _ACTIVE_POST_FORWARD_LOCK:
            current = _ACTIVE_POST_FORWARD.get(owner)
            # A newer clone may already have replaced this registration. An old
            # eject must not remove the newer generation.
            if current is None or current["token"] is not token:
                return
            _remove_post_forward_registration(current)
            try:
                del _ACTIVE_POST_FORWARD[owner]
            except KeyError:
                pass

    injection = comfy.patcher_extension.PatcherInjection(
        inject=inject_all,
        eject=eject_all,
    )
    new_model.set_injections("vdn_lora", [injection])
    return len(plans)


def apply_adapters(
    new_model,
    converted_by_name,
    strength,
    mode="merge",
    stage_path=None,
    verbose=False,
):
    """Apply released VDN adapters through merge or forward-post-hook bypass."""
    if mode not in ("merge", "bypass"):
        raise ValueError(
            f"VDN lora_mode must be 'merge' or 'bypass', got {mode!r}"
        )

    per_name = strength if isinstance(strength, dict) else None
    dm = new_model.get_model_object("diffusion_model")
    pruned = _is_pruned_base(dm)
    report = {}
    curve_terms = {}
    runtime_terms = defaultdict(list)

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

            for module in bypass_modules:
                down, up, term_scale = ordinary[module]
                runtime_terms[module].append(
                    (down, up, float(term_scale) * s)
                )
            bypass_targets = len(bypass_modules)
            if fused_terms:
                fused_native_targets = _native_patch_adapter(
                    new_model, fused_terms, s
                )
                native_patches += fused_native_targets

        report[name] = {
            "native_weight_patches": native_patches,
            "runtime_bypass_targets": bypass_targets,
            # Kept as a compatibility/logging alias. In v1.5.2 it counts
            # non-mutating runtime adapter terms, not ModelPatcher weight wrappers.
            "runtime_weight_targets": bypass_targets,
            "fused_native_targets": fused_native_targets,
            "curve_adaln": curve_count,
            "strength": s,
        }
        if verbose:
            _log.info(
                "[vdn] adapter %s: %d native patches, %d post-forward bypass targets, "
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
        forward_hook_modules = _install_post_forward_injection(
            new_model, dm, runtime_terms
        )
        runtime_term_count = sum(len(terms) for terms in runtime_terms.values())
        runtime_report = {
            "mode": "post_forward_hook_bypass",
            "forward_hooks": forward_hook_modules,
            "pytorch_forward_post_hooks": forward_hook_modules,
            "runtime_terms": runtime_term_count,
            "mutable_forward_wrappers": 0,
            "module_forward_untouched": True,
            "weight_wrappers": 0,
            "bias_wrappers": 0,
            "managed_adapter_bytes": 0,
            "delta_buffer_limit_bytes": 0,
            "owner_key": None,
            "stack_safe_cross_provider": True,
            "cross_provider_forward_chain_independent": True,
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
