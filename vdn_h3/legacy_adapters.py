"""Production-bisect adapter path: v1.3.1 math + PR #1 hook lifetime fix.

This module intentionally restores the adapter execution semantics that were
user-validated before the v1.5/upstream reconciliation:

* ordinary LoRA targets use Comfy ``BypassForwardHook`` through the original
  memory-frugal activation-side residual;
* fused INT8 ``mlp.fc2`` targets remain normal weight patches because the fused
  H3 path bypasses ``module.forward``;
* pruned/curve AdaLN targets use the original full-width silu(t_emb) e-grid
  reinjection instead of the v1.5 pruning-affine projection;
* PR #1's cross-provider stack-safe insertion/ejection is retained so current
  Comfy's same-order provider teardown cannot create stale/cyclic forwards.

The purpose is to bisect the real production regression against the last known
working fork semantics, not to introduce another new adapter architecture.
"""
from __future__ import annotations

import logging
import math
import os

import torch
import torch.nn.functional as F

import comfy.ldm.minimax.model
import comfy.lora
import comfy.patcher_extension
import comfy.utils
import comfy.weight_adapter

_log = logging.getLogger("comfy.vdn")

_TURBO_GRID = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir,
    "ComfyUI-MiniMax-H3-Turbo", "h3_silu_temb_grid.safetensors",
)


class _FrugalLoRA(comfy.weight_adapter.LoRAAdapter):
    """v1.3.1 activation-side additive LoRA path."""

    def bypass_forward(self, org_forward, x, *args, **kwargs):
        base_out = org_forward(x, *args, **kwargs)
        if getattr(self, "is_conv", False):
            return super().bypass_forward(org_forward, x, *args, **kwargs)
        up, down, alpha = self.weights[0], self.weights[1], self.weights[2]
        rank = down.shape[0]
        scale = (
            alpha / rank if alpha is not None else 1.0
        ) * getattr(self, "multiplier", 1.0)
        down = down.to(dtype=x.dtype)
        up = up.to(dtype=x.dtype)
        return base_out.add_(
            F.linear(F.linear(x, down), up), alpha=scale
        )


def _is_adaln(module):
    return module.endswith(".adaln_proj.linear")


def _is_pruned_base(dm):
    if getattr(dm, "use_adaln_curves", False):
        return True
    try:
        weight = comfy.utils.get_attr(dm, "blocks.0.adaln_proj.linear.weight")
        return weight.dim() == 2 and weight.shape[-1] < 64
    except Exception:
        return False


def _int8_fused_fc2(dm, modules):
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


def _bypass(new_model, loaded, key_map, modules, sd_keys, strength, hooks):
    manager = comfy.weight_adapter.BypassInjectionManager()
    count = 0
    for module in modules:
        key = key_map[module]
        adapter = loaded.get(key)
        if adapter is None or key not in sd_keys:
            continue
        if isinstance(adapter, comfy.weight_adapter.LoRAAdapter):
            adapter = _FrugalLoRA(adapter.loaded_keys, adapter.weights)
        elif not isinstance(adapter, comfy.weight_adapter.WeightAdapterBase):
            continue
        manager.add_adapter(key, adapter, strength=strength)
        count += 1
    manager.create_injections(new_model.model)
    hooks.extend(manager.hooks)
    return count


def _same_bound_method(left, right):
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
    hook_type = getattr(comfy.weight_adapter, "BypassForwardHook", None)
    owner = getattr(forward, "__self__", None)
    if isinstance(hook_type, type) and isinstance(owner, hook_type):
        return owner
    return None


def _inject_hook_stack_safe(hook):
    """Insert VDN below an already-active standard Comfy bypass chain."""
    if getattr(hook, "original_forward", None) is not None:
        return

    module = hook.module
    previous_forward = module.forward
    hook.inject()

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

    module.forward = previous_forward
    hook.original_forward = inner_forward
    current.original_forward = hook._bypass_forward


def _eject_hook_stack_safe(hook):
    """Remove VDN safely even while another provider remains around it."""
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

    # Another provider may already have detached this VDN hook. Never resurrect a
    # stale forward chain merely to satisfy our own teardown bookkeeping.
    hook.original_forward = None


def _install_injection(new_model, hooks):
    if not hooks:
        return
    owner = new_model.model

    def inject_all(model_patcher):
        del model_patcher
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
        del model_patcher
        for hook in reversed(hooks):
            _eject_hook_stack_safe(hook)
        if getattr(owner, "_vdn_live_hooks", None) is hooks:
            owner._vdn_live_hooks = None

    new_model.set_injections(
        "vdn_lora",
        [comfy.patcher_extension.PatcherInjection(
            inject=inject_all, eject=eject_all
        )],
    )


# ------------------------------------------------------------------ AdaLN e-grid

_EGRID = None


def _egrid():
    global _EGRID
    if _EGRID is None:
        path = os.path.abspath(_TURBO_GRID)
        if not os.path.exists(path):
            raise RuntimeError(
                "VDN v1.3.1 pruned-AdaLN compatibility requires the silu(t_emb) "
                "grid from ComfyUI-MiniMax-H3-Turbo; expected: " + path
            )
        _EGRID = comfy.utils.load_torch_file(path)["silu_t_emb_grid"]
    return _EGRID


def _interp_egrid(unique_t, grid, device, dtype):
    grid = grid.to(device)
    count = grid.shape[0]
    rows = []
    for timestep in unique_t:
        pos = min(max(timestep, 0.0), 1.0) * (count - 1)
        i0 = min(int(math.floor(pos)), count - 2)
        rows.append(
            torch.lerp(
                grid[i0].float(), grid[i0 + 1].float(), pos - i0
            )
        )
    return torch.stack(rows).to(dtype)


def _unique_t(timestep, shift_v, shift_a, payload):
    """Mirror the v1.3.1 MiniMax-H3 unique-timestep row selection."""
    sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    t_v = float(1.0 - sigma_v)
    t_a = float(
        1.0
        - comfy.ldm.minimax.model.time_shift_sigma(
            sigma_v, shift_v, shift_a
        )
    )
    values = {t_v, t_a}
    refs = payload.get("refs") or ()
    if payload.get("keyframes") or any(
        ref.get("kind") == "image" for ref in refs
    ):
        values.add(max(t_v, float(payload.get("visual_cond_noise_aug", 0.999))))
    if any(
        ref.get("kind") == "audio" and ref.get("ref_audio_t", 0) > 0
        for ref in refs
    ):
        values.add(max(t_a, float(payload.get("audio_cond_noise_aug", 1.0))))
    return sorted(values)


def _make_adaln_forward(base, a, b, shared, table=None, egrid=None):
    def forward(t_emb):
        result = base.linear(F.silu(t_emb) if base.apply_silu else t_emb)
        silu_temb = None
        if table is not None and egrid is not None and not base.apply_silu:
            try:
                table_active = table.to(t_emb.device, torch.float32)
                indices = torch.cdist(
                    t_emb.detach().float(), table_active
                ).argmin(dim=1)
                silu_temb = egrid.to(t_emb.device)[indices]
            except Exception:
                silu_temb = None
        if silu_temb is None:
            silu_temb = shared.get("silu_temb")
        if silu_temb is not None and silu_temb.shape[0] == result.shape[0]:
            av = a.to(result.device, result.dtype)
            bv = b.to(result.device, result.dtype)
            sv = silu_temb.to(result.device, result.dtype)
            result = result + (bv @ (av @ sv.T)).T
        result = result.view(
            result.shape[0] * base.modalities,
            base.expand * base.hidden,
        )
        return result.chunk(base.expand, dim=-1)

    forward._vdn_v131_egrid = True
    return forward


def _inject_adaln_egrid(new_model, dm, lora, adaln_modules, strength):
    grid = _egrid()
    shared = {"silu_temb": None}
    shift_v = float(getattr(dm, "sigma_shift_video", 12.0))
    shift_a = float(getattr(dm, "sigma_shift_audio", 3.0))

    table = None
    for name, tensor in list(dm.named_buffers()) + list(dm.named_parameters()):
        if name.endswith("adaln_t_table"):
            table = tensor
            break
    if table is not None and table.shape[0] != grid.shape[0]:
        table = None

    def wrap(executor, *args, **kwargs):
        timestep = args[1] if len(args) > 1 else kwargs.get("timestep")
        context = args[2] if len(args) > 2 else kwargs.get("context")
        payload = kwargs.get("minimax_payload") or {}
        shared["silu_temb"] = _interp_egrid(
            _unique_t(timestep, shift_v, shift_a, payload),
            grid,
            context.device,
            context.dtype,
        )
        return executor(*args, **kwargs)

    new_model.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        "vdn_adaln",
        wrap,
    )
    for name in adaln_modules:
        a = lora[name + ".lora_A.weight"]
        b = lora[name + ".lora_B.weight"] * strength
        key = "diffusion_model." + name.rsplit(".linear", 1)[0]
        new_model.add_object_patch(
            key + ".forward",
            _make_adaln_forward(
                new_model.get_model_object(key), a, b, shared, table, grid
            ),
        )


def apply_adapters(
    new_model,
    converted_by_name,
    strength,
    mode="merge",
    stage_path=None,
    verbose=False,
):
    """Apply adapters with the known-good v1.3.1 semantics.

    ``stage_path`` is accepted only for signature compatibility with the newer node
    plumbing. The v1.3.1 path deliberately does not use the v1.5 pruning-affine
    reconstruction.
    """
    del stage_path
    if mode not in ("merge", "bypass"):
        raise ValueError(f"VDN lora_mode must be merge or bypass, got {mode!r}")

    per_name = strength if isinstance(strength, dict) else None
    dm = new_model.get_model_object("diffusion_model")
    pruned = _is_pruned_base(dm)
    report = {
        "mode": "v1.3.1_pr1_bypass" if mode == "bypass" else "v1.3.1_merge",
        "pruned_adaln": "egrid_v1.3.1" if pruned and mode == "bypass" else None,
    }
    all_hooks = []

    for adapter_name, converted in converted_by_name.items():
        adapter_strength = float(
            per_name.get(adapter_name, 1.0)
            if per_name is not None
            else strength
        )
        modules = sorted(converted)
        lora = {}
        for path, (a, b, scale) in converted.items():
            lora[path + ".lora_A.weight"] = a.contiguous()
            lora[path + ".lora_B.weight"] = b.contiguous()
            lora[path + ".alpha"] = torch.tensor(scale * a.shape[0])
        key_map = {
            module: f"diffusion_model.{module}.weight"
            for module in modules
        }
        loaded = comfy.lora.load_lora(lora, key_map, log_missing=False)
        sd_keys = set(new_model.model.state_dict().keys())

        if mode == "merge":
            if pruned:
                adaln_keys = {
                    key_map[module]
                    for module in modules
                    if _is_adaln(module)
                }
                loaded = {
                    key: value
                    for key, value in loaded.items()
                    if key not in adaln_keys
                }
            count = len(new_model.add_patches(loaded, adapter_strength))
            report[adapter_name] = {
                "native_weight_patches": count,
                "adaln_egrid": 0,
            }
            continue

        backbone = [module for module in modules if not _is_adaln(module)]
        adaln = [module for module in modules if _is_adaln(module)]
        fused_fc2 = set(_int8_fused_fc2(dm, backbone))
        bypass_modules = [
            module for module in backbone if module not in fused_fc2
        ]

        bypass_count = _bypass(
            new_model,
            loaded,
            key_map,
            bypass_modules,
            sd_keys,
            adapter_strength,
            all_hooks,
        ) if bypass_modules else 0

        fused_count = 0
        if fused_fc2:
            accepted = new_model.add_patches(
                {
                    key: value
                    for key, value in loaded.items()
                    if key in {key_map[module] for module in fused_fc2}
                },
                adapter_strength,
            )
            fused_count = len(accepted)

        egrid_count = 0
        if adaln:
            if pruned:
                _inject_adaln_egrid(
                    new_model, dm, lora, adaln, adapter_strength
                )
                egrid_count = len(adaln)
            else:
                bypass_count += _bypass(
                    new_model,
                    loaded,
                    key_map,
                    adaln,
                    sd_keys,
                    adapter_strength,
                    all_hooks,
                )

        report[adapter_name] = {
            "bypass_targets": bypass_count,
            "int8_fc2_patches": fused_count,
            "adaln_egrid": egrid_count,
            "strength": adapter_strength,
        }
        if verbose:
            _log.info(
                "[vdn] v1.3.1 adapter %s: bypass=%d int8_fc2=%d adaln_egrid=%d",
                adapter_name,
                bypass_count,
                fused_count,
                egrid_count,
            )

    _install_injection(new_model, all_hooks)
    _log.warning(
        "[vdn] production bisect active: adapter_semantics=v1.3.1+PR1 "
        "pruned_adaln=%s",
        "egrid" if pruned and mode == "bypass" else "native/skip",
    )
    return report
