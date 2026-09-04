"""Applying the converted VDN adapters onto a cloned ModelPatcher.

The application machinery is a port of the ComfyUI-MiniMax-H3-Turbo node's
battle-tested paths (bypass injection with a memory-frugal additive LoRA, merge for
quantized-fused weights, and the e-grid re-injection for curve/pruned bases' adaln
updates), adapted to VDN's already-converted ComfyUI key space.
"""
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

_TURBO_GRID = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir,
                           "ComfyUI-MiniMax-H3-Turbo", "h3_silu_temb_grid.safetensors")


class _FrugalLoRA(comfy.weight_adapter.LoRAAdapter):
    """LoRA bypass adapter with an in-place additive path (ported from the
    MiniMax-H3-Turbo node): accumulates up(down(x)) * scale straight into the base
    output instead of allocating the full-size projection three times."""

    def bypass_forward(self, org_forward, x, *args, **kwargs):
        base_out = org_forward(x, *args, **kwargs)
        if getattr(self, "is_conv", False):
            return super().bypass_forward(org_forward, x, *args, **kwargs)
        up, down, alpha = self.weights[0], self.weights[1], self.weights[2]
        rank = down.shape[0]
        scale = (alpha / rank if alpha is not None else 1.0) \
            * getattr(self, "multiplier", 1.0)
        down = down.to(dtype=x.dtype)
        up = up.to(dtype=x.dtype)
        return base_out.add_(torch.nn.functional.linear(
            torch.nn.functional.linear(x, down), up), alpha=scale)


def _int8_fused_fc2(dm, modules):
    """MLP fc2 modules riding ComfyUI's fused int8 matmul: their fused forward reads
    linear.weight directly and never calls the module forward, so a bypass hook would
    silently drop the LoRA. Those must go through the merge/weight-function path.
    (Ported from the MiniMax-H3-Turbo node.)"""
    fused = []
    for m in modules:
        if not m.endswith(".mlp.fc2"):
            continue
        try:
            w = comfy.utils.get_attr(dm, m + ".weight")
        except Exception:
            continue
        if (getattr(w, "_layout_cls", None) == "TensorWiseINT8Layout"
                and not getattr(getattr(w, "_params", None), "transposed", False)):
            fused.append(m)
    return fused


def apply_adapters(new_model, converted_by_name, strength, mode, verbose=False):
    """converted_by_name: {adapter_name: {comfy_module: (A, B, scale)}}. Bypass is the
    sharp default; merge is the low-VRAM/quantized-friendly path. `strength` is a
    float applied to every adapter, or {adapter_name: float} for per-adapter
    control (missing names default to 1.0). Returns a report dict."""
    per_name = strength if isinstance(strength, dict) else None
    dm = new_model.get_model_object("diffusion_model")
    pruned = _is_pruned_base(dm)
    report = {}
    all_hooks = []
    for name, converted in converted_by_name.items():
        s = per_name.get(name, 1.0) if per_name is not None else strength
        modules = sorted(converted.keys())
        lora = {}
        for path, (a, b, scale) in converted.items():
            lora[path + ".lora_A.weight"] = a.contiguous()
            lora[path + ".lora_B.weight"] = b.contiguous()
            lora[path + ".alpha"] = torch.tensor(scale * a.shape[0])
        key_map = {m: f"diffusion_model.{m}.weight" for m in modules}
        loaded = comfy.lora.load_lora(lora, key_map, log_missing=False)
        sd_keys = set(new_model.model.state_dict().keys())

        if mode == "merge":
            if pruned:
                adaln_keys = {key_map[m] for m in modules if _is_adaln(m)}
                if adaln_keys:
                    loaded = {k: v for k, v in loaded.items()
                              if k not in adaln_keys}
                    _log.warning("[vdn] pruned base: %d adaln deltas cannot merge "
                                 "into a curve base's collapsed adaln; skipping them "
                                 "(lora_mode=bypass with the MiniMax-H3-Turbo node "
                                 "installed keeps them via e-grid re-injection)",
                                 len(adaln_keys))
            n = len(new_model.add_patches(loaded, s))
            report[name] = f"{n} weights merged"
            continue

        backbone = [m for m in modules if not _is_adaln(m)]
        adaln = [m for m in modules if _is_adaln(m)]
        fc2_fused = set(_int8_fused_fc2(dm, backbone))
        bypass_mods = [m for m in backbone if m not in fc2_fused]

        n = 0
        if bypass_mods:
            n += _bypass(new_model, loaded, key_map, bypass_mods, sd_keys, s,
                         all_hooks)
        if fc2_fused:
            n += len(new_model.add_patches(
                {k: v for k, v in loaded.items()
                 if k in {key_map[m] for m in fc2_fused}}, s))
        if adaln:
            if pruned:
                grid_path = os.path.abspath(_TURBO_GRID)
                if not os.path.exists(grid_path):
                    # Curve bases cannot take the adaln deltas through the normal
                    # paths (that is what the per-block '[96768, 8]' reshape errors
                    # were); skipping them is near-visual-neutral per community
                    # testing. One warning instead of 50 ERROR lines.
                    _log.warning("[vdn] pruned base: %d adaln adapters skipped "
                                 "(needs the silu-temb grid from the "
                                 "ComfyUI-MiniMax-H3-Turbo node at %s)",
                                 len(adaln), grid_path)
                    report[name] = (f"{n} adapters ({len(bypass_mods)} bypass, "
                                    f"{len(fc2_fused)} int8-fc2 merged, "
                                    f"{len(adaln)} adaln SKIPPED)")
                    continue
                _inject_adaln_egrid(new_model, dm, lora, adaln, s)
                n += len(adaln)
            else:
                n += _bypass(new_model, loaded, key_map, adaln, sd_keys, s,
                             all_hooks)
        report[name] = (f"{n} adapters ({len(bypass_mods)} bypass, "
                        f"{len(fc2_fused)} int8-fc2 merged, {len(adaln)} adaln)")
    _install_injection(new_model, all_hooks)
    return report


def _is_adaln(module):
    return module.endswith(".adaln_proj.linear")


def _is_pruned_base(dm):
    """Curve/pruned bases collapse adaln_proj.linear to a tiny t-feature input
    (the [96768, 8] weights); the trained weight takes the full silu(t_emb)
    width. The model flag alone missed some pruned checkpoints (issues #3/#5),
    so the weight shape is the reliable tell."""
    if getattr(dm, "use_adaln_curves", False):
        return True
    try:
        w = comfy.utils.get_attr(dm, "blocks.0.adaln_proj.linear.weight")
        return w.dim() == 2 and w.shape[-1] < 64
    except Exception:
        return False


def _bypass(new_model, loaded, key_map, modules, sd_keys, strength, hooks):
    manager = comfy.weight_adapter.BypassInjectionManager()
    n = 0
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
        n += 1
    manager.create_injections(new_model.model)
    hooks.extend(manager.hooks)
    return n


def _same_bound_method(left, right):
    """Identity check that is stable across repeated bound-method attribute reads."""
    if left is right:
        return True
    left_self = getattr(left, "__self__", None)
    right_self = getattr(right, "__self__", None)
    left_func = getattr(left, "__func__", None)
    right_func = getattr(right, "__func__", None)
    return left_self is right_self and left_func is not None and left_func is right_func


def _bypass_hook_owner(forward):
    """Return the Comfy bypass hook owning a bound forward, if there is one."""
    hook_type = getattr(comfy.weight_adapter, "BypassForwardHook", None)
    owner = getattr(forward, "__self__", None)
    if isinstance(hook_type, type) and isinstance(owner, hook_type):
        return owner
    return None


def _inject_hook_stack_safe(hook):
    """Inject VDN below any already-active Comfy bypass-hook chain.

    ModelPatcher injects and ejects distinct injection keys in the same order. If
    two providers both replace ``module.forward``, ordinary forward-order teardown
    can detach an outer hook or resurrect a stale inner one. Keeping VDN innermost
    makes either provider order recoverable, while preserving the additive adapter
    math.
    """
    if getattr(hook, "original_forward", None) is not None:
        return

    module = hook.module
    previous_forward = module.forward
    hook.inject()  # initializes adapter metadata/device placement as Comfy expects

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

    # hook.inject() temporarily made VDN outermost. Restore the pre-existing
    # chain, then splice VDN immediately above its true/base forward instead.
    module.forward = previous_forward
    hook.original_forward = inner_forward
    current.original_forward = hook._bypass_forward


def _eject_hook_stack_safe(hook):
    """Remove a VDN hook even when another Comfy bypass hook currently wraps it."""
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
            raise RuntimeError("VDN found a cyclic Comfy bypass-forward chain during eject")
        seen.add(marker)

        inner_forward = getattr(current, "original_forward", None)
        if _same_bound_method(inner_forward, target):
            current.original_forward = original_forward
            hook.original_forward = None
            return
        current = _bypass_hook_owner(inner_forward)

    # Another provider may already have detached this hook by restoring an older
    # forward. Do not clobber the currently valid chain by resurrecting our stale
    # original; just mark this VDN hook inactive.
    hook.original_forward = None


def _install_injection(new_model, hooks):
    """Install VDN bypass hooks with clone- and cross-provider-safe lifetime rules.

    All VDN hooks live inside one PatcherInjection and are removed in reverse VDN
    order. In addition, VDN is spliced below any already-active standard Comfy
    ``BypassForwardHook`` chain and can remove itself from the middle of such a
    chain. This matters because ModelPatcher currently walks distinct injection
    keys in insertion order for both inject and eject; another runtime-LoRA
    provider can therefore still be active when VDN tears down (or vice versa).

    Every Apply-VDN run clones the patcher but clones share one inner model. The
    live VDN hook set is tracked on that shared model and removed before a fresh
    VDN set is installed, preventing reruns from accumulating adapter deltas.
    """
    if not hooks:
        return
    owner = new_model.model      # shared by every clone of this model

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
        inject=inject_all, eject=eject_all)
    new_model.set_injections("vdn_lora", [injection])


# ------------------------------------------------------- pruned-base adaln path --

_EGRID = None


def _egrid():
    global _EGRID
    if _EGRID is None:
        path = os.path.abspath(_TURBO_GRID)
        if not os.path.exists(path):
            raise RuntimeError(
                "This VDN adapter updates the adaln projections, but the loaded base "
                "is a pruned (curve) MiniMax-H3 whose adaln weights were collapsed. "
                "Re-injection needs the silu(t_emb) grid bundled with the "
                "ComfyUI-MiniMax-H3-Turbo node; expected it at: " + path)
        _EGRID = comfy.utils.load_torch_file(path)["silu_t_emb_grid"]
    return _EGRID


def _interp_egrid(unique_t, e, device, dtype):
    e = e.to(device)
    n = e.shape[0]
    rows = []
    for t in unique_t:
        pos = min(max(t, 0.0), 1.0) * (n - 1)
        i0 = min(int(math.floor(pos)), n - 2)
        rows.append(torch.lerp(e[i0].float(), e[i0 + 1].float(), pos - i0))
    return torch.stack(rows).to(dtype)


def _unique_t(timestep, shift_v, shift_a, payload):
    """Mirror of the model's unique-timestep row computation (ported from the
    MiniMax-H3-Turbo node)."""
    sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    t_v = float(1.0 - sigma_v)
    t_a = float(1.0 - comfy.ldm.minimax.model.time_shift_sigma(sigma_v, shift_v,
                                                              shift_a))
    s = {t_v, t_a}
    refs = payload.get("refs") or ()
    if payload.get("keyframes") or any(r.get("kind") == "image" for r in refs):
        s.add(max(t_v, float(payload.get("visual_cond_noise_aug", 0.999))))
    if any(r.get("kind") == "audio" and r.get("ref_audio_t", 0) > 0 for r in refs):
        s.add(max(t_a, float(payload.get("audio_cond_noise_aug", 1.0))))
    return sorted(s)


def _make_adaln_forward(base, a, b, shared, table=None, egrid=None):
    """Curve-mode adaln injection as a forward-attribute patch (ported from the
    MiniMax-H3-Turbo node, which documents why the module tree must stay untouched)."""

    def forward(t_emb):
        x = base.linear(F.silu(t_emb) if base.apply_silu else t_emb)
        st = None
        if table is not None and egrid is not None and not base.apply_silu:
            try:
                tb = table.to(t_emb.device, torch.float32)
                idx = torch.cdist(t_emb.detach().float(), tb).argmin(dim=1)
                st = egrid.to(t_emb.device)[idx]
            except Exception:
                st = None
        if st is None:
            st = shared.get("silu_temb")
        if st is not None and st.shape[0] == x.shape[0]:
            av = a.to(x.device, x.dtype)
            bv = b.to(x.device, x.dtype)
            sv = st.to(x.device, x.dtype)
            x = x + (bv @ (av @ sv.T)).T
        x = x.view(x.shape[0] * base.modalities, base.expand * base.hidden)
        return x.chunk(base.expand, dim=-1)

    return forward


def _inject_adaln_egrid(new_model, dm, lora, adaln_modules, strength):
    e = _egrid()
    shared = {"silu_temb": None}
    shift_v = float(getattr(dm, "sigma_shift_video", 12.0))
    shift_a = float(getattr(dm, "sigma_shift_audio", 3.0))

    tt = None
    for _n, _t in list(dm.named_buffers()) + list(dm.named_parameters()):
        if _n.endswith("adaln_t_table"):
            tt = _t
            break
    if tt is not None and tt.shape[0] != e.shape[0]:
        tt = None

    def wrap(executor, *args, **kwargs):
        ts = args[1] if len(args) > 1 else kwargs.get("timestep")
        ctx = args[2] if len(args) > 2 else kwargs.get("context")
        payload = kwargs.get("minimax_payload") or {}
        shared["silu_temb"] = _interp_egrid(
            _unique_t(ts, shift_v, shift_a, payload), e, ctx.device, ctx.dtype)
        return executor(*args, **kwargs)

    new_model.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "vdn_adaln", wrap)
    for name in adaln_modules:
        a = lora[name + ".lora_A.weight"]
        b = lora[name + ".lora_B.weight"] * strength
        key = "diffusion_model." + name.rsplit(".linear", 1)[0]
        new_model.add_object_patch(
            key + ".forward",
            _make_adaln_forward(new_model.get_model_object(key), a, b, shared, tt, e))
