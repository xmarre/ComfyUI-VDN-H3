"""Regression tests for VDN bypass-hook reinjection and cross-provider teardown.

ComfyUI's ModelPatcher walks distinct injection keys in insertion order for both
inject and eject. Plain BypassForwardHook teardown is only safe when wrappers are
removed in strict global LIFO order, so two independent runtime-adapter providers
can otherwise detach/resurrect stale forwards and eventually recurse forever.

VDN's injection keeps its own adapter hooks in LIFO order, inserts VDN below an
already-active standard Comfy bypass chain, and splices VDN out safely when another
provider still wraps it. Run from anywhere with the ComfyUI venv python.
"""
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

_COMFYUI_ROOT = Path(__file__).resolve().parents[3]  # the ComfyUI checkout
_PACKAGE = Path(__file__).resolve().parents[1]       # this package, any folder name
sys.path.insert(0, str(_COMFYUI_ROOT))
sys.path.insert(0, str(_PACKAGE))

import comfy.model_management
import comfy.patcher_extension
import comfy.weight_adapter
from comfy.weight_adapter.bypass import BypassForwardHook
from vdn_h3.apply import _FrugalLoRA, _install_injection

comfy.model_management.get_torch_device = lambda: torch.device("cpu")


def _adapter():
    up = torch.randn(8, 4)
    down = torch.randn(4, 8) * 0.1
    return _FrugalLoRA(set(), (up, down, torch.tensor(4.0)))  # alpha/rank = 1


class _Patcher:
    def __init__(self, model=None):
        self.injections = {}
        self.model = model if model is not None else types.SimpleNamespace()

    def set_injections(self, key, value):
        self.injections[key] = value

    def inject_model(self):
        # Mirrors ModelPatcher.inject_model's insertion-order traversal.
        for injections in self.injections.values():
            for injection in injections:
                injection.inject(self)

    def eject_model(self):
        # Mirrors ModelPatcher.eject_model's insertion-order traversal.
        for injections in self.injections.values():
            for injection in injections:
                injection.eject(self)


def _fresh_module():
    torch.manual_seed(0)
    mod = nn.Linear(8, 8)
    return mod, mod.forward


def _legacy_cycle_breaks():
    """Documents the original same-provider bug: forward-order eject leaves a
    stale hook in place, so re-injecting makes the hook its own original forward."""
    mod, true_fwd = _fresh_module()
    hook_d = BypassForwardHook(mod, _adapter(), multiplier=1.0)
    hook_t = BypassForwardHook(mod, _adapter(), multiplier=1.0)
    hook_d.inject()
    hook_t.inject()                     # hook_t.original = hook_d._bypass_forward
    hook_d.eject()                      # mod.forward = true forward
    hook_t.eject()                      # mod.forward = hook_d._bypass_forward (!)
    hook_d.inject()                     # hook_d.original = its own bypass forward
    broken = hook_d.original_forward == hook_d._bypass_forward
    mod.forward = true_fwd              # restore for cleanliness
    return broken


def _adapter_lora(hook, x):
    up, down, _ = hook.adapter.weights
    return torch.nn.functional.linear(
        torch.nn.functional.linear(x, down), up)


def _external_injection(hook):
    """One independent provider using ComfyUI's ordinary bypass-hook lifetime."""
    return comfy.patcher_extension.PatcherInjection(
        inject=lambda _patcher: hook.inject(),
        eject=lambda _patcher: hook.eject(),
    )


def test_lifo_cycles():
    mod, true_fwd = _fresh_module()
    hook_d = BypassForwardHook(mod, _adapter(), multiplier=1.0)
    hook_t = BypassForwardHook(mod, _adapter(), multiplier=1.0)
    patcher = _Patcher()
    _install_injection(patcher, [hook_d, hook_t])
    injection = patcher.injections["vdn_lora"][0]

    x = torch.randn(3, 8)
    base = true_fwd(x)
    want = base + _adapter_lora(hook_d, x) + _adapter_lora(hook_t, x)

    for cycle in range(3):
        injection.inject(patcher)
        assert mod.forward == hook_d._bypass_forward, f"cycle {cycle}: wrong outermost hook"
        assert hook_d.original_forward == hook_t._bypass_forward, f"cycle {cycle}: D should wrap T"
        assert hook_t.original_forward == true_fwd, f"cycle {cycle}: T should wrap the true forward"
        got = mod(x)
        assert torch.allclose(got, want, atol=1e-5), f"cycle {cycle}: wrong value"
        injection.eject(patcher)
        assert mod.forward == true_fwd, f"cycle {cycle}: true forward not restored"
        assert hook_d.original_forward is None and hook_t.original_forward is None


def _run_cross_provider_cycles(vdn_first):
    mod, true_fwd = _fresh_module()
    vdn_hook = BypassForwardHook(mod, _adapter(), multiplier=1.0)
    external_hook = BypassForwardHook(mod, _adapter(), multiplier=1.0)
    patcher = _Patcher()

    if vdn_first:
        _install_injection(patcher, [vdn_hook])
        patcher.set_injections("external_runtime_lora", [_external_injection(external_hook)])
    else:
        patcher.set_injections("external_runtime_lora", [_external_injection(external_hook)])
        _install_injection(patcher, [vdn_hook])

    x = torch.randn(3, 8)
    want = true_fwd(x) + _adapter_lora(vdn_hook, x) + _adapter_lora(external_hook, x)

    for cycle in range(3):
        patcher.inject_model()
        # VDN deliberately stays inside the external Comfy bypass hook regardless
        # of which injection key was inserted first.
        assert mod.forward == external_hook._bypass_forward
        assert external_hook.original_forward == vdn_hook._bypass_forward
        assert vdn_hook.original_forward == true_fwd
        assert torch.allclose(mod(x), want, atol=1e-5)

        # This is the exact hazardous behavior in core today: providers are ejected
        # in the same order they were injected, not global LIFO order.
        patcher.eject_model()
        assert mod.forward == true_fwd
        assert vdn_hook.original_forward is None
        assert external_hook.original_forward is None


def test_cross_provider_forward_order_eject_vdn_first():
    _run_cross_provider_cycles(vdn_first=True)


def test_cross_provider_forward_order_eject_external_first():
    _run_cross_provider_cycles(vdn_first=False)


def test_live_vdn_replacement_under_external_hook():
    """A fresh Apply-VDN result may replace hooks on the clone-shared inner model
    while another runtime-adapter provider is still active. The old VDN hook must
    be removed from the middle instead of detaching the external wrapper."""
    mod, true_fwd = _fresh_module()
    shared_owner = types.SimpleNamespace()
    first = BypassForwardHook(mod, _adapter(), multiplier=1.0)
    second = BypassForwardHook(mod, _adapter(), multiplier=1.0)
    external = BypassForwardHook(mod, _adapter(), multiplier=1.0)

    patcher1 = _Patcher(shared_owner)
    _install_injection(patcher1, [first])
    inj1 = patcher1.injections["vdn_lora"][0]
    inj1.inject(patcher1)
    external.inject()

    assert mod.forward == external._bypass_forward
    assert external.original_forward == first._bypass_forward

    patcher2 = _Patcher(shared_owner)
    _install_injection(patcher2, [second])
    inj2 = patcher2.injections["vdn_lora"][0]
    inj2.inject(patcher2)

    assert first.original_forward is None
    assert mod.forward == external._bypass_forward
    assert external.original_forward == second._bypass_forward
    assert second.original_forward == true_fwd

    x = torch.randn(2, 8)
    want = true_fwd(x) + _adapter_lora(second, x) + _adapter_lora(external, x)
    assert torch.allclose(mod(x), want, atol=1e-5)

    inj2.eject(patcher2)
    assert mod.forward == external._bypass_forward
    assert external.original_forward == true_fwd
    external.eject()
    assert mod.forward == true_fwd


if __name__ == "__main__":
    assert _legacy_cycle_breaks(), "legacy cycle did not reproduce the bug"
    print("legacy cycle: reproduces the self-hook bug as expected")
    test_lifo_cycles()
    test_cross_provider_forward_order_eject_vdn_first()
    test_cross_provider_forward_order_eject_external_first()
    test_live_vdn_replacement_under_external_hook()
    print("ALL PASS")
