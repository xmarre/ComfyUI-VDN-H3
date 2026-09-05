"""Regression tests for VDN bypass-hook reinjection and cross-provider teardown.

ComfyUI's ModelPatcher can have several independently owned bypass injections on
one module. Plain BypassForwardHook teardown is only safe under a global LIFO
order, so VDN keeps its hooks inside the external chain and splices them out
safely regardless of provider insertion/ejection order.

These are the lifecycle invariants from the previously production-validated PR #1
path. v1.5.1 restores this architecture after the v1.5.0 weight-wrapper path
regressed a stacked VDN + runtime-DoRA quantized workflow.
"""
from __future__ import annotations

import types

import torch
import torch.nn as nn

import comfy.model_management
import comfy.patcher_extension
from comfy.weight_adapter.bypass import BypassForwardHook

from vdn_h3.apply import _FrugalLoRA, _install_injection


def _adapter():
    up = torch.randn(8, 4)
    down = torch.randn(4, 8) * 0.1
    return _FrugalLoRA(set(), (up, down, torch.tensor(4.0)))


class _Patcher:
    def __init__(self, model=None):
        self.injections = {}
        self.model = model if model is not None else types.SimpleNamespace()

    def set_injections(self, key, value):
        self.injections[key] = value

    def inject_model(self):
        for injections in self.injections.values():
            for injection in injections:
                injection.inject(self)

    def eject_model(self):
        # Deliberately mirrors the hazardous same-order provider teardown that the
        # VDN splicing logic must survive.
        for injections in self.injections.values():
            for injection in injections:
                injection.eject(self)


def _fresh_module():
    torch.manual_seed(0)
    mod = nn.Linear(8, 8)
    return mod, mod.forward


def _adapter_lora(hook, x):
    up, down, _ = hook.adapter.weights
    return torch.nn.functional.linear(
        torch.nn.functional.linear(x, down), up
    )


def _external_injection(hook):
    return comfy.patcher_extension.PatcherInjection(
        inject=lambda _patcher: hook.inject(),
        eject=lambda _patcher: hook.eject(),
    )


def test_vdn_multiple_hooks_repeated_cycles_restore_true_forward(monkeypatch):
    monkeypatch.setattr(
        comfy.model_management, "get_torch_device", lambda: torch.device("cpu")
    )
    mod, true_fwd = _fresh_module()
    hook_d = BypassForwardHook(mod, _adapter(), multiplier=1.0)
    hook_t = BypassForwardHook(mod, _adapter(), multiplier=1.0)
    patcher = _Patcher()
    _install_injection(patcher, [hook_d, hook_t])
    injection = patcher.injections["vdn_lora"][0]

    x = torch.randn(3, 8)
    want = true_fwd(x) + _adapter_lora(hook_d, x) + _adapter_lora(hook_t, x)

    for cycle in range(5):
        injection.inject(patcher)
        assert mod.forward == hook_d._bypass_forward, cycle
        assert hook_d.original_forward == hook_t._bypass_forward, cycle
        assert hook_t.original_forward == true_fwd, cycle
        assert torch.allclose(mod(x), want, atol=1e-5), cycle

        injection.eject(patcher)
        assert mod.forward == true_fwd, cycle
        assert hook_d.original_forward is None
        assert hook_t.original_forward is None


def _run_cross_provider_cycles(monkeypatch, vdn_first):
    monkeypatch.setattr(
        comfy.model_management, "get_torch_device", lambda: torch.device("cpu")
    )
    mod, true_fwd = _fresh_module()
    vdn_hook = BypassForwardHook(mod, _adapter(), multiplier=1.0)
    external_hook = BypassForwardHook(mod, _adapter(), multiplier=1.0)
    patcher = _Patcher()

    if vdn_first:
        _install_injection(patcher, [vdn_hook])
        patcher.set_injections(
            "external_runtime_lora", [_external_injection(external_hook)]
        )
    else:
        patcher.set_injections(
            "external_runtime_lora", [_external_injection(external_hook)]
        )
        _install_injection(patcher, [vdn_hook])

    x = torch.randn(3, 8)
    want = (
        true_fwd(x)
        + _adapter_lora(vdn_hook, x)
        + _adapter_lora(external_hook, x)
    )

    for cycle in range(5):
        patcher.inject_model()

        # External provider is outermost in both insertion orders; VDN is safely
        # spliced underneath it and above the true base forward.
        assert mod.forward == external_hook._bypass_forward, cycle
        assert external_hook.original_forward == vdn_hook._bypass_forward, cycle
        assert vdn_hook.original_forward == true_fwd, cycle
        assert torch.allclose(mod(x), want, atol=1e-5), cycle

        patcher.eject_model()
        assert mod.forward == true_fwd, cycle
        assert vdn_hook.original_forward is None
        assert external_hook.original_forward is None


def test_cross_provider_forward_order_eject_vdn_first(monkeypatch):
    _run_cross_provider_cycles(monkeypatch, vdn_first=True)


def test_cross_provider_forward_order_eject_external_first(monkeypatch):
    _run_cross_provider_cycles(monkeypatch, vdn_first=False)


def test_live_vdn_replacement_under_external_hook(monkeypatch):
    monkeypatch.setattr(
        comfy.model_management, "get_torch_device", lambda: torch.device("cpu")
    )
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


def test_cycle_detection_fails_closed(monkeypatch):
    monkeypatch.setattr(
        comfy.model_management, "get_torch_device", lambda: torch.device("cpu")
    )
    mod, _ = _fresh_module()
    external = BypassForwardHook(mod, _adapter(), multiplier=1.0)
    external.inject()
    external.original_forward = external._bypass_forward

    vdn = BypassForwardHook(mod, _adapter(), multiplier=1.0)
    patcher = _Patcher()
    _install_injection(patcher, [vdn])
    injection = patcher.injections["vdn_lora"][0]

    try:
        injection.inject(patcher)
    except RuntimeError as exc:
        assert "cyclic Comfy bypass-forward chain" in str(exc)
    else:
        raise AssertionError("cyclic bypass chain was accepted")
