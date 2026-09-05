from __future__ import annotations

import torch
from torch import nn

import comfy.model_management
import comfy.model_patcher

from vdn_h3.apply import apply_adapters


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8, bias=False)
        self.use_adaln_curves = False


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = Diffusion()
        self.device = torch.device("cpu")


def _base_patcher():
    torch.manual_seed(10)
    return comfy.model_patcher.ModelPatcher(
        ToyModel(), torch.device("cpu"), torch.device("cpu")
    )


def _converted(seed=20):
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(3, 8, generator=gen)
    b = torch.randn(8, 3, generator=gen)
    return {"default": {"linear": (a, b, 1.0)}}, a, b


def test_bypass_apply_uses_injection_and_never_weight_wrappers(monkeypatch):
    monkeypatch.setattr(
        comfy.model_management, "get_torch_device", lambda: torch.device("cpu")
    )
    base = _base_patcher()
    module = base.model.diffusion_model.linear
    true_forward = module.forward
    converted, a, b = _converted()

    vdn = base.clone()
    report = apply_adapters(
        vdn, converted, 0.75, mode="bypass", stage_path=None
    )

    assert "vdn_lora" in vdn.injections
    assert vdn.weight_wrapper_patches == {}
    assert report["default"]["runtime_bypass_targets"] == 1
    assert report["default"]["runtime_weight_targets"] == 1
    runtime = report["runtime_lowvram"]
    assert runtime["mode"] == "stack_safe_bypass"
    assert runtime["forward_hooks"] == 1
    assert runtime["weight_wrappers"] == 0
    assert runtime["bias_wrappers"] == 0
    assert runtime["managed_adapter_bytes"] == 0
    assert runtime["owner_key"] is None
    assert runtime["stack_safe_cross_provider"] is True

    x = torch.randn(4, 8)
    base_out = true_forward(x)
    want = base_out + 0.75 * torch.nn.functional.linear(
        torch.nn.functional.linear(x, a), b
    )

    injection = vdn.injections["vdn_lora"][0]
    injection.inject(vdn)
    try:
        got = module(x)
        assert torch.allclose(got, want, atol=1e-5, rtol=1e-5)
    finally:
        injection.eject(vdn)

    assert module.forward == true_forward


def test_merge_apply_stays_on_normal_weight_patches():
    base = _base_patcher()
    converted, _, _ = _converted()
    merged = base.clone()

    report = apply_adapters(
        merged, converted, 1.0, mode="merge", stage_path=None
    )

    assert merged.injections == {}
    assert merged.weight_wrapper_patches == {}
    assert "diffusion_model.linear.weight" in merged.patches
    assert report["default"]["native_weight_patches"] == 1
    assert report["default"]["runtime_bypass_targets"] == 0


def test_bypass_no_longer_requires_weight_function_capability(monkeypatch):
    monkeypatch.setattr(
        comfy.model_management, "get_torch_device", lambda: torch.device("cpu")
    )
    # Plain nn.Linear deliberately has no Comfy weight_function list. v1.5.0
    # rejected this class because its bypass path depended on add_weight_wrapper;
    # v1.5.1 must use the activation-side Comfy bypass contract instead.
    base = _base_patcher()
    converted, _, _ = _converted()
    vdn = base.clone()
    apply_adapters(vdn, converted, 1.0, mode="bypass", stage_path=None)
    assert "vdn_lora" in vdn.injections
    assert vdn.weight_wrapper_patches == {}
