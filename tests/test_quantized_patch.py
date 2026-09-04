from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

import comfy.model_patcher

from vdn_h3.apply import apply_adapters


class QuantizedLikeLinear(nn.Module):
    """Synthetic Comfy custom-weight module exercising both adapter lifecycles.

    ``convert_weight`` / ``set_weight`` stand in for dequantize/requantize in the
    eager merge path. ``weight_function`` is the core-supported runtime path used by
    Comfy castable linears. This is deliberately not a fake quantizer; it proves VDN
    chooses the correct lifecycle abstraction without changing ``module.forward``.
    """

    def __init__(self, dim=8):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, dim), requires_grad=False)
        self.convert_calls = 0
        self.set_calls = 0
        self.comfy_cast_weights = False
        self.weight_function = []
        self.bias_function = []

    def convert_weight(self, weight, inplace=False, **kwargs):
        self.convert_calls += 1
        return weight.float()

    def set_weight(self, weight, inplace_update=False, seed=None,
                   return_weight=False, **kwargs):
        self.set_calls += 1
        if return_weight:
            return weight
        self.weight = nn.Parameter(weight.detach().clone(), requires_grad=False)
        return self.weight

    def forward(self, x):
        weight = self.weight
        for function in self.weight_function:
            weight = function(weight)
        return F.linear(x, weight)


class QuantizedLikeDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc2 = QuantizedLikeLinear()
        self.use_adaln_curves = False


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = QuantizedLikeDiffusion()
        self.device = torch.device("cpu")


def _patcher():
    model = ToyModel()
    return model, comfy.model_patcher.ModelPatcher(
        model, torch.device("cpu"), torch.device("cpu"))


def test_native_adapter_uses_custom_weight_convert_and_set_contract():
    torch.manual_seed(71)
    model, patcher = _patcher()
    module = model.diffusion_model.fc2
    base_weight = module.weight.detach().clone()
    base_forward = module.forward.__func__

    a = torch.randn(3, 8)
    b = torch.randn(8, 3)
    converted = {"default": {"fc2": (a, b, 1.0)}}
    report = apply_adapters(
        patcher, converted, 1.0, mode="merge", stage_path=None)
    assert report["default"]["native_weight_patches"] == 1
    assert module.forward.__func__ is base_forward

    patcher.patch_model(device_to=torch.device("cpu"))
    assert module.convert_calls >= 1
    assert module.set_calls >= 1
    assert module.forward.__func__ is base_forward
    expected = base_weight.float() + b.float() @ a.float()
    assert torch.allclose(module.weight.float(), expected, atol=2e-6, rtol=2e-6)

    x = torch.randn(4, 8)
    got = module(x)
    assert torch.allclose(got, F.linear(x, expected), atol=2e-6, rtol=2e-6)

    patcher.unpatch_model(device_to=torch.device("cpu"))
    assert module.forward.__func__ is base_forward
    assert torch.equal(module.weight, base_weight)


def test_custom_weight_patch_repeated_clone_cycles_do_not_accumulate():
    torch.manual_seed(72)
    model, patcher = _patcher()
    module = model.diffusion_model.fc2
    base_weight = module.weight.detach().clone()

    a = torch.randn(2, 8)
    b = torch.randn(8, 2)
    apply_adapters(
        patcher, {"default": {"fc2": (a, b, 1.0)}},
        1.0, mode="merge", stage_path=None)
    expected = base_weight.float() + b.float() @ a.float()

    for cycle in range(6):
        clone = patcher.clone()
        clone.patch_model(device_to=torch.device("cpu"))
        assert torch.allclose(
            module.weight.float(), expected, atol=2e-6, rtol=2e-6), cycle
        clone.unpatch_model(device_to=torch.device("cpu"))
        assert torch.equal(module.weight, base_weight), cycle


def test_runtime_lowvram_custom_weight_does_not_eager_merge_or_set_weight():
    torch.manual_seed(73)
    model, patcher = _patcher()
    module = model.diffusion_model.fc2
    base_weight = module.weight.detach().clone()
    base_forward = module.forward.__func__
    a = torch.randn(3, 8)
    b = torch.randn(8, 3)

    report = apply_adapters(
        patcher, {"default": {"fc2": (a, b, 1.0)}},
        1.0, mode="bypass", stage_path=None)
    assert report["default"]["native_weight_patches"] == 0
    assert report["default"]["runtime_weight_targets"] == 1
    assert not patcher.patches
    assert not patcher.injections
    assert module.forward.__func__ is base_forward

    patcher.patch_model(device_to=torch.device("cpu"))
    # Runtime low-VRAM mode does not materialize/requantize a patched resident
    # parameter during load. The base representation stays untouched.
    assert module.convert_calls == 0
    assert module.set_calls == 0
    assert torch.equal(module.weight, base_weight)
    assert module.forward.__func__ is base_forward
    assert len(module.weight_function) == 1

    x = torch.randn(4, 8)
    expected_weight = base_weight + b @ a
    got = module(x)
    assert torch.allclose(
        got, F.linear(x, expected_weight), atol=2e-6, rtol=2e-6)
    assert torch.equal(module.weight, base_weight)

    patcher.unpatch_model(device_to=torch.device("cpu"))
    assert torch.equal(module.weight, base_weight)
    assert module.forward.__func__ is base_forward
    assert module.weight_function == []


def test_runtime_lowvram_repeated_custom_weight_clone_cycles_do_not_accumulate():
    torch.manual_seed(74)
    model, patcher = _patcher()
    module = model.diffusion_model.fc2
    base_weight = module.weight.detach().clone()
    a = torch.randn(2, 8)
    b = torch.randn(8, 2)
    x = torch.randn(3, 8)
    expected = F.linear(x, base_weight + b @ a)

    apply_adapters(
        patcher, {"default": {"fc2": (a, b, 1.0)}},
        1.0, mode="bypass", stage_path=None)

    for cycle in range(8):
        clone = patcher.clone()
        clone.patch_model(device_to=torch.device("cpu"))
        assert len(module.weight_function) == 1, cycle
        assert torch.allclose(module(x), expected, atol=2e-6, rtol=2e-6), cycle
        assert torch.equal(module.weight, base_weight), cycle
        clone.unpatch_model(device_to=torch.device("cpu"))
        assert module.weight_function == [], cycle
        assert torch.equal(module.weight, base_weight), cycle
