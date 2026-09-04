from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

import comfy.model_patcher

from vdn_h3.apply import apply_adapters


class QuantizedLikeLinear(nn.Module):
    """Synthetic Comfy custom-weight module exercising convert_weight/set_weight.

    Real comfy-kitchen quantized linears expose these same ModelPatcher hooks. The
    test deliberately does not pretend to reproduce a specific quantizer; it proves
    VDN registers the LoRA at the weight-patching abstraction that invokes the custom
    conversion/requantization contract instead of relying on module.forward.
    """

    def __init__(self, dim=8):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, dim), requires_grad=False)
        self.convert_calls = 0
        self.set_calls = 0

    def convert_weight(self, weight, inplace=False, **kwargs):
        self.convert_calls += 1
        # Stand-in for dequantization into the compute representation.
        return weight.float()

    def set_weight(self, weight, inplace_update=False, seed=None,
                   return_weight=False, **kwargs):
        self.set_calls += 1
        if return_weight:
            return weight
        # Stand-in for a quantizer's requantize_from_float + replacement.
        self.weight = nn.Parameter(weight.detach().clone(), requires_grad=False)
        return self.weight

    def forward(self, x):
        return F.linear(x, self.weight)


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


def test_native_adapter_uses_custom_weight_convert_and_set_contract():
    torch.manual_seed(71)
    model = ToyModel()
    module = model.diffusion_model.fc2
    base_weight = module.weight.detach().clone()
    base_forward = module.forward.__func__
    patcher = comfy.model_patcher.ModelPatcher(
        model, torch.device("cpu"), torch.device("cpu"))

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
    model = ToyModel()
    module = model.diffusion_model.fc2
    base_weight = module.weight.detach().clone()
    patcher = comfy.model_patcher.ModelPatcher(
        model, torch.device("cpu"), torch.device("cpu"))

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
