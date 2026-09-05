from __future__ import annotations

import torch
from torch import nn

import comfy.model_patcher

from vdn_h3.apply import apply_adapters
from vdn_h3.runtime_introspection import _build_runtime_adapter_registry


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


def _patcher():
    torch.manual_seed(511)
    return comfy.model_patcher.ModelPatcher(
        ToyModel(), torch.device("cpu"), torch.device("cpu")
    )


def _captured_registries(injection):
    found = []
    seen = set()
    for function in (injection.inject, injection.eject):
        for cell in getattr(function, "__closure__", None) or ():
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            adapters = getattr(value, "adapters", None)
            if isinstance(adapters, dict) and id(value) not in seen:
                seen.add(id(value))
                found.append(value)
    return found


def test_bypass_injection_publishes_zero_copy_classic_lora_metadata():
    patcher = _patcher()
    a0 = torch.randn(3, 8)
    b0 = torch.randn(8, 3)
    a1 = torch.randn(2, 8)
    b1 = torch.randn(8, 2)

    apply_adapters(
        patcher,
        {
            "default": {"linear": (a0, b0, 0.5)},
            "turbo": {"linear": (a1, b1, 0.25)},
        },
        {"default": 0.8, "turbo": 0.6},
        mode="bypass",
        stage_path=None,
    )

    injection = patcher.injections["vdn_lora"][0]
    registries = _captured_registries(injection)
    assert len(registries) == 1
    entries = registries[0].adapters
    assert len(entries) == 2
    assert "diffusion_model.linear.weight" in entries
    assert any(key.startswith("diffusion_model.linear.weight#vdn-runtime-") for key in entries)

    factors = []
    for adapter, strength in entries.values():
        assert adapter.name == "lora"
        up, down, alpha, mid, dora_scale, reshape = adapter.weights
        assert alpha == float(down.shape[0])
        assert mid is dora_scale is reshape is None
        factors.append((down, up, strength))

    # Metadata must describe exactly the same sum of low-rank operators without
    # allocating concatenated copies of the factors.
    x = torch.randn(4, 8)
    got = sum(
        torch.nn.functional.linear(torch.nn.functional.linear(x, down), up) * strength
        for down, up, strength in factors
    )
    expected = (
        torch.nn.functional.linear(torch.nn.functional.linear(x, a0), b0) * 0.4
        + torch.nn.functional.linear(torch.nn.functional.linear(x, a1), b1) * 0.15
    )
    assert torch.allclose(got, expected, atol=1e-6, rtol=1e-6)

    referenced = {
        id(value)
        for adapter, _strength in entries.values()
        for value in (adapter.weights[0], adapter.weights[1])
    }
    assert referenced == {id(a0), id(b0), id(a1), id(b1)}


def test_projected_curve_metadata_marks_constant_bias_conservatively_unknown():
    down = torch.randn(4, 8)
    up = torch.randn(16, 4)
    offset = torch.randn(16)
    module = "blocks.49.adaln_proj.linear"

    registry = _build_runtime_adapter_registry(
        {module: [(down, up, 0.75)]},
        {module: [(offset, 0.75)]},
    )
    assert set(registry.adapters) == {
        "diffusion_model.blocks.49.adaln_proj.linear.weight",
        "diffusion_model.blocks.49.adaln_proj.linear.bias",
    }

    weight_adapter, weight_strength = registry.adapters[
        "diffusion_model.blocks.49.adaln_proj.linear.weight"
    ]
    bias_adapter, bias_strength = registry.adapters[
        "diffusion_model.blocks.49.adaln_proj.linear.bias"
    ]
    assert weight_adapter.name == "lora"
    assert weight_strength == 0.75
    assert weight_adapter.weights[0] is up
    assert weight_adapter.weights[1] is down
    assert bias_adapter.name == "vdn_runtime_bias"
    assert bias_strength == 0.75
    assert bias_adapter.weights[0] is offset


def test_introspection_wrapper_does_not_change_runtime_injection_math():
    patcher = _patcher()
    module = patcher.model.diffusion_model.linear
    base_forward = module.forward
    a = torch.randn(3, 8)
    b = torch.randn(8, 3)
    x = torch.randn(5, 8)
    expected = base_forward(x) + torch.nn.functional.linear(
        torch.nn.functional.linear(x, a), b
    ) * 0.7

    apply_adapters(
        patcher,
        {"default": {"linear": (a, b, 1.0)}},
        0.7,
        mode="bypass",
        stage_path=None,
    )
    injection = patcher.injections["vdn_lora"][0]
    injection.inject(patcher)
    try:
        assert module.forward == base_forward
        assert torch.allclose(module(x), expected, atol=1e-6, rtol=1e-6)
    finally:
        injection.eject(patcher)
    assert module.forward == base_forward
