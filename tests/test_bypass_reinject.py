"""Lifecycle regressions for VDN merge and safe runtime-low-VRAM adapters.

The original Continuum crash was a cyclic ``module.forward`` chain. The restored
``lora_mode=bypass`` no longer means BypassForwardHook: it is a Comfy-owned
``weight_function`` wrapper backed by an additional ModelPatcher containing only the
low-rank A/B tensors.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import comfy.model_patcher
import comfy.ops

from vdn_h3.apply import _RuntimeLoRAWeight, apply_adapters
from vdn_h3.managed import RuntimeLoRATermsModel


def _comfy_linear(dim=8):
    layer = comfy.ops.disable_weight_init.Linear(
        dim, dim, bias=False, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(dim, dim))
    return layer


class Diffusion(nn.Module):
    def __init__(self, runtime_capable=False):
        super().__init__()
        self.linear = _comfy_linear() if runtime_capable else nn.Linear(8, 8, bias=False)
        self.use_adaln_curves = False


class ToyModel(nn.Module):
    def __init__(self, runtime_capable=False):
        super().__init__()
        self.diffusion_model = Diffusion(runtime_capable=runtime_capable)
        self.device = torch.device("cpu")


class ConditioningDiffusion(nn.Module):
    def __init__(self, runtime_capable=False):
        super().__init__()
        self.token_refiner = nn.Module()
        self.token_refiner.fc1 = (
            _comfy_linear() if runtime_capable else nn.Linear(8, 8, bias=False))
        self.transformer_eval = (
            _comfy_linear() if runtime_capable else nn.Linear(8, 8, bias=False))
        self.use_adaln_curves = False

    def preprocess_text_embeds(self, x):
        return torch.nn.functional.silu(self.token_refiner.fc1(x))

    def forward(self, x):
        return self.transformer_eval(x)


class ConditioningToyModel(nn.Module):
    def __init__(self, runtime_capable=False):
        super().__init__()
        self.diffusion_model = ConditioningDiffusion(runtime_capable=runtime_capable)
        self.device = torch.device("cpu")


def _base_patcher(runtime_capable=False):
    torch.manual_seed(10)
    return comfy.model_patcher.ModelPatcher(
        ToyModel(runtime_capable=runtime_capable),
        torch.device("cpu"), torch.device("cpu"))


def _conditioning_base_patcher(runtime_capable=False):
    torch.manual_seed(11)
    return comfy.model_patcher.ModelPatcher(
        ConditioningToyModel(runtime_capable=runtime_capable),
        torch.device("cpu"), torch.device("cpu"))


def _converted(seed=20):
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(3, 8, generator=gen)
    b = torch.randn(8, 3, generator=gen)
    return {"default": {"linear": (a, b, 1.0)}}


def _conditioning_converted(seed=21):
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(3, 8, generator=gen)
    b = torch.randn(8, 3, generator=gen)
    return {"default": {"token_refiner.fc1": (a, b, 1.0)}}


def _patched_from(base, strength=1.0, seed=20, mode="merge"):
    patcher = base.clone()
    report = apply_adapters(
        patcher, _converted(seed), strength, mode=mode, stage_path=None)
    if mode == "merge":
        assert report["default"]["native_weight_patches"] == 1
    else:
        assert report["default"]["runtime_weight_targets"] == 1
        runtime = report["runtime_lowvram"]
        assert runtime["weight_wrappers"] == 1
        assert runtime["forward_hooks"] == 0
        assert runtime["managed_adapter_bytes"] > 0
        assert runtime["delta_buffer_limit_bytes"] > 0
    return patcher


def _direct_runtime_wrapper(terms):
    source = RuntimeLoRATermsModel(
        {"linear": terms}, torch.device("cpu"))
    return _RuntimeLoRAWeight(
        "diffusion_model.linear.weight", "linear", source)


def test_runtime_wrapper_matches_explicit_delta_without_mutating_input():
    torch.manual_seed(1)
    weight = torch.randn(8, 8)
    original = weight.clone()
    a1 = torch.randn(3, 8)
    b1 = torch.randn(8, 3)
    a2 = torch.randn(2, 8)
    b2 = torch.randn(8, 2)
    wrapper = _direct_runtime_wrapper(
        [(a1, b1, 0.75), (a2, b2, -0.2)])

    got = wrapper(weight)
    expected = weight + 0.75 * (b1 @ a1) - 0.2 * (b2 @ a2)
    assert torch.allclose(got, expected, atol=2e-6, rtol=2e-6)
    assert torch.equal(weight, original)
    assert not hasattr(wrapper, "prepared_patches")
    assert not hasattr(wrapper, "original_forward")


def test_runtime_mode_registers_weight_wrapper_not_injection_or_forward_patch():
    base = _base_patcher(runtime_capable=True)
    module = base.model.diffusion_model.linear
    original_forward = module.forward.__func__
    vdn = _patched_from(base, mode="bypass")

    assert not vdn.injections
    assert not any(key.endswith("linear.forward") for key in vdn.object_patches)
    wrappers = vdn.weight_wrapper_patches["diffusion_model.linear.weight"]
    assert len(wrappers) == 1
    assert wrappers[0]._vdn_runtime_lora is True
    assert wrappers[0].source.term_count() == 1
    assert "vdn_runtime_lora" in vdn.additional_models
    assert module.forward.__func__ is original_forward


def test_runtime_mode_fails_closed_without_weight_function_contract():
    base = _base_patcher(runtime_capable=False)
    clone = base.clone()
    try:
        apply_adapters(clone, _converted(), 1.0, mode="bypass", stage_path=None)
    except RuntimeError as exc:
        assert "weight_function" in str(exc)
        assert "merge" in str(exc)
    else:
        raise AssertionError("runtime mode accepted an unsupported plain nn.Linear")


def test_native_patch_does_not_replace_forward():
    base = _base_patcher()
    module = base.model.diffusion_model.linear
    true_func = module.forward.__func__
    _patched_from(base)
    assert module.forward.__func__ is true_func


def test_repeated_merge_clone_load_unload_is_stable_and_restores_base():
    base = _base_patcher()
    module = base.model.diffusion_model.linear
    original = module.weight.detach().clone()
    x = torch.randn(4, 8)
    reference = None

    vdn = _patched_from(base, strength=1.0)
    for chunk in range(8):
        clone = vdn.clone()
        clone.patch_model(device_to=torch.device("cpu"))
        got = module(x).detach().clone()
        if reference is None:
            reference = got
        else:
            assert torch.allclose(got, reference, atol=1e-6, rtol=1e-6), chunk
        clone.unpatch_model(device_to=torch.device("cpu"))
        assert torch.equal(module.weight, original), chunk


def test_repeated_runtime_clone_load_unload_is_stable_and_never_merges_base():
    base = _base_patcher(runtime_capable=True)
    module = base.model.diffusion_model.linear
    original = module.weight.detach().clone()
    original_forward = module.forward.__func__
    x = torch.randn(4, 8)
    reference = None

    vdn = _patched_from(base, strength=1.0, mode="bypass")
    for chunk in range(12):
        clone = vdn.clone()
        clone.patch_model(device_to=torch.device("cpu"))
        assert torch.equal(module.weight, original), chunk
        assert module.forward.__func__ is original_forward
        got = module(x).detach().clone()
        if reference is None:
            reference = got
        else:
            assert torch.allclose(got, reference, atol=1e-6, rtol=1e-6), chunk
        clone.unpatch_model(device_to=torch.device("cpu"))
        assert torch.equal(module.weight, original), chunk
        assert module.forward.__func__ is original_forward


def test_continuum_conditioning_then_forward_survives_runtime_mode_repeated_clones():
    base = _conditioning_base_patcher(runtime_capable=True)
    dm = base.model.diffusion_model
    original_fc1 = dm.token_refiner.fc1.weight.detach().clone()
    original_forward = dm.token_refiner.fc1.forward.__func__
    x = torch.randn(5, 8)

    vdn = base.clone()
    report = apply_adapters(
        vdn, _conditioning_converted(), 1.0, mode="bypass", stage_path=None)
    assert report["default"]["runtime_weight_targets"] == 1
    assert not vdn.injections

    conditioning_reference = None
    forward_reference = None
    for chunk in range(12):
        chunk_model = vdn.clone()
        chunk_model.patch_model(device_to=torch.device("cpu"))

        conditioning = dm.preprocess_text_embeds(x).detach().clone()
        transformed = dm(x).detach().clone()
        assert dm.token_refiner.fc1.forward.__func__ is original_forward
        assert torch.equal(dm.token_refiner.fc1.weight, original_fc1)

        if conditioning_reference is None:
            conditioning_reference = conditioning
            forward_reference = transformed
        else:
            assert torch.allclose(
                conditioning, conditioning_reference, atol=1e-6, rtol=1e-6), chunk
            assert torch.allclose(
                transformed, forward_reference, atol=1e-6, rtol=1e-6), chunk

        chunk_model.unpatch_model(device_to=torch.device("cpu"))
        assert torch.equal(dm.token_refiner.fc1.weight, original_fc1), chunk
        assert dm.token_refiner.fc1.forward.__func__ is original_forward


def test_runtime_strength_change_reapplies_from_true_base_not_previous_delta():
    base = _base_patcher(runtime_capable=True)
    module = base.model.diffusion_model.linear
    original = module.weight.detach().clone()
    x = torch.randn(3, 8)

    one = _patched_from(base, strength=1.0, mode="bypass")
    one.patch_model(device_to=torch.device("cpu"))
    y1 = module(x).detach().clone()
    one.unpatch_model(device_to=torch.device("cpu"))

    half = _patched_from(base, strength=0.5, mode="bypass")
    half.patch_model(device_to=torch.device("cpu"))
    yhalf = module(x).detach().clone()
    half.unpatch_model(device_to=torch.device("cpu"))

    y0 = module(x).detach().clone()
    assert torch.equal(module.weight, original)
    assert torch.allclose(yhalf - y0, 0.5 * (y1 - y0), atol=1e-5, rtol=1e-5)


def test_external_forward_owner_is_never_mutated_by_runtime_lifecycle():
    base = _base_patcher(runtime_capable=True)
    module = base.model.diffusion_model.linear
    true_forward = module.forward
    calls = {"n": 0}

    def external_forward(x):
        calls["n"] += 1
        return true_forward(x)

    module.forward = external_forward
    vdn = _patched_from(base, mode="bypass")
    x = torch.randn(2, 8)
    for _ in range(5):
        clone = vdn.clone()
        assert module.forward is external_forward
        clone.patch_model(device_to=torch.device("cpu"))
        module(x)
        assert module.forward is external_forward
        clone.unpatch_model(device_to=torch.device("cpu"))
        assert module.forward is external_forward
    assert calls["n"] == 5


def test_runtime_mode_aggregates_multiple_adapters_into_one_weight_wrapper():
    base = _base_patcher(runtime_capable=True)
    gen = torch.Generator().manual_seed(30)
    a1, b1 = torch.randn(2, 8, generator=gen), torch.randn(8, 2, generator=gen)
    a2, b2 = torch.randn(3, 8, generator=gen), torch.randn(8, 3, generator=gen)
    converted = {
        "default": {"linear": (a1, b1, 1.0)},
        "turbo": {"linear": (a2, b2, 0.5)},
    }
    patcher = base.clone()
    apply_adapters(
        patcher, converted, {"default": 0.75, "turbo": 1.2},
        mode="bypass", stage_path=None)

    wrappers = patcher.weight_wrapper_patches["diffusion_model.linear.weight"]
    assert len(wrappers) == 1
    assert wrappers[0].source.term_count() == 2

    weight = base.model.diffusion_model.linear.weight.detach()
    expected_weight = weight + 0.75 * (b1 @ a1) + (1.2 * 0.5) * (b2 @ a2)
    got_weight = wrappers[0](weight)
    assert torch.allclose(got_weight, expected_weight, atol=2e-6, rtol=2e-6)
