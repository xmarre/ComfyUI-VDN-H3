"""Lifecycle regression for the adapter architecture that replaced VDN bypass hooks.

The original Continuum crash was a cyclic ``module.forward`` chain.  VDN no longer
installs such chains, so the regression now checks the stronger invariants directly:
weight contribution is stable across clone/load/unload cycles, the base is restored,
option/strength changes do not accumulate, and an unrelated forward owner remains
untouched throughout.
"""
from __future__ import annotations

import torch
import torch.nn as nn

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
        ToyModel(), torch.device("cpu"), torch.device("cpu"))


def _converted(seed=20):
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(3, 8, generator=gen)
    b = torch.randn(8, 3, generator=gen)
    return {"default": {"linear": (a, b, 1.0)}}


def _patched_from(base, strength=1.0, seed=20):
    patcher = base.clone()
    report = apply_adapters(
        patcher, _converted(seed), strength, mode="merge", stage_path=None)
    assert report["default"]["native_weight_patches"] == 1
    return patcher


def test_native_patch_does_not_replace_forward():
    base = _base_patcher()
    module = base.model.diffusion_model.linear
    true_func = module.forward.__func__
    _patched_from(base)
    assert module.forward.__func__ is true_func


def test_repeated_clone_load_unload_is_stable_and_restores_base():
    base = _base_patcher()
    module = base.model.diffusion_model.linear
    original = module.weight.detach().clone()
    x = torch.randn(4, 8)
    reference = None

    vdn = _patched_from(base, strength=1.0)
    for chunk in range(8):
        # Continuum-style sequential MODEL clone reuse over the same resident inner model.
        clone = vdn.clone()
        clone.patch_model(device_to=torch.device("cpu"))
        got = module(x).detach().clone()
        if reference is None:
            reference = got
        else:
            assert torch.allclose(got, reference, atol=1e-6, rtol=1e-6), chunk
        clone.unpatch_model(device_to=torch.device("cpu"))
        assert torch.equal(module.weight, original), chunk


def test_strength_change_reapplies_from_true_base_not_previous_delta():
    base = _base_patcher()
    module = base.model.diffusion_model.linear
    x = torch.randn(3, 8)

    one = _patched_from(base, strength=1.0)
    one.patch_model(device_to=torch.device("cpu"))
    y1 = module(x).detach().clone()
    one.unpatch_model(device_to=torch.device("cpu"))

    half = _patched_from(base, strength=0.5)
    half.patch_model(device_to=torch.device("cpu"))
    yhalf = module(x).detach().clone()
    half.unpatch_model(device_to=torch.device("cpu"))

    y0 = module(x).detach().clone()
    assert torch.allclose(yhalf - y0, 0.5 * (y1 - y0), atol=1e-5, rtol=1e-5)


def test_external_forward_owner_is_never_mutated_by_vdn_lifecycle():
    base = _base_patcher()
    module = base.model.diffusion_model.linear
    true_forward = module.forward

    # Structural stand-in for another provider owning module.forward.  VDN must not
    # traverse, splice, replace or restore this chain at all.
    calls = {"n": 0}
    def external_forward(x):
        calls["n"] += 1
        return true_forward(x)
    module.forward = external_forward

    vdn = _patched_from(base)
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


def test_removed_bypass_fails_explicitly():
    base = _base_patcher()
    clone = base.clone()
    try:
        apply_adapters(clone, _converted(), 1.0, mode="bypass")
    except RuntimeError as exc:
        assert "removed" in str(exc).lower()
        assert "module.forward" in str(exc)
    else:
        raise AssertionError("legacy bypass mode unexpectedly remained active")
