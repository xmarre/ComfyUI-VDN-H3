from __future__ import annotations

import torch
from torch import nn

import comfy.model_patcher

import vdn_h3.managed as managed


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(1), requires_grad=False)
        self.device = torch.device("cpu")


def _base_patcher():
    return comfy.model_patcher.ModelPatcher(
        BaseModel(), torch.device("cpu"), torch.device("cpu"))


def _branches():
    return [
        {
            "alpha.A_log": torch.arange(3, dtype=torch.float32),
            "to_out_linear.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        },
        {
            "alpha.A_log": torch.arange(3, dtype=torch.float32) + 10,
            "to_out_linear.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4) + 20,
        },
    ]


def test_managed_branch_is_a_real_additional_model_patcher():
    base = _base_patcher()
    weights, patcher = managed.make_managed_branch_patcher(_branches(), base)

    assert isinstance(patcher, comfy.model_patcher.ModelPatcher)
    assert patcher.model is weights
    assert patcher.load_device == base.load_device
    assert patcher.offload_device == base.offload_device
    assert patcher.model_size() == sum(p.numel() * p.element_size() for p in weights.parameters())


def test_managed_weights_preserve_names_values_and_casts():
    base = _base_patcher()
    weights, _ = managed.make_managed_branch_patcher(_branches(), base)

    first = weights.weights_on(0, torch.device("cpu"), torch.float32)
    assert set(first) == {"alpha.A_log", "to_out_linear.weight"}
    assert torch.equal(first["alpha.A_log"], torch.arange(3, dtype=torch.float32))
    assert torch.equal(
        first["to_out_linear.weight"],
        torch.arange(12, dtype=torch.float32).reshape(3, 4),
    )
    half = weights.weights_on(0, torch.device("cpu"), torch.float16)
    assert half["alpha.A_log"].dtype == torch.float16
    assert half["to_out_linear.weight"].dtype == torch.float16


def test_additional_model_clone_keeps_state_reference_on_same_parameter_tree():
    base = _base_patcher()
    weights, branch_patcher = managed.make_managed_branch_patcher(_branches(), base)
    applied = base.clone()
    applied.set_additional_models("vdn_branch", [branch_patcher])

    clone = applied.clone()
    cloned_branch = clone.get_additional_models_with_key("vdn_branch")[0]

    # ModelPatcher.clone() clones patcher bookkeeping but deliberately shares the
    # underlying nn.Module for normal (non-dynamic) patchers. VDNState captures
    # `weights`, so this identity is the key resident-mode lifetime invariant.
    assert cloned_branch is not branch_patcher
    assert cloned_branch.model is weights
    original_params = list(weights.parameters())
    cloned_params = list(cloned_branch.model.parameters())
    assert [id(p) for p in cloned_params] == [id(p) for p in original_params]


def test_managed_patcher_repeated_load_unload_preserves_values():
    base = _base_patcher()
    weights, patcher = managed.make_managed_branch_patcher(_branches(), base)
    reference = {
        key: value.clone()
        for key, value in weights.weights_on(0, torch.device("cpu"), torch.float32).items()
    }

    for _ in range(4):
        patcher.patch_model(device_to=torch.device("cpu"))
        loaded = weights.weights_on(0, torch.device("cpu"), torch.float32)
        for key in reference:
            assert torch.equal(loaded[key], reference[key])
        patcher.unpatch_model(device_to=torch.device("cpu"))
        restored = weights.weights_on(0, torch.device("cpu"), torch.float32)
        for key in reference:
            assert torch.equal(restored[key], reference[key])


def test_resident_quantized_branch_fails_closed(monkeypatch):
    class QuantizedTensor(torch.Tensor):
        pass

    fake = torch.zeros(2).as_subclass(QuantizedTensor)
    monkeypatch.setattr(
        managed,
        "resolve_branch_weights",
        lambda weights, device, dtype=None: {"q": fake},
    )
    try:
        managed.BranchBlockWeights({"ignored": torch.ones(1)})
    except RuntimeError as exc:
        assert "resident mode" in str(exc)
        assert "quantized" in str(exc).lower()
    else:
        raise AssertionError("resident mode silently materialized a quantized branch")
