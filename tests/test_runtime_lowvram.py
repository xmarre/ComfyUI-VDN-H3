from __future__ import annotations

import torch
from torch import nn

import comfy.lora
import comfy.model_management
import comfy.model_patcher

from vdn_h3.adapters import convert_adapter
from vdn_h3.apply import (
    _RuntimeLoRAWeight,
    _comfy_lora,
    _validate_runtime_weight_targets,
)
from vdn_h3.managed import (
    RuntimeLoRATermsModel,
    make_managed_runtime_lora_patcher,
)


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(1), requires_grad=False)
        self.device = torch.device("cpu")


def _base_patcher():
    return comfy.model_patcher.ModelPatcher(
        BaseModel(), torch.device("cpu"), torch.device("cpu"))


def test_managed_runtime_terms_preserve_checkpoint_dtype_and_clone_identity():
    a = torch.randn(3, 8, dtype=torch.bfloat16)
    b = torch.randn(8, 3, dtype=torch.bfloat16)
    terms = {"linear": [(a, b, 0.75)]}
    base = _base_patcher()
    model, patcher = make_managed_runtime_lora_patcher(terms, base)

    params = list(model.parameters())
    assert len(params) == 2
    assert all(p.dtype == torch.bfloat16 for p in params)
    assert model.term_count() == 1
    assert patcher.model_size() == sum(p.numel() * p.element_size() for p in params)

    applied = base.clone()
    applied.set_additional_models("vdn_runtime_lora", [patcher])
    clone = applied.clone()
    cloned = clone.get_additional_models_with_key("vdn_runtime_lora")[0]
    assert cloned is not patcher
    assert cloned.model is model
    assert [id(p) for p in cloned.model.parameters()] == [id(p) for p in params]


def test_runtime_weight_math_matches_comfy_merge_adapter_math_float32():
    torch.manual_seed(123)
    weight = torch.randn(8, 8, dtype=torch.float32)
    a = torch.randn(3, 8, dtype=torch.float32)
    b = torch.randn(8, 3, dtype=torch.float32)
    scale = 0.75
    strength = 0.6
    key = "diffusion_model.linear.weight"

    raw = _comfy_lora({"linear": (a, b, scale)})
    loaded = comfy.lora.load_lora(raw, {"linear": key}, log_missing=False)
    adapter = loaded[key]
    compute_dtype = comfy.model_management.lora_compute_dtype(weight.device)
    if compute_dtype is None:
        compute_dtype = weight.dtype
    expected = adapter.calculate_weight(
        weight.to(dtype=compute_dtype, copy=True),
        key,
        strength,
        1.0,
        None,
        lambda x: x,
        intermediate_dtype=compute_dtype,
    )

    source = RuntimeLoRATermsModel(
        {"linear": [(a, b, scale * strength)]}, torch.device("cpu"))
    wrapper = _RuntimeLoRAWeight(key, "linear", source)
    got = wrapper(weight)

    assert torch.equal(got, expected)


def test_runtime_delta_buffer_is_bounded_independently_of_full_weight_size():
    source = RuntimeLoRATermsModel(
        {"linear": [(torch.zeros(2, 8192), torch.zeros(16384, 2), 1.0)]},
        torch.device("cpu"),
    )
    wrapper = _RuntimeLoRAWeight(
        "diffusion_model.linear.weight", "linear", source)
    out = torch.empty(16384, 8192, dtype=torch.bfloat16)
    rows = wrapper._rows_per_chunk(out, 8192)
    assert rows >= 1
    assert rows * 8192 * out.element_size() <= (8 << 20)
    assert rows < out.shape[0]


def test_openvdn_token_refiner_targets_validate_against_comfy_fused_modules():
    rank, hidden, inner = 2, 5, 4

    def marked_linear(in_features, out_features):
        layer = nn.Linear(in_features, out_features, bias=False)
        layer.weight_function = None
        return layer

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv_proj = marked_linear(hidden, inner * 3)
            self.out_proj = marked_linear(inner, hidden)

    class RefinerBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = Attention()

    class TokenRefiner(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([RefinerBlock()])

    class DiffusionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_refiner = TokenRefiner()

    class Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.diffusion_model = DiffusionModel()
            self.device = torch.device("cpu")

    state = {}
    for suffix in ("to_q", "to_k", "to_v"):
        module = f"token_refiner.refiner_blocks.0.attn.{suffix}"
        state[f"{module}.lora_A.weight"] = torch.randn(rank, hidden)
        state[f"{module}.lora_B.weight"] = torch.randn(inner, rank)
    out_module = "token_refiner.refiner_blocks.0.attn.to_out.0"
    state[f"{out_module}.lora_A.weight"] = torch.randn(rank, inner)
    state[f"{out_module}.lora_B.weight"] = torch.randn(hidden, rank)

    converted = convert_adapter(state, {"rank": rank, "alpha": rank})
    patcher = comfy.model_patcher.ModelPatcher(
        Root(), torch.device("cpu"), torch.device("cpu"))
    terms = {module: [(a, b, scale)] for module, (a, b, scale) in converted.items()}

    _validate_runtime_weight_targets(patcher, terms)
