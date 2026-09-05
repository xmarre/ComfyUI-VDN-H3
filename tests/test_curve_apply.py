from __future__ import annotations

import torch
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional as F

import comfy.model_patcher
from comfy.weight_adapter.bypass import BypassForwardHook
from comfy.weight_adapter.lora import LoRAAdapter

from vdn_h3.apply import apply_adapters


class Adaln(nn.Module):
    def __init__(self, curve_width, out):
        super().__init__()
        self.linear = nn.Linear(curve_width, out, bias=True)
        self.apply_silu = False


class Block(nn.Module):
    def __init__(self, curve_width, out):
        super().__init__()
        self.adaln_proj = Adaln(curve_width, out)


class DM(nn.Module):
    def __init__(self, table, out):
        super().__init__()
        self.use_adaln_curves = True
        self.register_buffer("adaln_t_table", table)
        self.blocks = nn.ModuleList([Block(table.shape[1], out)])


class Root(nn.Module):
    def __init__(self, table, out):
        super().__init__()
        self.diffusion_model = DM(table, out)
        self.device = torch.device("cpu")


def _patcher(table, out):
    return comfy.model_patcher.ModelPatcher(
        Root(table, out), torch.device("cpu"), torch.device("cpu")
    )


def _stage(tmp_path, basis, mean):
    stage = tmp_path / "stage"
    stage.mkdir()
    save_file(
        {"adaln_basis": basis.float(), "adaln_mean": mean.float()},
        stage / "adaln_affine.safetensors",
    )
    return stage


def _converted(dense, rank, out, scale=1.0):
    return {
        "turbo": {
            "blocks.0.adaln_proj.linear": (
                torch.randn(rank, dense),
                torch.randn(out, rank),
                scale,
            )
        }
    }


def test_curve_bypass_uses_projected_post_hook_not_native_patches(tmp_path):
    torch.manual_seed(421)
    table = torch.randn(9, 3)
    dense, rank, out = 7, 2, 6
    stage = _stage(tmp_path, torch.randn(3, dense), torch.randn(dense))
    patcher = _patcher(table, out)

    report = apply_adapters(
        patcher,
        _converted(dense, rank, out),
        0.75,
        mode="bypass",
        stage_path=str(stage),
    )

    weight_key = "diffusion_model.blocks.0.adaln_proj.linear.weight"
    bias_key = "diffusion_model.blocks.0.adaln_proj.linear.bias"
    assert weight_key not in patcher.patches
    assert bias_key not in patcher.patches
    assert patcher.weight_wrapper_patches == {}
    assert "vdn_lora" in patcher.injections
    runtime = report["runtime_lowvram"]
    assert runtime["weight_wrappers"] == 0
    assert runtime["bias_wrappers"] == 0
    assert runtime["forward_hooks"] == 1
    assert runtime["projected_curve_runtime_targets"] == 1
    assert runtime["projected_curve_weight_patches"] == 0
    assert runtime["projected_curve_bias_patches"] == 0
    assert runtime["runtime_preloaded_on_inject"] is True
    assert runtime["managed_adapter_bytes"] > 0
    projection = report["curve_adaln_projection"]
    assert projection["mode"] == "bypass_post_forward_projected_residual"
    assert projection["weight_patches"] == 0
    assert projection["bias_patches"] == 0
    assert projection["runtime_targets"] == 1


def test_curve_merge_registers_normal_weight_and_bias_patches(tmp_path):
    torch.manual_seed(422)
    table = torch.randn(9, 3)
    dense, rank, out = 7, 2, 6
    stage = _stage(tmp_path, torch.randn(3, dense), torch.randn(dense))
    patcher = _patcher(table, out)

    report = apply_adapters(
        patcher,
        _converted(dense, rank, out, scale=0.8),
        0.5,
        mode="merge",
        stage_path=str(stage),
    )

    weight_key = "diffusion_model.blocks.0.adaln_proj.linear.weight"
    bias_key = "diffusion_model.blocks.0.adaln_proj.linear.bias"
    assert weight_key in patcher.patches
    assert bias_key in patcher.patches
    assert patcher.weight_wrapper_patches == {}
    assert patcher.injections == {}
    assert report["curve_adaln_projection"]["mode"] == "merge"
    assert report["curve_adaln_projection"]["weight_patches"] == 1
    assert report["curve_adaln_projection"]["bias_patches"] == 1
    assert report["curve_adaln_projection"]["runtime_targets"] == 0


def test_curve_bypass_projected_post_hook_matches_affine_dense_delta(tmp_path):
    torch.manual_seed(424)
    table = torch.randn(9, 3)
    dense, rank, out = 7, 2, 6
    basis = torch.randn(3, dense)
    mean = torch.randn(dense)
    stage = _stage(tmp_path, basis, mean)
    patcher = _patcher(table, out)
    linear = patcher.get_model_object(
        "diffusion_model.blocks.0.adaln_proj.linear"
    )
    base_weight = linear.weight.detach().clone()
    base_bias = linear.bias.detach().clone()
    true_forward = linear.forward
    a = torch.randn(rank, dense)
    b = torch.randn(out, rank)
    strength = 0.625

    apply_adapters(
        patcher,
        {"turbo": {"blocks.0.adaln_proj.linear": (a, b, 1.0)}},
        strength,
        mode="bypass",
        stage_path=str(stage),
    )

    coords = torch.randn(5, 3)
    dense_input = mean + coords @ basis
    expected = F.linear(coords, base_weight, base_bias) + F.linear(
        F.linear(dense_input, a), b
    ) * strength

    patcher.patch_model(device_to=torch.device("cpu"))
    try:
        assert linear.forward == true_forward
        assert torch.equal(linear.weight, base_weight)
        assert torch.equal(linear.bias, base_bias)
        got = linear(coords)
        assert torch.allclose(got, expected, atol=2e-5, rtol=2e-5)
    finally:
        patcher.unpatch_model(device_to=torch.device("cpu"))

    assert linear.forward == true_forward
    assert torch.equal(linear.weight, base_weight)
    assert torch.equal(linear.bias, base_bias)


def test_curve_bypass_coexists_with_external_comfy_bypass_chain(tmp_path, monkeypatch):
    torch.manual_seed(425)
    table = torch.randn(9, 3)
    dense, rank, out = 7, 2, 6
    basis = torch.randn(3, dense)
    mean = torch.randn(dense)
    stage = _stage(tmp_path, basis, mean)
    patcher = _patcher(table, out)
    linear = patcher.get_model_object(
        "diffusion_model.blocks.0.adaln_proj.linear"
    )
    true_forward = linear.forward

    vdn_a = torch.randn(rank, dense)
    vdn_b = torch.randn(out, rank)
    apply_adapters(
        patcher,
        {"turbo": {"blocks.0.adaln_proj.linear": (vdn_a, vdn_b, 1.0)}},
        0.7,
        mode="bypass",
        stage_path=str(stage),
    )

    ext_down = torch.randn(2, 3)
    ext_up = torch.randn(out, 2)
    alpha = torch.tensor(float(ext_down.shape[0]))
    adapter = LoRAAdapter(
        set(), (ext_up, ext_down, alpha, None, None, None)
    )
    external = BypassForwardHook(linear, adapter, multiplier=0.4)
    monkeypatch.setattr(
        "comfy.model_management.get_torch_device", lambda: torch.device("cpu")
    )

    injection = patcher.injections["vdn_lora"][0]
    coords = torch.randn(4, 3)
    dense_input = mean + coords @ basis
    vdn_delta = F.linear(F.linear(dense_input, vdn_a), vdn_b) * 0.7
    ext_delta = F.linear(F.linear(coords, ext_down), ext_up) * 0.4
    base = true_forward(coords)

    external.inject()
    external_forward = linear.forward
    injection.inject(patcher)
    try:
        assert linear.forward == external_forward
        assert torch.allclose(
            linear(coords), base + ext_delta + vdn_delta, atol=2e-5, rtol=2e-5
        )
        injection.eject(patcher)
        assert linear.forward == external_forward
        assert torch.allclose(linear(coords), base + ext_delta, atol=2e-5, rtol=2e-5)
    finally:
        injection.eject(patcher)
        external.eject()

    assert linear.forward == true_forward
