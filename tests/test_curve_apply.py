from __future__ import annotations

import torch
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional as F

import comfy.model_patcher

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


def test_curve_bypass_uses_native_projected_patches_not_weight_wrappers(tmp_path):
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
    assert weight_key in patcher.patches
    assert bias_key in patcher.patches
    assert patcher.weight_wrapper_patches == {}
    # Curve-only bypass has nothing to activation-hook; its exact projected
    # native terms are ordinary Comfy weight/bias patches.
    assert patcher.injections == {}
    assert report["runtime_lowvram"]["weight_wrappers"] == 0
    assert report["runtime_lowvram"]["bias_wrappers"] == 0
    assert report["runtime_lowvram"]["forward_hooks"] == 0
    assert (
        report["curve_adaln_projection"]["mode"]
        == "bypass_native_projected_patch"
    )
    assert report["curve_adaln_projection"]["weight_patches"] == 1
    assert report["curve_adaln_projection"]["bias_patches"] == 1


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


def test_curve_bypass_projected_patch_matches_affine_dense_delta(tmp_path):
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
        got = F.linear(coords, linear.weight, linear.bias)
        assert torch.allclose(got, expected, atol=2e-5, rtol=2e-5)
    finally:
        patcher.unpatch_model(device_to=torch.device("cpu"))

    assert torch.equal(linear.weight, base_weight)
    assert torch.equal(linear.bias, base_bias)
