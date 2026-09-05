from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file

import comfy.model_patcher

from vdn_h3.apply import apply_adapters


class MarkedLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)
        # Current Comfy castable Linear modules expose these lists. Runtime VDN
        # targets them through ModelPatcher.add_weight_wrapper; module.forward stays
        # untouched.
        self.weight_function = []
        self.bias_function = []


class Adaln(nn.Module):
    def __init__(self, curve_width, out):
        super().__init__()
        self.linear = MarkedLinear(curve_width, out)
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
        Root(table, out), torch.device("cpu"), torch.device("cpu"))


def _stage(tmp_path, basis, mean):
    stage = tmp_path / "stage"
    stage.mkdir()
    save_file(
        {"adaln_basis": basis.float(), "adaln_mean": mean.float()},
        stage / "adaln_affine.safetensors")
    return stage


def test_curve_bypass_uses_weight_and_bias_wrappers_without_forward_patch(tmp_path):
    torch.manual_seed(421)
    table = torch.randn(9, 3)
    dense, rank, out = 7, 2, 6
    basis = torch.randn(3, dense)
    mean = torch.randn(dense)
    stage = _stage(tmp_path, basis, mean)
    patcher = _patcher(table, out)
    forward_before = patcher.get_model_object(
        "diffusion_model.blocks.0.adaln_proj").forward
    converted = {
        "turbo": {
            "blocks.0.adaln_proj.linear": (
                torch.randn(rank, dense), torch.randn(out, rank), 1.0)
        }
    }

    report = apply_adapters(
        patcher, converted, 0.75, mode="bypass", stage_path=str(stage))

    weight_key = "diffusion_model.blocks.0.adaln_proj.linear.weight"
    bias_key = "diffusion_model.blocks.0.adaln_proj.linear.bias"
    assert weight_key in patcher.weight_wrapper_patches
    assert bias_key in patcher.weight_wrapper_patches
    assert len(patcher.weight_wrapper_patches[weight_key]) == 1
    assert len(patcher.weight_wrapper_patches[bias_key]) == 1
    assert patcher.object_patches == {}
    assert patcher.get_model_object(
        "diffusion_model.blocks.0.adaln_proj").forward == forward_before
    assert report["runtime_lowvram"]["bias_wrappers"] == 1
    assert report["curve_adaln_projection"]["curve_width"] == 3
    assert report["curve_adaln_projection"]["dense_width"] == dense


def test_curve_merge_registers_normal_weight_and_bias_patches(tmp_path):
    torch.manual_seed(422)
    table = torch.randn(9, 3)
    dense, rank, out = 7, 2, 6
    stage = _stage(tmp_path, torch.randn(3, dense), torch.randn(dense))
    patcher = _patcher(table, out)
    converted = {
        "turbo": {
            "blocks.0.adaln_proj.linear": (
                torch.randn(rank, dense), torch.randn(out, rank), 0.8)
        }
    }

    report = apply_adapters(
        patcher, converted, 0.5, mode="merge", stage_path=str(stage))

    weight_key = "diffusion_model.blocks.0.adaln_proj.linear.weight"
    bias_key = "diffusion_model.blocks.0.adaln_proj.linear.bias"
    assert weight_key in patcher.patches
    assert bias_key in patcher.patches
    assert patcher.weight_wrapper_patches == {}
    assert patcher.object_patches == {}
    assert report["curve_adaln_projection"]["mode"] == "merge"
    assert report["curve_adaln_projection"]["weight_patches"] == 1
    assert report["curve_adaln_projection"]["bias_patches"] == 1


def test_curve_bypass_managed_model_owns_projected_bias_offset(tmp_path):
    torch.manual_seed(423)
    table = torch.randn(9, 3)
    dense, rank, out = 7, 2, 6
    stage = _stage(tmp_path, torch.randn(3, dense), torch.randn(dense))
    patcher = _patcher(table, out)
    converted = {
        "turbo": {
            "blocks.0.adaln_proj.linear": (
                torch.randn(rank, dense), torch.randn(out, rank), 1.0)
        }
    }

    report = apply_adapters(
        patcher, converted, 1.0, mode="bypass", stage_path=str(stage))
    owner_key = report["runtime_lowvram"]["owner_key"]
    managed_patcher = patcher.get_additional_models_with_key(owner_key)[0]
    managed = managed_patcher.model

    assert managed.term_count() == 1
    assert managed.bias_term_count() == 1
    assert any(p.dtype == torch.float32 and p.numel() == out for p in managed.parameters())


def test_curve_bypass_weight_plus_bias_wrapper_matches_affine_dense_delta(tmp_path):
    torch.manual_seed(424)
    table = torch.randn(9, 3)
    dense, rank, out = 7, 2, 6
    basis = torch.randn(3, dense)
    mean = torch.randn(dense)
    stage = _stage(tmp_path, basis, mean)
    patcher = _patcher(table, out)
    linear = patcher.get_model_object("diffusion_model.blocks.0.adaln_proj.linear")
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
    weight_key = "diffusion_model.blocks.0.adaln_proj.linear.weight"
    bias_key = "diffusion_model.blocks.0.adaln_proj.linear.bias"
    wrapped_weight = patcher.weight_wrapper_patches[weight_key][0](base_weight)
    wrapped_bias = patcher.weight_wrapper_patches[bias_key][0](base_bias)

    coords = torch.randn(5, 3)
    got = F.linear(coords, wrapped_weight, wrapped_bias)
    dense_input = mean + coords @ basis
    expected = F.linear(coords, base_weight, base_bias) + F.linear(
        F.linear(dense_input, a), b) * strength

    assert torch.allclose(got, expected, atol=2e-5, rtol=2e-5)
