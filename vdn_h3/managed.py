"""ComfyUI-managed ownership for optional resident VDN branch weights."""
from __future__ import annotations

import torch
from torch import nn

import comfy.model_management
import comfy.model_patcher

from vdn_h3.spec import resolve_branch_weights


class BranchBlockWeights(nn.Module):
    """One block's checkpoint tensors as non-trainable Parameters.

    ParameterList gives ModelPatcher concrete objects to size/load/offload while the
    private name map preserves the official checkpoint names consumed by LinearBranch.
    """
    def __init__(self, weights):
        super().__init__()
        resolved = resolve_branch_weights(weights, "cpu", dtype=None)
        self.values = nn.ParameterList()
        self._names = []
        for name, tensor in resolved.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"resident VDN branch tensor {name} is not a torch.Tensor; use stream mode")
            # QuantizedTensor subclasses may not support Parameter wrapping correctly;
            # fail closed rather than dequantizing a resident branch behind the user's back.
            if type(tensor) is not torch.Tensor and tensor.__class__.__name__ == "QuantizedTensor":
                raise RuntimeError(
                    "resident mode does not currently materialize quantized VDN branch "
                    "files; select stream mode so their native quantized layout is preserved")
            self.values.append(nn.Parameter(tensor, requires_grad=False))
            self._names.append(name)

    def weights_on(self, device, dtype):
        out = {}
        for name, value in zip(self._names, self.values):
            out[name] = comfy.model_management.cast_to(
                value, device=device, dtype=dtype, copy=False)
        return out


class VDNBranchWeightsModel(nn.Module):
    def __init__(self, branches, offload_device):
        super().__init__()
        self.blocks = nn.ModuleList([BranchBlockWeights(weights) for weights in branches])
        self.device = offload_device

    def weights_on(self, index, device, dtype):
        return self.blocks[index].weights_on(device, dtype)


def make_managed_branch_patcher(branches, base_patcher):
    """Build the optional resident branch as a real additional ModelPatcher."""
    model = VDNBranchWeightsModel(branches, base_patcher.offload_device)
    patcher = comfy.model_patcher.ModelPatcher(
        model,
        load_device=base_patcher.load_device,
        offload_device=base_patcher.offload_device,
    )
    return model, patcher
