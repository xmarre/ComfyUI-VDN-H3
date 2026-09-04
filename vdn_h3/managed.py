"""ComfyUI-managed ownership for VDN branch and runtime-adapter tensors."""
from __future__ import annotations

import torch
from torch import nn

import comfy.model_management
import comfy.model_patcher

from vdn_h3.spec import resolve_branch_weights


class BranchBlockWeights(nn.Module):
    """One block's checkpoint tensors as non-trainable Parameters."""
    def __init__(self, weights):
        super().__init__()
        resolved = resolve_branch_weights(weights, "cpu", dtype=None)
        self.values = nn.ParameterList()
        self._names = []
        for name, tensor in resolved.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"resident VDN branch tensor {name} is not a torch.Tensor; use stream mode")
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


class RuntimeLoRATermsModel(nn.Module):
    """Low-rank runtime adapter tensors owned by an additional ModelPatcher.

    The model contains only A/B factors, never a full patched H3 weight. Keeping them
    as Parameters lets normal Comfy model loading decide whether those small tensors
    stay on the compute device or are offloaded. Runtime weight wrappers are stateless
    references into this model, so they do not carry a private GPU cache across clones.
    """

    def __init__(self, terms_by_module, offload_device):
        super().__init__()
        self.values = nn.ParameterList()
        self._terms = {}
        self.device = offload_device

        for module in sorted(terms_by_module):
            refs = []
            for a, b, scale in terms_by_module[module]:
                if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
                    raise TypeError(f"runtime LoRA {module} A/B must be torch tensors")
                ai = len(self.values)
                self.values.append(nn.Parameter(a.detach().contiguous(), requires_grad=False))
                bi = len(self.values)
                self.values.append(nn.Parameter(b.detach().contiguous(), requires_grad=False))
                refs.append((ai, bi, float(scale)))
            self._terms[module] = tuple(refs)

    def terms_on(self, module, device, dtype):
        refs = self._terms[module]
        return tuple(
            (
                comfy.model_management.cast_to(
                    self.values[ai], device=device, dtype=dtype, copy=False),
                comfy.model_management.cast_to(
                    self.values[bi], device=device, dtype=dtype, copy=False),
                scale,
            )
            for ai, bi, scale in refs
        )

    def term_count(self):
        return sum(len(refs) for refs in self._terms.values())


def make_managed_runtime_lora_patcher(terms_by_module, base_patcher):
    """Build Comfy-owned storage for runtime-low-VRAM LoRA A/B factors."""
    model = RuntimeLoRATermsModel(terms_by_module, base_patcher.offload_device)
    patcher = comfy.model_patcher.ModelPatcher(
        model,
        load_device=base_patcher.load_device,
        offload_device=base_patcher.offload_device,
    )
    return model, patcher
