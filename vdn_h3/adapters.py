"""VDN-H3 adapter (LoRA) loading for ComfyUI.

The released adapters are stored against the diffusers MiniMax-H3 layout with peft
key names. ComfyUI's MiniMax-H3 instead fuses the attention projections into one
qkv_proj and packs the MLP swiglu as [gate; value] where diffusers packs [value;
gate]. This module rewrites adapter tensors onto ComfyUI module paths in memory at
load time; nothing on disk is converted.

Fusion of three per-projection LoRA pairs (to_q/to_k/to_v) into one fused-qkv pair is
exact: each source projection contributes ``scale_i * (B_i @ A_i)``. The fused A is
the concatenation of all ranks and the fused B is block diagonal, with per-projection
scale absorbed into its B block when necessary. Different Q/K/V ranks and alpha/rank
values therefore remain representable without approximation.
"""
import torch

# diffusers target suffix (after transformer_blocks.N. / token_refiner.refiner_blocks.N.)
# -> (kind, comfy suffix)
_ATTN_QKV = ("attn.orig.to_q", "attn.orig.to_k", "attn.orig.to_v")
_ATTN_OUT = "attn.orig.to_out.0"


def _comfy_path(key, is_refiner):
    """diffusers module path -> (comfy module path, conversion kind)."""
    if is_refiner:
        stem = key.replace("token_refiner.refiner_blocks.", "token_refiner.blocks.")
    else:
        stem = key.replace("transformer_blocks.", "blocks.")
    for proj in _ATTN_QKV:
        if stem.endswith("." + proj):
            return stem[: -len(proj)] + "attn.qkv_proj", "qkv"
    if stem.endswith("." + _ATTN_OUT):
        return stem[: -len(_ATTN_OUT)] + "attn.out_proj", "out"
    if stem.endswith(".ff.net.0.proj"):
        return stem[: -len("ff.net.0.proj")] + "mlp.fc1", "swiglu"
    if stem.endswith(".ff.net.2"):
        return stem[: -len("ff.net.2")] + "mlp.fc2", "plain"
    if stem.endswith(".adaln_proj.linear") or stem == "norm_out.linear":
        path = stem if stem.endswith(".adaln_proj.linear") \
            else stem.replace("norm_out.linear", "final_layer.adaln_proj.linear")
        return path, "adaln"
    return stem, "plain"


def parse_adapter_state(sd):
    """peft-named safetensors dict -> {(diffusers module, infix-free): {lora_A/lora_B}}."""
    out = {}
    for key, tensor in sd.items():
        if ".lora_" not in key:
            continue
        module, rest = key.split(".lora_", 1)
        side = rest.split(".")[0]                    # A or B
        if side not in ("A", "B"):
            continue
        slot = out.setdefault(module, {})
        if side in slot:
            raise ValueError(f"duplicate LoRA {side} tensor for {module}")
        slot[side] = tensor.float()
    return out


def per_module_scale(adapter_cfg, module):
    """peft scaling alpha/rank for one module, honoring rank_pattern/alpha_pattern."""
    cfg = adapter_cfg.get("config", adapter_cfg)
    rank = cfg.get("rank", 64)
    alpha = cfg.get("alpha", rank)
    for pattern, value in (cfg.get("rank_pattern") or {}).items():
        if pattern in module:
            rank = value
    for pattern, value in (cfg.get("alpha_pattern") or {}).items():
        if pattern in module:
            alpha = value
    if not isinstance(rank, (int, float)) or rank <= 0:
        raise ValueError(f"invalid LoRA rank {rank!r} for {module}")
    if not isinstance(alpha, (int, float)):
        raise ValueError(f"invalid LoRA alpha {alpha!r} for {module}")
    return alpha / rank


def _require_pair(module, sides):
    if set(sides) != {"A", "B"}:
        raise ValueError(
            f"LoRA module {module} must contain exactly A and B tensors; got {sorted(sides)}")
    a, b = sides["A"], sides["B"]
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
        raise ValueError(
            f"LoRA module {module} has incompatible A{tuple(a.shape)} B{tuple(b.shape)}")
    return a, b


def convert_adapter(sd, adapter_cfg):
    """{diffusers module: {A, B}} -> {comfy module: (lora_A, lora_B, scale)}.

    qkv targets fold into one variable-rank block-diagonal pair; swiglu halves swap;
    refiner keys reroot onto ComfyUI's token_refiner.blocks naming.
    """
    parsed = parse_adapter_state(sd)
    qkv_groups = {}
    out = {}
    for module, sides in parsed.items():
        is_refiner = module.startswith("token_refiner.")
        path, kind = _comfy_path(module, is_refiner)
        a, b = _require_pair(module, sides)
        scale = per_module_scale(adapter_cfg, module)
        if kind == "qkv":
            qkv_groups.setdefault(path, []).append((module, a, b, scale))
            continue
        if kind == "swiglu":
            if b.shape[0] % 2:
                raise ValueError(
                    f"SwiGLU LoRA B rows must be even for {module}; got {b.shape[0]}")
            half = b.shape[0] // 2
            value_half, gate_half = b[:half], b[half:]
            b = torch.cat([gate_half, value_half], dim=0)
        if path in out:
            raise ValueError(f"multiple adapter modules map to ComfyUI target {path}")
        out[path] = (a, b, scale)

    expected_suffixes = ("to_q", "to_k", "to_v")
    for path, group in qkv_groups.items():
        by_suffix = {}
        for item in group:
            module = item[0]
            suffixes = [s for s in expected_suffixes if module.endswith("." + s)]
            if len(suffixes) != 1:
                raise ValueError(f"cannot identify Q/K/V projection for {module}")
            suffix = suffixes[0]
            if suffix in by_suffix:
                raise ValueError(f"duplicate {suffix} LoRA projection for {path}")
            by_suffix[suffix] = item
        missing = [s for s in expected_suffixes if s not in by_suffix]
        if missing:
            raise ValueError(
                f"incomplete Q/K/V LoRA triplet for {path}; missing {missing}")
        ordered = [by_suffix[s] for s in expected_suffixes]

        ranks = [item[1].shape[0] for item in ordered]
        input_dims = [item[1].shape[1] for item in ordered]
        output_dims = [item[2].shape[0] for item in ordered]
        scales = [float(item[3]) for item in ordered]
        if len(set(input_dims)) != 1 or len(set(output_dims)) != 1:
            raise ValueError(
                f"incompatible Q/K/V projection shapes for {path}: "
                f"input={input_dims}, output={output_dims}")

        # Preserve a common external scale when possible. Otherwise absorb each
        # projection's scale into its B block and expose a unit aggregate scale.
        common_scale = scales[0] if len(set(scales)) == 1 and scales[0] != 0.0 else 1.0
        total_rank = sum(ranks)
        out_dim = output_dims[0]
        a_fused = torch.cat([item[1] for item in ordered], dim=0)  # [sum(r_i), in]
        b_fused = torch.zeros(out_dim * 3, total_rank, dtype=a_fused.dtype)
        col = 0
        for i, (_mod, _a, b, scale) in enumerate(ordered):
            rank = b.shape[1]
            b_fused[i * out_dim:(i + 1) * out_dim, col:col + rank] = (
                b * (float(scale) / common_scale))
            col += rank
        if path in out:
            raise ValueError(f"multiple adapter modules map to ComfyUI target {path}")
        out[path] = (a_fused, b_fused, common_scale)
    return out


def load_comfy_lora_format(converted, strength=1.0):
    """{comfy module: (A, B, scale)} -> the dict comfy.lora.load_lora consumes, keyed
    by diffusion-model weight key with peft lora_A/lora_B names. ComfyUI divides the
    stored `.alpha` by the rank itself, so write alpha = scale * rank; the node's
    strength is applied separately at injection time."""
    lora = {}
    for path, (a, b, scale) in converted.items():
        lora[path + ".lora_A.weight"] = a.contiguous()
        lora[path + ".lora_B.weight"] = b.contiguous()
        lora[path + ".alpha"] = torch.tensor(scale * a.shape[0] * strength)
    return lora