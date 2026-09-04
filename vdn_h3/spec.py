"""VDN-H3 checkpoint discovery, strict ModelSpec validation and bounded resources.

The released checkpoint directory is consumed in place; tensors are re-keyed only in
memory.  No ``safe_open`` handle survives a function call.  This is deliberate: an
indefinite mmap keyed only by path can keep a replaced checkpoint stale and can keep
the old file mapped on Windows.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import struct
from collections import OrderedDict
from dataclasses import dataclass

import folder_paths
import torch
from safetensors import safe_open

_log = logging.getLogger("comfy.vdn")

SPEC_FORMAT_VERSION = 2
HYBRID_TRANSFORM_VERSION = 2
SUPPORTED_DELTA_RULES = ("vdn_solve", "sana_scaled", "vdn_scaled")
SUPPORTED_ANCHORS = ("none", "columns", "rows", "both")
SHORT_CONV_TARGETS = ("q", "k", "v")
BRIDGE_VALUES = ("alpha", "none")
_RUNTIME_KEYS = {
    "softmax_backend", "rmsnorm_backend", "fp8", "compile",
    "inference_kernels", "optimized_paths", "w_o_far_scale", "window_decomp",
    "warmup_steps",
}

BRANCH_FILE = "model.safetensors"
BRANCH_FILE_INT8 = "model_int8_convrot_comfyui.safetensors"

try:
    from comfy_kitchen.tensor import QuantizedTensor, TensorWiseINT8Layout
    _KITCHEN_OK = True
except ImportError:  # pragma: no cover - current Comfy ships comfy-kitchen
    QuantizedTensor = TensorWiseINT8Layout = None
    _KITCHEN_OK = False

SAFETENSORS_DTYPES = {
    "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
    "I8": torch.int8, "U8": torch.uint8, "I16": torch.int16,
    "I32": torch.int32, "I64": torch.int64,
}


@dataclass(frozen=True)
class FileIdentity:
    realpath: str
    mtime_ns: int
    size: int
    inode: int | None


def file_identity(path: str) -> FileIdentity:
    real = os.path.realpath(path)
    st = os.stat(real)
    return FileIdentity(real, st.st_mtime_ns, st.st_size, getattr(st, "st_ino", None))


def _same_identity(expected: FileIdentity):
    current = file_identity(expected.realpath)
    if current != expected:
        raise RuntimeError(
            f"VDN checkpoint file changed after it was loaded: {expected.realpath}. "
            "Re-run the Apply VDN node so descriptors are rebuilt from the new file.")


def _branch_file(path):
    plain = os.path.join(path, "linear_branch", BRANCH_FILE)
    if os.path.isfile(plain):
        return plain
    quant = os.path.join(path, "linear_branch", BRANCH_FILE_INT8)
    if os.path.isfile(quant):
        return quant
    return plain


def _read_header(path):
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: truncated safetensors header")
        size = struct.unpack("<Q", raw)[0]
        if size <= 0 or size > (64 << 20):
            raise ValueError(f"{path}: invalid safetensors header size {size}")
        payload = fh.read(size)
        if len(payload) != size:
            raise ValueError(f"{path}: truncated safetensors JSON header")
        return json.loads(payload)


def _owned_or_transferred(t, device, dtype=None):
    device = torch.device(device)
    if device.type == "cpu":
        out = t.clone()
        return out if dtype is None or out.dtype == dtype else out.to(dtype=dtype)
    return t.to(device=device, dtype=dtype or t.dtype)


class LazyBranchTensor:
    """Descriptor for one tensor in a branch safetensors file.

    The descriptor owns no mmap.  ``resolve_branch_weights`` opens the containing file
    once for the whole block and closes it after every requested tensor has either
    been cloned to CPU or transferred to the target device.
    """
    __slots__ = (
        "identity", "key", "scale_key", "conf", "shape", "dtype",
    )

    def __init__(self, identity, key, shape, dtype, scale_key=None, conf=None):
        self.identity = identity
        self.key = key
        self.scale_key = scale_key
        self.conf = conf
        self.shape = shape
        self.dtype = dtype


def resolve_branch_weights(weights: dict, device, dtype=None) -> dict:
    """Resolve a block's descriptors with one bounded safe_open lifetime."""
    if not weights:
        return {}
    descriptors = [v for v in weights.values() if isinstance(v, LazyBranchTensor)]
    if not descriptors:
        return {
            k: (v if dtype is None and v.device == torch.device(device)
                else v.to(device=device, dtype=dtype or v.dtype))
            for k, v in weights.items()
        }
    identities = {d.identity for d in descriptors}
    if len(identities) != 1:
        raise RuntimeError("one VDN branch block unexpectedly spans multiple files")
    identity = next(iter(identities))
    _same_identity(identity)
    out = {}
    with safe_open(identity.realpath, framework="pt", device="cpu") as handle:
        for name, desc in weights.items():
            if not isinstance(desc, LazyBranchTensor):
                out[name] = desc.to(device=device, dtype=dtype or desc.dtype)
                continue
            if desc.conf is None:
                out[name] = _owned_or_transferred(
                    handle.get_tensor(desc.key), device, dtype)
                continue
            if not _KITCHEN_OK:
                raise RuntimeError(
                    "quantized VDN branch requires comfy-kitchen from current ComfyUI")
            qdata = _owned_or_transferred(handle.get_tensor(desc.key), device)
            scale = _owned_or_transferred(handle.get_tensor(desc.scale_key), device)
            out[name] = QuantizedTensor(
                qdata,
                "TensorWiseINT8Layout",
                TensorWiseINT8Layout.Params(
                    scale=scale,
                    orig_dtype=dtype or torch.bfloat16,
                    orig_shape=tuple(desc.shape),
                    is_weight=True,
                    convrot=bool(desc.conf.get("convrot", False)),
                    convrot_groupsize=int(desc.conf.get("convrot_groupsize", 256)),
                ),
            )
    return out


def _lazy_branch_sd(path):
    identity = file_identity(path)
    header = _read_header(path)
    conf_keys = [k for k in header if k.endswith(".comfy_quant")]
    confs = {}
    if conf_keys:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in conf_keys:
                layer = key[: -len(".comfy_quant")]
                raw = handle.get_tensor(key).clone().tolist()
                confs[layer] = json.loads(bytes(raw).decode("utf-8"))

    out = {}
    for key, meta in header.items():
        if key == "__metadata__" or key.endswith(".comfy_quant"):
            continue
        if key.endswith(".weight_scale") and key[: -len(".weight_scale")] in confs:
            continue
        layer = key[: -len(".weight")] if key.endswith(".weight") else None
        conf = confs.get(layer) if layer else None
        scale_key = key + "_scale" if conf else None
        if conf and scale_key not in header:
            raise ValueError(f"{path}: quantized tensor {key} is missing {scale_key}")
        dtype = SAFETENSORS_DTYPES.get(meta["dtype"])
        if dtype is None:
            raise ValueError(f"{path}: unsupported safetensors dtype {meta['dtype']} for {key}")
        out[key] = LazyBranchTensor(
            identity,
            key,
            torch.Size(meta["shape"]),
            dtype,
            scale_key,
            conf,
        )
    return out


def register_folder():
    for base in {os.path.dirname(p) for p in folder_paths.get_folder_paths("loras")}:
        folder_paths.add_model_folder_path("vdn", os.path.join(base, "vdn"))


def vdn_folders():
    if "vdn" not in folder_paths.folder_names_and_paths:
        register_folder()
    return folder_paths.get_folder_paths("vdn")


def list_vdn_checkpoints():
    found = []
    for root in vdn_folders():
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, _files in os.walk(root):
            if os.path.isfile(_branch_file(dirpath)):
                found.append(os.path.relpath(dirpath, root).replace("\\", "/"))
                dirnames[:] = []
    return sorted(found)


def resolve_vdn_checkpoint(name):
    for root in vdn_folders():
        path = os.path.join(root, *name.split("/"))
        if os.path.isfile(_branch_file(path)):
            return path
    raise FileNotFoundError(
        f"VDN checkpoint {name!r} not found under {vdn_folders()}. Keep the official "
        "OpenVDN stage directory layout intact under models/vdn.")


def _read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _config_hash(resolved_config):
    blob = json.dumps(resolved_config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _require_resolved(value, where):
    if value is None:
        raise ValueError(
            f"{where} is unresolved (null); checkpoint ModelSpec values must be resolved")
    if isinstance(value, dict):
        for key, child in value.items():
            _require_resolved(child, f"{where}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _require_resolved(child, f"{where}[{index}]")


def _flat_keys(config):
    for key, value in config.items():
        yield key
        if isinstance(value, dict):
            yield from _flat_keys(value)


def _validate_hybrid(config):
    lin = config.get("linear_attention")
    soft = config.get("softmax_attention")
    if not isinstance(lin, dict) or not isinstance(soft, dict):
        raise ValueError("hybrid_attention requires linear_attention and softmax_attention objects")
    if lin.get("delta_rule") not in SUPPORTED_DELTA_RULES:
        raise ValueError(
            f"delta_rule {lin.get('delta_rule')!r} not in {SUPPORTED_DELTA_RULES}")
    if lin.get("bridge") not in BRIDGE_VALUES:
        raise ValueError(f"bridge {lin.get('bridge')!r} not in {BRIDGE_VALUES}")
    short = lin.get("short_conv")
    targets = short.get("targets") if isinstance(short, dict) else None
    if (not isinstance(targets, list)
            or len(set(targets)) != len(targets)
            or any(target not in SHORT_CONV_TARGETS for target in targets)):
        raise ValueError(
            f"short_conv must contain a distinct subset of {SHORT_CONV_TARGETS}; got {short!r}")
    if config.get("anchor_frames") not in SUPPORTED_ANCHORS:
        raise ValueError(
            f"anchor_frames {config.get('anchor_frames')!r} not in {SUPPORTED_ANCHORS}")
    if not isinstance(lin.get("linear_head_dim"), int) or lin["linear_head_dim"] <= 0:
        raise ValueError("linear_head_dim must be a resolved positive int")
    if not isinstance(soft.get("radius"), int) or soft["radius"] < 0:
        raise ValueError("softmax_attention.radius must be a resolved non-negative int")
    if not isinstance(soft.get("chunk"), int) or soft["chunk"] < 0:
        raise ValueError("softmax_attention.chunk must be a resolved non-negative int")
    for key in ("enable_softmax_gate",):
        if not isinstance(config.get(key), bool):
            raise ValueError(f"{key} must be a resolved bool")
    for key in ("a_fp32", "enable_text_state"):
        if key in lin and not isinstance(lin[key], bool):
            raise ValueError(f"linear_attention.{key} must be a resolved bool")


def validate_model_spec(payload):
    """Mirror the official OpenVDN ModelSpec v2 contract relevant to inference."""
    if not isinstance(payload, dict):
        raise ValueError("model_spec.json must contain an object")
    if payload.get("format_version") != SPEC_FORMAT_VERSION:
        raise ValueError(
            f"spec format_version {payload.get('format_version')} != {SPEC_FORMAT_VERSION}")

    base = payload.get("base")
    if not isinstance(base, dict) or not isinstance(base.get("resolved_config"), dict):
        raise ValueError("ModelSpec.base.resolved_config is required")
    expected_hash = _config_hash(base["resolved_config"])
    stored_hash = base.get("config_hash")
    if stored_hash and stored_hash != expected_hash:
        raise ValueError(
            f"base.config_hash {stored_hash[:12]} does not match resolved_config "
            f"({expected_hash[:12]})")

    transforms = payload.get("transforms", [])
    if not isinstance(transforms, list):
        raise ValueError("ModelSpec.transforms must be a list")
    hybrids = []
    for index, transform in enumerate(transforms):
        if not isinstance(transform, dict):
            raise ValueError(f"transforms[{index}] must be an object")
        config = transform.get("config")
        if not isinstance(config, dict):
            raise ValueError(f"transforms[{index}].config must be an object")
        _require_resolved(config, f"transforms[{transform.get('type')}]")
        leaked = sorted(set(_flat_keys(config)) & _RUNTIME_KEYS)
        if leaked:
            raise ValueError(
                f"transforms[{transform.get('type')}] carries runtime keys {leaked}")
        if transform.get("type") == "hybrid_attention":
            if transform.get("version") != HYBRID_TRANSFORM_VERSION:
                raise ValueError(
                    f"hybrid_attention transform version {transform.get('version')}; "
                    f"expected {HYBRID_TRANSFORM_VERSION}")
            _validate_hybrid(config)
            hybrids.append(transform)
    if len(hybrids) != 1:
        raise ValueError(
            "model_spec.json must carry exactly one hybrid_attention transform, "
            f"got {len(hybrids)}")

    adapters = payload.get("adapters", [])
    if not isinstance(adapters, list):
        raise ValueError("ModelSpec.adapters must be a list")
    for index, adapter in enumerate(adapters):
        if not isinstance(adapter, dict):
            raise ValueError(f"adapters[{index}] must be an object")
        if not isinstance(adapter.get("type"), str) or not isinstance(adapter.get("version"), int):
            raise ValueError(f"adapters[{index}] needs string type and integer version")
        config = adapter.get("config")
        if not isinstance(config, dict):
            raise ValueError(f"adapters[{index}].config must be an object")
        _require_resolved(config, f"adapters[{adapter.get('type')}]")
    return hybrids[0]


def transform_config(spec):
    transform = validate_model_spec(spec)
    cfg = transform["config"]
    lin = cfg["linear_attention"]
    soft = cfg["softmax_attention"]
    return {
        "enable_softmax_gate": cfg["enable_softmax_gate"],
        "anchor_frames": cfg["anchor_frames"],
        "radius": soft["radius"],
        "chunk": soft["chunk"],
        "delta_rule": lin["delta_rule"],
        "bridge": lin["bridge"],
        "a_fp32": lin.get("a_fp32", True),
        "linear_head_dim": lin["linear_head_dim"],
        "short_conv": tuple(lin["short_conv"]["targets"]),
        "enable_text_state": lin.get("enable_text_state", False),
    }


def _required_branch_tensors(cfg):
    required = {
        "to_out_linear.weight",
        "beta_proj.weight",
        "norm.weight",
        "alpha.A_log",
        "alpha.dt_bias",
        "alpha.down.weight",
        "alpha.up.weight",
        "output_gate.down.weight",
        "output_gate.up.weight",
        "output_gate.up.bias",
    }
    if cfg["enable_softmax_gate"]:
        required |= {"softmax_gate.up.weight", "softmax_gate.up.bias"}
    for target in cfg["short_conv"]:
        required |= {
            f"short_conv.{target}_sp.weight",
            f"short_conv.{target}_tm.weight",
        }
    return required


def _split_branches(path, branch_sd, cfg):
    indices = set()
    pattern = re.compile(r"^transformer_blocks\.(\d+)\.attn\.")
    for key in branch_sd:
        match = pattern.match(key)
        if match:
            indices.add(int(match.group(1)))
    if not indices:
        raise ValueError(f"{path}: branch checkpoint has no transformer blocks")
    expected_indices = set(range(max(indices) + 1))
    if indices != expected_indices:
        raise ValueError(
            f"{path}: branch block indices are not contiguous; got {sorted(indices)}")

    required = _required_branch_tensors(cfg)
    branches = []
    for index in range(max(indices) + 1):
        prefix = f"transformer_blocks.{index}.attn."
        weights = {}
        for key, tensor in branch_sd.items():
            if not key.startswith(prefix):
                continue
            name = key[len(prefix):]
            if name.startswith("linear_attention."):
                name = name[len("linear_attention."):]
            weights[name] = tensor
        missing = sorted(required - set(weights))
        if missing:
            raise ValueError(
                f"{path}: block {index} is missing required branch tensors: {missing}")
        branches.append(weights)
    return branches


def _load_owned_safetensors(path):
    identity = file_identity(path)
    out = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            out[key] = handle.get_tensor(key).clone()
    _same_identity(identity)
    return out


def _validate_adapter_weights(name, state, config, path):
    modules = {}
    for key, tensor in state.items():
        if ".lora_" not in key:
            continue
        module, rest = key.split(".lora_", 1)
        side = rest.split(".", 1)[0]
        if side in ("A", "B"):
            modules.setdefault(module, {})[side] = tensor
    if not modules:
        raise ValueError(f"{path}: adapter {name!r} contains no LoRA A/B tensors")
    for module, sides in modules.items():
        if set(sides) != {"A", "B"}:
            raise ValueError(
                f"{path}: adapter {name!r} module {module} does not have both A and B")
        a, b = sides["A"], sides["B"]
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            raise ValueError(
                f"{path}: adapter {name!r} module {module} has incompatible "
                f"A{tuple(a.shape)} B{tuple(b.shape)}")
    if not isinstance(config, dict):
        raise ValueError(f"{path}: adapter {name!r} config must be an object")


def _stage_identity(path):
    files = [_branch_file(path), os.path.join(path, "model_spec.json")]
    adapters_root = os.path.join(path, "adapters")
    if os.path.isdir(adapters_root):
        for name in sorted(os.listdir(adapters_root)):
            directory = os.path.join(adapters_root, name)
            for filename in ("adapter_config.json", "adapter_model.safetensors"):
                candidate = os.path.join(directory, filename)
                if os.path.isfile(candidate):
                    files.append(candidate)
    return tuple(file_identity(p) for p in files)


_CACHE = OrderedDict()
_MAX_CACHE = 2


def load_vdn_checkpoint(path):
    """Load one official stage with strict invalidation and owned adapter tensors."""
    identity = _stage_identity(path)
    hit = _CACHE.get(identity)
    if hit is not None:
        _CACHE.move_to_end(identity)
        return hit

    branch_path = _branch_file(path)
    if not os.path.isfile(branch_path):
        raise FileNotFoundError(f"{path}: missing linear_branch/{BRANCH_FILE}")
    spec_path = os.path.join(path, "model_spec.json")
    if not os.path.isfile(spec_path):
        raise FileNotFoundError(f"{path}: missing model_spec.json")

    model_spec = _read_json(spec_path)
    cfg = transform_config(model_spec)
    branch_sd = _lazy_branch_sd(branch_path)
    branches = _split_branches(path, branch_sd, cfg)

    adapters = {}
    adapters_root = os.path.join(path, "adapters")
    if os.path.isdir(adapters_root):
        for name in sorted(os.listdir(adapters_root)):
            directory = os.path.join(adapters_root, name)
            config_path = os.path.join(directory, "adapter_config.json")
            weights_path = os.path.join(directory, "adapter_model.safetensors")
            if not (os.path.isfile(config_path) and os.path.isfile(weights_path)):
                continue
            config = _read_json(config_path)
            state = _load_owned_safetensors(weights_path)
            _validate_adapter_weights(name, state, config, path)
            adapters[name] = (state, config)

    result = (cfg, branches, adapters)
    _CACHE[identity] = result
    _CACHE.move_to_end(identity)
    while len(_CACHE) > _MAX_CACHE:
        _CACHE.popitem(last=False)
    return result
