from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, got {count}")
    return text.replace(old, new, 1)


def replace_section(text: str, start: str, end: str, replacement: str, label: str) -> str:
    s = text.find(start)
    if s < 0:
        raise RuntimeError(f"{label}: start marker not found")
    e = text.find(end, s)
    if e < 0:
        raise RuntimeError(f"{label}: end marker not found")
    return text[:s] + replacement.rstrip() + "\n\n" + text[e:]


# --- apply.py report wording -------------------------------------------------
path = ROOT / "vdn_h3" / "apply.py"
text = path.read_text()
text = replace_once(
    text,
    "            # Kept for one release as a compatibility/logging alias. It no longer\n            # means ModelPatcher weight wrappers in v1.5.1.\n",
    "            # Kept as a compatibility/logging alias. In v1.5.2 it counts\n            # non-mutating runtime adapter terms, not ModelPatcher weight wrappers.\n",
    "apply compatibility comment",
)
text = replace_once(
    text,
    '            "stack_safe_cross_provider": True,\n',
    '            "stack_safe_cross_provider": True,\n            "cross_provider_forward_chain_independent": True,\n',
    "apply runtime report",
)
path.write_text(text)

# --- runtime report regression ----------------------------------------------
path = ROOT / "tests" / "test_runtime_lowvram.py"
text = path.read_text()
text = replace_once(
    text,
    '    assert runtime["stack_safe_cross_provider"] is True\n',
    '    assert runtime["stack_safe_cross_provider"] is True\n    assert runtime["cross_provider_forward_chain_independent"] is True\n',
    "runtime report test",
)
path.write_text(text)

# --- node UI ----------------------------------------------------------------
path = ROOT / "vdn_h3" / "nodes.py"
text = path.read_text()
text = replace_once(
    text,
    '''                "tooltip": "merge uses normal Comfy weight patches. bypass uses "\n                           "stack-safe Comfy BypassForwardHook adapters for ordinary "\n                           "targets, preserving native quantized forwards while safely "\n                           "stacking with other runtime bypass providers."}),''',
    '''                "tooltip": "merge uses normal Comfy weight patches. bypass uses "\n                           "VDN-owned PyTorch forward post-hooks for ordinary LoRA "\n                           "targets: module.forward is never replaced or spliced, and "\n                           "other runtime-bypass providers keep independent ownership."}),''',
    "basic bypass tooltip",
)
text = replace_once(
    text,
    '''                "tooltip": "bypass uses VDN's stack-safe Comfy bypass-hook lifecycle "\n                           "for ordinary LoRA targets; projected curve AdaLN remains "\n                           "under normal native weight/bias patches."}),''',
    '''                "tooltip": "bypass uses VDN-owned PyTorch forward post-hooks and "\n                           "never replaces module.forward; fused INT8 fc2 and projected "\n                           "curve AdaLN terms remain native Comfy patches."}),''',
    "advanced bypass tooltip",
)
path.write_text(text)

# --- README -----------------------------------------------------------------
path = ROOT / "README.md"
text = path.read_text()
text = text.replace(
    "| `lora_mode` | `merge` or stack-safe runtime `bypass` |",
    "| `lora_mode` | `merge` or non-mutating runtime `bypass` |",
)
new_adapter = '''## Adapter modes

### `lora_mode=merge`

Uses ordinary Comfy `ModelPatcher.add_patches()` weight ownership. This remains the conservative reference path for matched output validation.

### `lora_mode=bypass`

v1.5.2 keeps ordinary VDN LoRA terms off both Comfy weight wrappers and mutable `module.forward` bypass chains. Each affected module receives one VDN-owned PyTorch **forward post-hook**. The module's normal forward executes first, including any independently managed provider that chooses to wrap it, and VDN then adds its exact low-rank residual to the returned tensor.

The important properties are:

- VDN never replaces, splices, saves, or restores `module.forward`;
- VDN does not install `ModelPatcher.add_weight_wrapper()` / `weight_function` callbacks;
- all active VDN terms for one module are combined into one exact low-rank residual, avoiding nested VDN hook chains;
- post-hook handles are owned by one `PatcherInjection` generation on the clone-shared inner model;
- applying a newer VDN clone replaces the previous VDN post-hook registration instead of accumulating deltas;
- ejecting an older/stale clone cannot remove the newer registration;
- another provider using Comfy `BypassForwardHook` remains completely outside VDN's ownership and its forward chain is never rewritten by VDN;
- fused INT8 `mlp.fc2` targets whose H3 fast path bypasses `module.forward` remain ordinary Comfy weight patches.

### Why v1.5.2 changes bypass

v1.5.0 used `ModelPatcher.add_weight_wrapper()` / `weight_function`. On quantized H3 this changes execution onto Comfy's copied/dequantized compute-weight path, and the production stacked-adapter workflow hard-aborted with a CUDA illegal-memory-access error.

v1.5.1 removed those wrappers and restored the older `BypassForwardHook` implementation, including custom linked-list splicing so VDN could coexist with another runtime-bypass provider. The same real RTX PRO 6000 workflow nevertheless still hard-aborted at the first actual H3 evaluation. In that run VDN correctly reported zero weight/bias wrappers, while the fatal Python stack surfaced inside core `LoRAAdapter.h` / `BypassForwardHook` execution. CUDA failure reporting is asynchronous, so that stack does **not** prove which kernel originally faulted and this repository does not attribute the crash to the other provider.

v1.5.2 instead removes VDN from the shared mutable-forward topology entirely. The independent provider may keep using Comfy's bypass mechanism; VDN observes only the completed module call and adds its own residual through a standard PyTorch post-hook. This is a structural isolation change, not a claim that a particular CUDA kernel has been identified.

The v1.5.0 weight-wrapper path and the v1.5.1 VDN `BypassForwardHook` chain are both absent from the active `lora_mode=bypass` path in v1.5.2.
'''
text = replace_section(text, "## Adapter modes", "## Pruned / curve MiniMax-H3 bases", new_adapter, "README adapter modes")
text = text.replace(
    "In v1.5.1, the projected curve weight and constant bias terms use ordinary Comfy weight/bias patches in both adapter modes. They do not use the activation-side VDN hooks and do not use the superseded v1.5.0 weight-wrapper path.",
    "In v1.5.2, the projected curve weight and constant bias terms use ordinary Comfy weight/bias patches in both adapter modes. They do not use the VDN post-hook path and do not use the superseded v1.5.0 weight-wrapper path.",
)
text = text.replace(
    "- Runtime LoRA/DoRA providers using Comfy's ordinary `BypassForwardHook` mechanism may coexist with VDN bypass; the cross-provider chain is regression-tested in both insertion orders.",
    "- Runtime LoRA/DoRA providers using Comfy's ordinary `BypassForwardHook` mechanism may coexist with VDN bypass. VDN does not join or rewrite that provider's forward chain; coexistence is regression-tested in both insertion orders.",
)
new_validation = '''## Validation

v1.5.2 adds explicit regression coverage for:

- exact VDN post-forward LoRA residual math while `module.forward` remains byte-for-byte owned by its original/external provider;
- VDN-first and external-provider-first coexistence with a real Comfy `BypassForwardHook`;
- VDN removal while the external provider remains live;
- clone-shared VDN replacement and stale-clone eject without accumulation;
- repeated pseudo-Continuum chunk injection/ejection;
- zero `weight_wrapper_patches` and zero VDN mutable-forward wrappers in active bypass mode;
- custom/quantized-like modules retaining their native weight path;
- projected pruned-AdaLN math through native patches;
- current ComfyUI import/registration and the existing OpenVDN numerical/oracle suite.

CPU/oracle CI validates those ownership and numerical contracts. The API-2 mixed-grid path remains the VDN contract used by the production Flow-Aligned Regenerate Continuum workflow. The specific CUDA illegal-address regression still requires the real stacked RTX PRO 6000 workflow as the acceptance gate before v1.5.2 is released.
'''
text = replace_section(text, "## Validation", "## Upstream and licensing", new_validation, "README validation")
path.write_text(text)

# --- Benchmarks/current-runtime contract ------------------------------------
path = ROOT / "Benchmarks.md"
text = path.read_text()
text = text.replace(
    "**Historical numbers are not performance validation of v1.5.1.**",
    "**Historical numbers are not performance validation of v1.5.2.**",
)
new_current = '''## Current v1.5.2 execution differences

Compared with the older measurements below:

- `lora_mode=bypass` uses one VDN-owned PyTorch forward post-hook per affected ordinary module;
- VDN never replaces or splices `module.forward`, so another Comfy runtime-bypass provider retains independent ownership of its own forward chain;
- the v1.5.0 `add_weight_wrapper()` / `weight_function` adapter path is not active;
- the v1.5.1 VDN `BypassForwardHook` linked-list/splicing path is not active;
- all VDN LoRA terms for one module are combined into one exact low-rank residual at runtime;
- fused INT8 `mlp.fc2` targets that do not call `module.forward` remain ordinary Comfy weight patches;
- full-width curve/pruned AdaLN adapters are projected through the exact pruning affine and applied as native curve weight + bias patches;
- the old private GPU branch cache remains replaced by bounded streaming or a Comfy-managed additional `ModelPatcher`;
- selected upstream v1.4 performance work remains under state-owned lifecycle rules: VRAM-aware branch selection, native INT8 ConvRot streaming under pressure, execution-leased retained scratch, and one-block streaming prefetch;
- `auto` placement reserves still-unloaded base-model bytes before assigning VDN residency.

New matched GPU measurements are required before assigning speed or VRAM numbers to v1.5.2.
'''
text = replace_section(text, "## Current v1.5.1 execution differences", "## Historical RTX 5090 measurements", new_current, "bench current")
text = text.replace(
    "These measurements predate the v1.5/v1.5.1 lifecycle work.",
    "These measurements predate the v1.5-v1.5.2 lifecycle work.",
)
new_bypass = '''### `lora_mode=bypass`

Ordinary adapter targets keep the resident base parameter untouched. VDN registers a standard PyTorch forward post-hook and adds the exact low-rank residual after the module returns. VDN does not replace `module.forward`, does not enter another provider's `BypassForwardHook` chain, and does not install a Comfy `weight_function` wrapper.

The v1.5.0 weight-wrapper design and the v1.5.1 mutable-forward VDN chain both failed the real stacked INT8 ConvRot H3 + external runtime-DoRA + `cudaMallocAsync` workflow at the first H3 evaluation. The latest fatal stack surfaced in core bypass-LoRA execution, but CUDA reporting is asynchronous and therefore does not identify the originating kernel. v1.5.2 treats this as an ownership/topology problem and structurally removes VDN from the shared mutable-forward chain rather than attributing the fault to a specific external hook or CUDA kernel.
'''
text = replace_section(text, "### `lora_mode=bypass`", "## Curve/pruned AdaLN projection", new_bypass, "bench bypass")
text = text.replace(
    "The constant term is required. v1.5.1 registers the projected native curve weight and bias terms as ordinary Comfy patches in both adapter modes; there is no reconstructed dense timestep MLP and no runtime weight wrapper for these terms.",
    "The constant term is required. v1.5.2 registers the projected native curve weight and bias terms as ordinary Comfy patches in both adapter modes; there is no reconstructed dense timestep MLP and no runtime weight wrapper for these terms.",
)
text = text.replace(
    "- stack-safe bypass lifecycle tests across independent providers and repeated cycles;",
    "- non-mutating post-forward bypass lifecycle tests across independent providers and repeated cycles;",
)
path.write_text(text)

# --- upstream reconciliation -------------------------------------------------
path = ROOT / "docs" / "UPSTREAM_RECONCILIATION.md"
text = path.read_text()
new_lifecycle = '''## Adapter lifecycle: v1.5.2

`merge` remains normal `ModelPatcher.add_patches()` ownership.

For ordinary LoRA targets, `lora_mode=bypass` uses VDN-owned PyTorch forward post-hooks. VDN intentionally does not use either of the two adapter-execution topologies that failed the real stacked quantized workflow:

- no `weight_function` / `add_weight_wrapper` execution from v1.5.0;
- no VDN `BypassForwardHook` linked list or custom `module.forward` splicing from v1.5.1.

Instead:

- the existing module forward executes normally;
- any independently managed runtime provider keeps sole ownership of whatever forward wrapper chain it installs;
- VDN adds its exact low-rank residual from a PyTorch post-hook after that call returns;
- all VDN terms for one module are combined into one post-hook residual;
- one generation-owned `PatcherInjection` registers/removes the handles;
- a newer clone-shared VDN generation replaces the older registration;
- stale older ejects cannot remove the newer generation;
- fused INT8 `mlp.fc2` targets that bypass `module.forward` remain ordinary weight patches.

The v1.5.1 real-workflow failure was asynchronously reported at core `LoRAAdapter.h` / `BypassForwardHook` during the first H3 evaluation. That stack is not sufficient to identify the originating CUDA kernel or to blame the independent provider. v1.5.2 therefore narrows the fix to the VDN-owned topology: VDN no longer participates in the mutable forward chain at all.

Full-width released AdaLN LoRAs on supported pruned/curve H3 bases are still projected through the exact pruning affine, including the required constant bias term. The already-native projected curve weight/bias terms use normal Comfy patches in both adapter modes.
'''
text = replace_section(text, "## Adapter lifecycle: v1.5.1", "## Flow-Aligned external sequence", new_lifecycle, "upstream lifecycle")
path.write_text(text)

print("updated v1.5.2 docs, UI, and runtime-report contract")
