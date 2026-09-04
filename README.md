# ComfyUI-VDN-H3 — VDN-H3 for MiniMax-H3

<img width="1039" height="505" alt="VDN-H3" src="https://github.com/user-attachments/assets/ab4c1691-bff5-46fe-8b3e-635429b0700f" />

**[中文](README_ZH.md)**

A ComfyUI port of the released [OpenVDN VDN-H3](https://github.com/OpenVDN/vdn-minimax-h3) hybrid-attention architecture for ComfyUI's native MiniMax-H3 model.

VDN-H3 keeps exact softmax attention over a local frame window and uses a bidirectional Video Delta Attention branch for temporal context outside that window. This repository loads the released VDN stage directories directly and applies their branch weights and adapters without modifying ComfyUI core files.

## What this node preserves

The port follows the released checkpoint architecture, including:

- packed MiniMax-H3 text/video/audio layout;
- frame- or chunk-aligned softmax windows;
- anchor modes (`none`, `rows`, `columns`, `both`);
- shared raw pre-QK-norm / pre-RoPE Q, K and V for the linear branch;
- optional separable short convolution on configured Q/K/V targets;
- per-token beta, per-frame KDA alpha, and the checkpoint-selected delta rule;
- forward and reverse recurrent scans;
- optional text-state initialization and alpha boundary bridge;
- branch RMSNorm/output gate and `to_out_linear` readout;
- optional softmax output gate;
- exact dense-attention fallback when the configured window covers the full clip.

Architecture values come from the checkpoint's `model_spec.json` by default. The Advanced node can override selected values only after explicitly selecting `architecture_mode=override`; those settings are ablations and are not claimed to reproduce the trained checkpoint.

## Installation

Clone the node under `ComfyUI/custom_nodes/` and restart ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Saganaki22/ComfyUI-VDN-H3
```

Download an official VDN stage directory under `ComfyUI/models/vdn/`, preserving its directory structure:

```bash
hf download OpenVDN/vdn-minimax-h3 \
  --include "stage-dmd-step-250/*" \
  --local-dir <ComfyUI>/models/vdn
```

A stage is expected to contain its `model_spec.json`, `linear_branch/` weights and adapter directories exactly as released.

Official releases currently identify:

- `stage-dmd-step-250`: VDN-H3 8-step stage, including the Turbo/DMD adapter;
- `stage-b-step-2000`: VDN-H3 50-step stage with the Stage-B/default adapter.

The checkpoint/model-weight license is **not** Apache-2.0; see [Licensing and provenance](#licensing-and-provenance) before downloading or using model weights.

## Nodes

### Apply VDN-H3 (MiniMax-H3 Hybrid Attention)

`MODEL -> MODEL`

| Input | Meaning |
|---|---|
| `vdn_checkpoint` | Official stage directory under `models/vdn/` |
| `apply_turbo_adapter` | Apply the stage's released Turbo/DMD adapter when present; enabled for the released 8-step DMD stage |
| `strength` | Adapter strength; `1.0` is the released setting |
| `lora_mode` | `merge` only: adapters are registered through ComfyUI's native `ModelPatcher` weight-patch lifecycle |
| `branch_weights` | `stream` or `resident`, described below |
| `attention_backend` | `grouped` portable windowed SDPA, or opt-in `flex` with grouped fallback |
| `verbose` | Additional VDN layout/adapter logging |

The former VDN `bypass` LoRA mode has been removed. VDN no longer installs or repairs mutable `module.forward` LoRA bypass chains. This is intentional: adapter correctness must not depend on injection/ejection order, clone order, another wrapper provider, or Continuum chunk lifecycle.

### Apply VDN-H3 Advanced

Adds independent Stage-B/Turbo strengths, optional fast kernels and explicit architecture ablations.

`architecture_mode=checkpoint` is the default and ignores the ablation fields. Select `architecture_mode=override` before changing:

- `window_radius`;
- `window_chunk`;
- `anchor_frames`;
- `text_state`;
- `linear_branch`.

When override mode changes the checkpoint architecture, the node logs the difference.

`fast_kernels` optionally uses `torch.compile` for selected branch helpers. It preserves the same algorithm but can change BF16 rounding because fused kernels round at different operation boundaries. Compilation failure falls back to eager execution.

## Adapter lifecycle and quantized bases

Ordinary VDN LoRA targets are registered with ComfyUI's `ModelPatcher.add_patches` path. ComfyUI therefore owns weight backup/restore, clone behavior and load/offload transitions.

This matters for fused or quantized MiniMax-H3 modules: VDN does not dequantize and replace those modules itself. It delegates weight conversion/materialization to the same `convert_weight` / `set_weight` abstraction used by current ComfyUI. Synthetic CI coverage verifies this path and repeated restoration; real GPU validation is still required for each production quantization/layout combination.

The Q/K/V adapter conversion preserves independent LoRA ranks and `alpha/rank` scales by constructing an equivalent fused block-diagonal patch. Incomplete or malformed adapter targets fail during checkpoint/application validation instead of being silently skipped.

## Curve/pruned MiniMax-H3 bases

Some MiniMax-H3 checkpoints collapse the dense time embedding into `adaln_t_table`. A full-width learned AdaLN LoRA cannot in general be projected into that smaller curve basis exactly.

This node therefore does **not** drop those adapter weights and does not approximate them. For a curve/pruned base it reconstructs the matching dense time-embedder input and applies the original low-rank AdaLN delta at runtime while leaving the base curve projection intact.

To do that exactly, VDN needs the dense time embedder corresponding to the curve base. It resolves either:

1. `dense_time_embedder.safetensors` placed directly in the VDN stage directory; or
2. a matching installed dense MiniMax-H3 checkpoint under `models/diffusion_models`.

You can extract the small companion from a matching dense H3 checkpoint:

```bash
python tools/extract_h3_time_embedder.py \
  <path-to-dense-h3.safetensors> \
  <ComfyUI>/models/vdn/<stage>/dense_time_embedder.safetensors
```

If no compatible dense embedder can be established, application fails closed with an actionable error. It never silently omits learned AdaLN adapter parameters.

## Branch-weight residency

`branch_weights=stream`

- does not keep the complete VDN branch resident as an additional model;
- resolves each block from the stage file when needed;
- keeps safetensors mappings bounded to each load operation rather than retaining process-global mmap handles.

`branch_weights=resident`

- wraps the branch tensors in a separate Comfy-managed `ModelPatcher`;
- registers it as an additional model so Comfy owns device placement and load/offload lifecycle;
- avoids the old untracked process-global GPU branch cache.

Quantized VDN branch files currently require `stream`; resident mode fails closed rather than silently dequantizing them.

## Composition and lifecycle

VDN replaces `diffusion_model.blocks.*.attn.forward` through a Comfy object patch. Another extension that owns that exact object-patch target is incompatible; VDN refuses an existing conflicting owner rather than stacking ambiguous forward replacements.

Ordinary model weight patches/LoRAs use a different Comfy lifecycle and are not traversed or reordered by VDN.

The test suite includes repeated `ModelPatcher` clone/load/unload cycles and a pseudo-Continuum sequence that executes the conditioning path (`preprocess_text_embeds -> token_refiner.fc1`) before a transformer evaluation on every chunk. It verifies stable outputs, restored base weights, unchanged forwards and no 2x/3x adapter accumulation. That is structural CPU coverage; it is not a substitute for a real GPU Continuum render.

## Attention backends

- `grouped` is the portable default. Frames sharing the same VDN window are evaluated as grouped dense SDPA calls.
- `flex` uses PyTorch FlexAttention when available and suitable. Failure falls back to grouped execution for that call without mutating shared VDN state.
- full-coverage windows use ComfyUI's normal optimized dense-attention path and disable the linear complement because nothing lies outside the softmax window.

See [Benchmarks.md](Benchmarks.md) for historical measurements. Those measurements predate this lifecycle refactor unless explicitly marked otherwise and must not be treated as performance validation of the current branch.

## Validation

The repository CI has two distinct lanes:

1. **Pinned Comfy + official oracle**
   - ComfyUI `6c53f8c9a06d95f3d847009ceaae55c624169247`;
   - OpenVDN `b8cb28fbfca0266d1c7742a9f25ab8b58191de97`;
   - direct reduced-dimension CPU comparisons against imported OpenVDN source;
   - independent math, adapter conversion, ModelSpec/checkpoint, curve, quantized-patch and lifecycle regressions.
2. **Current Comfy main smoke**
   - checks current Comfy `master` for package import and node registration against the latest API surface.

The direct oracle covers window bounds/anchors, frame statistics, all supported delta rules, forward/reverse scans, alpha bridge, feature preparation/short-conv behavior and the complete `BidirectionalLinearBranch` under reduced CPU dimensions. The pinned suite currently contains 70 passing tests.

No large checkpoint or GPU render is run in CI. A green synthetic/oracle suite establishes implementation and lifecycle contracts; it does not establish real-render quality, GPU memory use or wall-clock performance.

## Compatibility requirements

- current ComfyUI MiniMax-H3 implementation with `diffusion_model.blocks[].attn.qkv_proj`;
- the official VDN v2 ModelSpec/hybrid-transform contract;
- stage/base block count and every enabled trained branch tensor shape must match;
- malformed, incomplete, unsupported or stale-replaced checkpoint resources fail early.

## Licensing and provenance

**Source code:** this repository is distributed under the Apache License 2.0. It originates from Saganaki22's ComfyUI-VDN-H3 implementation and ports/adapts the VDN-H3 architecture and algorithms released by OpenVDN. See `LICENSE` and `NOTICE`.

**OpenVDN:** the OpenVDN source repository is Apache-2.0. Its NOTICE states separately that VDN-H3 model weights are derivatives of MiniMax-H3 and are distributed under the MiniMax-H3 Community License Agreement.

**Model/checkpoint weights:** this repository does not relicense MiniMax-H3 or VDN-H3 weights. Downloading or using those weights remains subject to their applicable MiniMax-H3 license terms and eligibility restrictions.

Upstream/research sources:

- OpenVDN VDN-H3: https://github.com/OpenVDN/vdn-minimax-h3
- Original ComfyUI port: https://github.com/Saganaki22/ComfyUI-VDN-H3
- ComfyUI: https://github.com/Comfy-Org/ComfyUI
- MiniMax-H3 model release: https://huggingface.co/Comfy-Org/MiniMax-H3
- VDN-H3 model release: https://huggingface.co/OpenVDN/vdn-minimax-h3

## Historical media

Existing upstream example media and performance measurements are intentionally treated as historical evidence rather than revalidation of the refactored lifecycle path. The latest upstream Ref2V example is 928x928; see the original repository for its current media assets and presentation.