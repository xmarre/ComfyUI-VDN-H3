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
| `lora_mode` | `merge` or safe runtime `bypass`; see [LoRA adapter modes](#lora-adapter-modes) |
| `branch_weights` | `auto`, `stream` or `resident`; controls the **VDN linear branch**, not LoRA application |
| `retain_buffers` | `auto`, `on` or `off`; controls reusable VDN scratch, not model weights |
| `attention_backend` | `grouped` portable windowed SDPA, or opt-in `flex` with grouped fallback |
| `verbose` | Additional VDN layout/adapter logging |

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

## LoRA adapter modes

`lora_mode`, `branch_weights` and `retain_buffers` solve different memory problems and are intentionally independent.

### `lora_mode=merge`

- registers Stage-B/Turbo adapters through normal `ModelPatcher.add_patches()`;
- Comfy owns backup/restore, load/offload, custom-weight conversion and requantization;
- this remains the reference/eager adapter path and the conservative choice for output validation;
- on quantized bases, eager dequantize -> patch -> requantize can have a substantial temporary VRAM cost.

### `lora_mode=bypass` — safe runtime low-VRAM mode

The `bypass` name is preserved for existing workflows, but this is **not the old `BypassForwardHook` implementation**.

The current runtime mode:

- registers adapter work with Comfy's public `ModelPatcher.add_weight_wrapper()` / `weight_function` lifecycle;
- never replaces, traverses, splices or restores a LoRA target's `module.forward`;
- never installs a VDN LoRA `PatcherInjection`, `_vdn_live_hooks` chain or private forward owner;
- keeps the resident base parameter unmerged;
- stores LoRA factors in a separate Comfy-managed additional `ModelPatcher` rather than a private GPU cache;
- aggregates Stage-B/Turbo terms targeting the same base weight into one runtime wrapper;
- creates only the current layer's transient compute weight;
- evaluates `B @ A` in bounded output-row chunks, then applies scale and addition in merge-style operation order; the extra delta temporary is capped at 8 MiB rather than another full weight-sized tensor;
- preserves checkpoint factor storage dtype and uses Comfy's selected LoRA compute dtype for the invocation;
- assigns each distinct Apply execution a distinct managed-runtime ownership key, while clones of the same Apply result remain weight-equivalent.

This restores the low-VRAM adapter option without reintroducing the cross-provider forward-chain recursion that could crash Continuum before chunk 2 reached its first transformer evaluation.

For quantized/fused MiniMax-H3 modules, Comfy's cast path remains authoritative. A runtime weight wrapper can cause a patched INT8 layer to use a dequantized compute fallback for that invocation instead of the fused INT8 kernel. That trades speed for avoiding eager patched-weight materialization; measure it on the actual base and workflow rather than assuming either mode is universally cheaper.

**Output note:** historical VDN bypass measurements used a different activation-level forward-hook implementation and showed quality differences on the 8-step DMD stage. They do not establish the numerical behavior of this weight-level runtime mode. Until matched GPU renders are complete, use `merge` as the Stage-DMD reference path and treat runtime `bypass` as a low-VRAM path requiring real-render validation.

## Curve/pruned MiniMax-H3 bases

Some MiniMax-H3 checkpoints collapse the dense AdaLN timestep representation to a small coordinate table such as `adaln_t_table`. The supported pruned lineage is constructed with an affine approximation

```text
dense(t) ≈ mean + curve(t) @ basis
```

The released VDN Turbo adapter contains full-width learned AdaLN LoRAs. For one such update `B @ A`, VDN projects it once onto the native pruned coordinates as

```text
A_pruned   = A @ basis.T
bias_delta = B @ (A @ mean)
```

Both terms are required. The constant `bias_delta` is not optional and is never silently dropped.

The projection is performed once in float64 from the stored adapter and pruning-affine tensors. The projected A matrix and constant bias offset are stored as float32, while B retains its checkpoint storage dtype until Comfy selects the invocation compute dtype. This introduces **no additional model approximation beyond the pruned base's existing affine approximation**; it does not claim bitwise equivalence to the original unpruned dense timestep MLP.

Runtime behavior stays native to the pruned model:

- there is no reconstructed dense timestep MLP in the sampling loop;
- there is no AdaLN `forward` object patch;
- `merge` registers the projected low-rank weight term and constant bias as normal Comfy patches;
- `bypass` stores projected A/B plus the float32 bias offset in the same Comfy-managed additional model and applies them through Comfy weight/bias wrappers.

### Resolving the pruning affine

VDN must use the exact `adaln_basis` + `adaln_mean` pair corresponding to the loaded `adaln_t_table`; an unrelated basis is not interchangeable. Resolution is fail-closed and follows this order:

1. `adaln_affine.safetensors` in the selected VDN stage directory;
2. the selected diffusion-model checkpoint and sibling `.safetensors` files;
3. other installed `models/diffusion_models` candidates.

Installed checkpoint candidates must prove that their curve table matches the loaded base. A mismatched or unverified affine is rejected.

For the repaired pruned MiniMax-H3 Comfy lineage, the BF16 source checkpoint can retain `adaln_basis` and `adaln_mean`, while INT8 derivatives may intentionally omit those inference-unused auxiliaries. If the matching BF16 file is still beside the selected INT8/INT8-ConvRot file, VDN can resolve the affine automatically and reads only the tiny affine tensors/table from that file; it does not load the full BF16 model.

If you do not keep the BF16 sibling, extract the approximately 97 KB companion once:

```bash
python tools/extract_h3_adaln_affine.py \
  <path-to-matching-pruned-bf16.safetensors> \
  <ComfyUI>/models/vdn/<stage>/adaln_affine.safetensors
```

The extractor records a curve-table identity when the source contains the table. If no verified affine can be established, VDN fails with an actionable error rather than dropping the 51 released Turbo AdaLN updates or guessing a basis.

## VDN branch-weight residency

This is separate from `lora_mode`.

### `branch_weights=auto`

The automatic policy incorporates upstream v1.4's VRAM-aware intent while retaining this branch's stronger ownership rules. The memory budget is **effective free VRAM**: Comfy's current free-memory value minus any bytes of the supplied base `MODEL` that are not resident yet. This prevents an unloaded H3 base from being counted as free space available to VDN.

- if the ordinary branch fits the upstream `1.5 x branch size + 4 GiB` headroom rule in that effective budget, use `resident` BF16 branch weights;
- otherwise, if `model_int8_convrot_comfyui.safetensors` exists, select it and use `stream`;
- otherwise stream the ordinary branch.

The INT8 branch deliberately remains streamed under auto mode. `resident` means a real Comfy-managed parameter tree here; quantized branch tensors are not silently copied into an untracked VDN GPU cache.

### `branch_weights=stream`

- does not keep the complete VDN linear branch resident as an additional model;
- resolves each block from the selected stage file with a bounded `safe_open` lifetime;
- never keeps a process-global safetensors mmap handle;
- when retained runtime buffers are enabled on CUDA, uses one-block lookahead transfer through one bounded worker executor; the executor owns no model tensor cache and each VDN state owns at most one cancellable in-flight result;
- prefetched results are keyed by block index, full device identity and compute dtype, so a lookahead cannot be reused after a placement/dtype change.

### `branch_weights=resident`

- wraps ordinary branch tensors in a separate Comfy-managed `ModelPatcher`;
- registers it as an additional model so Comfy owns device placement and load/offload lifecycle;
- avoids the old untracked process-global GPU branch cache.

Quantized VDN branch files currently require `stream`; explicit resident mode fails closed rather than silently dequantizing them.

A minimum-residency setup can combine `lora_mode=bypass` with `branch_weights=stream`: the first controls adapter materialization, while the second controls VDN linear-branch residency.

## Retained runtime buffers

Upstream v1.4 demonstrated that repeated scratch allocation can cost meaningful time. This branch incorporates that optimization without adopting process-global GPU scratch banks.

`retain_buffers=on` reuses state-owned scratch for:

- raw video/text Q/K/V copies used by the linear complement;
- forward/reverse recurrence banks;
- grouped-window row-index plans;
- grouped-window K/V gather storage;
- the one-block stream prefetch state.

Ownership rules are explicit:

- one retained pool belongs to one `VDNState` / Apply result;
- the pool is leased for the duration of a diffusion-model execution;
- nested or concurrent executions that cannot acquire that lease receive isolated transient scratch instead of racing the retained tensors;
- large retained categories keep only the most recent geometry; small index plans are bounded separately;
- cancellation clears that state's retained scratch/prefetch state;
- branch weights and LoRA factors are **not** stored in this scratch pool.

`retain_buffers=off` uses transient allocation behavior. `auto` applies the selected branch size + 10 GiB headroom rule to the same effective free-VRAM budget after reserving still-unloaded base-model bytes.

CPU tests require retained and transient scan/window/complete-linear-branch paths to match the reference path exactly. Real CUDA allocation and speed still require production GPU validation.

## Composition and lifecycle

VDN replaces `diffusion_model.blocks.*.attn.forward` for the **VDN hybrid-attention transform** through a Comfy object patch. Another extension that owns that exact attention object-patch target is incompatible; VDN refuses an existing conflicting owner rather than stacking ambiguous attention replacements.

That attention ownership is distinct from LoRA runtime mode. `lora_mode=bypass` does **not** patch any LoRA target's `module.forward`; it uses weight/bias wrappers instead. Ordinary model weight patches/LoRAs remain under Comfy's weight lifecycle and are not traversed or reordered by VDN.

The test suite includes repeated `ModelPatcher` clone/load/unload cycles and a pseudo-Continuum sequence that executes the conditioning path (`preprocess_text_embeds -> token_refiner.fc1`) before a transformer evaluation on every chunk. Runtime-mode regressions additionally require:

- no VDN LoRA injections;
- unchanged `module.forward` ownership;
- unchanged resident base weights;
- one aggregate runtime wrapper per target weight;
- projected curve AdaLN constant terms applied through bias wrappers rather than a forward patch;
- stable output across repeated clones;
- no 2x/3x adapter accumulation;
- independent strength changes from the true base;
- different Apply configurations to be non-equivalent to Comfy model management while clones of one Apply remain equivalent.

That is structural CPU coverage; it is not a substitute for a real GPU Continuum render or VRAM measurement.

## Attention backends

- `grouped` is the portable default. Frames sharing the same VDN window are evaluated as grouped dense SDPA calls. Transformer options are forwarded to Comfy's optimized-attention call so normal Comfy composition remains available.
- `flex` uses PyTorch FlexAttention when available and suitable. Failure falls back to grouped execution for that call without mutating shared VDN state. Its process-level BlockMask cache is a bounded 8-entry LRU keyed by full device and layout identity.
- full-coverage windows use ComfyUI's normal optimized dense-attention path and disable the linear complement because nothing lies outside the softmax window.

See [Benchmarks.md](Benchmarks.md) for historical measurements. Measurements that predate this lifecycle/runtime refactor are not performance validation of the current branch.

## Upstream v1.4 reconciliation

During this correctness PR, the original Comfy port advanced to v1.4.0 with faster streaming and VRAM-aware buffer retention. Those goals are incorporated here, but not by copying its resource implementation verbatim.

Retained from v1.4's intent:

- automatic branch placement;
- optional INT8 ConvRot branch selection under pressure;
- one-block stream lookahead;
- reusable scan/window/activation scratch;
- automatic scratch-retention headroom policy.

Deliberately replaced in this branch:

- persistent process-global safetensors handles -> bounded file opens with file-identity invalidation;
- private process-global GPU branch cache -> Comfy-managed resident branch model or streaming;
- process-global CUDA scan/KV scratch -> per-VDN execution-leased scratch;
- per-state immortal prefetch worker -> one bounded tensor-less executor plus at most one state-owned in-flight future;
- unbounded/device-type-only Flex BlockMask cache -> bounded 8-entry LRU keyed by full device/layout identity.

This means upstream v1.4 benchmark numbers cannot simply be assigned to this implementation. The optimized lifecycle needs its own real GPU measurements.

## Validation

The repository CI has two distinct lanes:

1. **Pinned Comfy + official oracle**
   - ComfyUI `6c53f8c9a06d95f3d847009ceaae55c624169247`;
   - OpenVDN `b8cb28fbfca0266d1c7742a9f25ab8b58191de97`;
   - direct reduced-dimension CPU comparisons against imported OpenVDN source;
   - direct instantiation/comparison of the released OpenVDN `HybridAttention` orchestration;
   - adapter conversion, ModelSpec/checkpoint, curve-affine, quantized/custom-weight, runtime-buffer, placement-policy and lifecycle regressions.
2. **Current Comfy main smoke**
   - checks current Comfy `master` for package import and node registration against the latest API surface.

The pinned CI suite includes the production-shaped KJ selected-INT8 + matching-BF16-sibling affine resolver regression; see the CI run for the current test count. The official oracle covers window bounds/anchors, frame statistics, all supported delta rules, forward/reverse scans, alpha bridge, feature preparation/short-conv behavior, the complete `BidirectionalLinearBranch`, and the complete reduced `HybridAttention` local-softmax + recurrent-linear orchestration. Separate parity/lifecycle tests cover retained vs transient execution, base-residency-aware placement, prefetch identity, bounded Flex cache, fused adapter naming, curve affine projection including constant bias, wrong-table rejection and file replacement invalidation.

No large checkpoint or GPU render is run in CI. A green synthetic/oracle suite establishes implementation and lifecycle contracts; it does not establish real-render quality, peak VRAM or wall-clock performance.

## Compatibility requirements

- current ComfyUI MiniMax-H3 implementation with fused `diffusion_model.blocks[].attn.qkv_proj`;
- runtime `bypass` additionally requires target Comfy modules to expose the supported `weight_function` contract; projected curve AdaLN bias targets also require `bias_function`;
- curve/pruned bases with full-width released AdaLN adapters require the verified pruning affine (`adaln_basis` + `adaln_mean`) matching the loaded `adaln_t_table`;
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

Existing upstream example media and historical performance measurements are evidence for their specific revisions, not revalidation of this refactored lifecycle path. The latest upstream Ref2V example is 928x928; see the original repository for its current media assets and presentation.