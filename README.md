# ComfyUI-VDN-H3

A ComfyUI port of the released [OpenVDN VDN-H3](https://github.com/OpenVDN/vdn-minimax-h3) hybrid-attention architecture for ComfyUI's native MiniMax-H3 model.

This xmarre fork keeps the released VDN checkpoint/math contract while adding current pruned/INT8 H3 support, stricter Comfy lifecycle handling, and the external mixed-grid sequence contract used by [MiniMax-H3 Flow-Aligned Regenerate](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate).

> The VDN model weights are separate from this repository and retain their upstream license. See [NOTICE](NOTICE) for implementation provenance and attribution.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xmarre/ComfyUI-VDN-H3.git
```

Restart ComfyUI.

Download an official VDN stage under `ComfyUI/models/vdn/` while preserving its directory structure, for example:

```bash
hf download OpenVDN/vdn-minimax-h3 \
  --include "stage-dmd-step-250/*" \
  --local-dir <ComfyUI>/models/vdn
```

Official stages include:

- `stage-dmd-step-250` — released 8-step DMD/Turbo stage;
- `stage-b-step-2000` — released Stage-B/default stage.

## Nodes

### Apply VDN-H3

| Input | Meaning |
|---|---|
| `vdn_checkpoint` | Stage directory under `models/vdn/` |
| `apply_turbo_adapter` | Apply the released Turbo/DMD adapter when the stage provides it |
| `strength` | Adapter strength; `1.0` is the released setting |
| `lora_mode` | `merge` or stack-safe runtime `bypass` |
| `branch_weights` | `auto`, `stream`, or `resident` |
| `retain_buffers` | `auto`, `on`, or `off` |
| `attention_backend` | `grouped` or opt-in `flex` |
| `verbose` | Additional runtime logging |

### Apply VDN-H3 Advanced

Adds independent Stage-B/Turbo strengths, optional compiled helpers, and explicit architecture ablations.

`architecture_mode=checkpoint` is the default for newly created nodes and uses `model_spec.json` exactly. Select `architecture_mode=override` only when intentionally changing `window_radius`, `window_chunk`, `anchor_frames`, `text_state`, or `linear_branch`.

## Adapter modes

### `lora_mode=merge`

Uses ordinary Comfy `ModelPatcher.add_patches()` weight ownership. This remains the conservative reference path for matched output validation.

### `lora_mode=bypass`

v1.5.1 uses Comfy's activation-side `BypassForwardHook` mechanism for ordinary VDN LoRA targets, with an additional VDN lifecycle layer that makes independently owned runtime adapter providers safe to stack.

The important properties are:

- VDN does not assume it is the only runtime-bypass provider on a module;
- if another Comfy bypass hook is already active, VDN inserts underneath it rather than blindly becoming the outermost owner;
- teardown can splice VDN out of the middle of the live chain without restoring a stale `module.forward`;
- the same logic works whether VDN or the external provider was registered first;
- rerunning Apply VDN replaces the previous live VDN hook set on the clone-shared inner model instead of accumulating another adapter delta;
- pre-existing cyclic bypass chains fail closed;
- fused INT8 `mlp.fc2` targets that do not call `module.forward` stay on normal Comfy weight patches.

### Why v1.5.1 changed bypass again

v1.5.0 temporarily implemented bypass with `ModelPatcher.add_weight_wrapper()` / `weight_function` and a managed A/B-factor model. On quantized H3, a weight function changes execution onto Comfy's copied/dequantized compute-weight path.

A production RTX PRO 6000 run combining the INT8 ConvRot H3 base, VDN bypass, a second runtime-bypass DoRA/LoRA provider, `cudaMallocAsync`, Flow Mixed-Grid, Continuum and Spectrum/SA-PECE hard-aborted on the first actual H3 evaluation with a CUDA illegal memory access. The fatal stack was inside the external Comfy bypass-LoRA chain. Because CUDA reporting is asynchronous, the stack does not by itself identify the exact originating kernel, but the v1.5.0 weight-wrapper execution path was the new regression boundary.

The older stack-safe activation-hook implementation had already been validated with the same cross-provider lifecycle, so v1.5.1 restores that proven execution architecture rather than trying to patch around the quantized weight-wrapper interaction.

The v1.5.0 wrapper path is **not** installed by `lora_mode=bypass` in v1.5.1.

## Pruned / curve MiniMax-H3 bases

Supported pruned H3 checkpoints represent the original dense AdaLN timestep field approximately as:

```text
dense(t) ≈ mean + curve(t) @ basis
```

Released VDN Turbo adapters contain full-width AdaLN LoRAs. For an update `B @ A`, this fork projects it once into native curve coordinates:

```text
A_pruned   = A @ basis.T
bias_delta = B @ (A @ mean)
```

Both terms are required. VDN must resolve the matching `adaln_basis` + `adaln_mean` pair and fails closed rather than guessing or dropping incompatible AdaLN updates.

In v1.5.1, the projected curve weight and constant bias terms use ordinary Comfy weight/bias patches in both adapter modes. They do not use the activation-side VDN hooks and do not use the superseded v1.5.0 weight-wrapper path.

If a matching BF16 source checkpoint remains beside an INT8 derivative, VDN can read only the small affine tensors from it. Otherwise use:

```bash
python tools/extract_h3_adaln_affine.py \
  <matching-pruned-bf16.safetensors> \
  <ComfyUI>/models/vdn/<stage>/adaln_affine.safetensors
```

## Branch weights and retained buffers

`branch_weights` controls the VDN linear branch and is independent from `lora_mode`.

- `auto` reserves still-unloaded H3 base-model bytes before deciding what VDN may keep resident. It uses resident BF16 branch weights when headroom is sufficient; otherwise it streams and prefers the native INT8 ConvRot branch when available.
- `stream` resolves one block at a time and can use one-block lookahead when retained buffers are enabled.
- `resident` registers ordinary BF16 branch weights as a Comfy-managed additional `ModelPatcher`.

`retain_buffers=on` reuses execution-owned scan/window/activation scratch. The retained pool belongs to one VDN state and is leased for one diffusion-model execution; nested/concurrent runs that cannot acquire it use isolated transient scratch.

Under `cudaMallocAsync`, explicit `record_stream` bookkeeping is skipped because the allocator is already stream ordered. This incorporates upstream v1.4.3's torch-warning fix in this fork's state-owned prefetch implementation.

## Flow-Aligned Regenerate interoperability

VDN supports the external-sequence contracts used by MiniMax-H3 Flow-Aligned Regenerate:

- API 1 target-sparse compatibility;
- API 2 `mixed_grid_low_suffix` for a target-grid protected prefix plus a genuine low-grid generated suffix.

During an API-2 mixed sequence, VDN keeps the learned dense softmax gate active and disables only geometry-dependent local-window/linear-complement work that cannot be interpreted on the mixed lattice. The fresh target-grid stage returns to normal VDN execution automatically.

See [docs/MIXED_SEQUENCE_API.md](docs/MIXED_SEQUENCE_API.md) for the exact fail-closed contract.

## Legacy Advanced-node workflow compatibility

The original Advanced node had 14 positional widgets. v1.5 added `retain_buffers` and `architecture_mode`, which would shift values in workflows saved before ComfyUI emitted `widgets_values_named`.

The included frontend migration:

- publishes the original 14 names through `fallbackWidgetsValuesNames`;
- restores old positional values by name;
- inserts `retain_buffers="auto"`;
- inserts `architecture_mode="override"` for old workflows because those architecture fields historically applied unconditionally;
- preserves the short-lived 16-value positional layout without changing its semantics;
- leaves workflows that already contain `widgets_values_named` untouched.

Newly created nodes still default to `architecture_mode="checkpoint"`.

## Compatibility notes

- `grouped` is the portable attention backend and remains the default. `flex` is opt-in and falls back to grouped if unavailable.
- VDN owns the H3 attention object patch. Another extension that tries to own the same `diffusion_model.blocks.*.attn.forward` target is rejected rather than ambiguously stacked.
- Runtime LoRA/DoRA providers using Comfy's ordinary `BypassForwardHook` mechanism may coexist with VDN bypass; the cross-provider chain is regression-tested in both insertion orders.
- The AIMDO malloc-graph compatibility guard is scoped only around VDN diffusion-model execution and restores the user's compiler setting afterward.
- Historical benchmark numbers in [Benchmarks.md](Benchmarks.md) predate the current lifecycle work and are not assigned to this runtime without matched measurement.

## Validation

v1.5.1 adds explicit regression coverage for:

- repeated VDN hook injection/ejection;
- VDN-first and external-provider-first stacked bypass providers;
- live VDN replacement while an external runtime provider stays active;
- cyclic-chain rejection;
- zero `weight_wrapper_patches` in the active bypass path;
- projected pruned-AdaLN math through native patches;
- current ComfyUI import/registration and the existing OpenVDN numerical/oracle suite.

The API-2 mixed-grid path remains the VDN contract used by the production Flow-Aligned Regenerate Continuum workflow.

## Upstream and licensing

- OpenVDN reference implementation: https://github.com/OpenVDN/vdn-minimax-h3
- Original ComfyUI port: https://github.com/Saganaki22/ComfyUI-VDN-H3
- This maintained fork: https://github.com/xmarre/ComfyUI-VDN-H3

Repository code is distributed under [Apache-2.0](LICENSE). Third-party model weights and repositories retain their own licenses. See [NOTICE](NOTICE) for attribution details.
