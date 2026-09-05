# ComfyUI-VDN-H3

A ComfyUI port of the released [OpenVDN VDN-H3](https://github.com/OpenVDN/vdn-minimax-h3) hybrid-attention architecture for ComfyUI's native MiniMax-H3 model.

This xmarre fork keeps the released VDN math/checkpoint contract while tightening ComfyUI lifecycle ownership, supporting current pruned/INT8 H3 bases, and adding the external mixed-grid sequence contract used by [MiniMax-H3 Flow-Aligned Regenerate](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate).

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

The normal node exposes the released checkpoint with the main runtime controls:

| Input | Meaning |
|---|---|
| `vdn_checkpoint` | Stage directory under `models/vdn/` |
| `apply_turbo_adapter` | Apply the released Turbo/DMD adapter when the stage provides it |
| `strength` | Adapter strength; `1.0` is the released setting |
| `lora_mode` | `merge` or Comfy-managed runtime `bypass` |
| `branch_weights` | `auto`, `stream`, or `resident` |
| `retain_buffers` | `auto`, `on`, or `off` |
| `attention_backend` | `grouped` or opt-in `flex` |
| `verbose` | Additional runtime logging |

### Apply VDN-H3 Advanced

Adds independent Stage-B/Turbo strengths, optional compiled helpers, and explicit architecture ablations.

`architecture_mode=checkpoint` is the default for newly created nodes and uses `model_spec.json` exactly. Select `architecture_mode=override` only when intentionally changing `window_radius`, `window_chunk`, `anchor_frames`, `text_state`, or `linear_branch`.

## Adapter modes

### `lora_mode=merge`

Uses ordinary Comfy `ModelPatcher.add_patches()` ownership. This remains the conservative reference path for matched output validation.

### `lora_mode=bypass`

The workflow-facing name is retained, but the old forward-hook bypass implementation is gone.

The current runtime path:

- uses Comfy's public weight/bias wrapper lifecycle;
- does not replace or traverse LoRA target `module.forward` methods;
- leaves resident base parameters unmerged;
- stores LoRA factors as Comfy-managed additional-model state;
- bounds low-rank delta temporaries;
- keeps separate Apply executions distinct while normal model clones remain equivalent.

On fused/quantized H3 layers, a runtime weight wrapper can make Comfy use a dequantized compute path for that invocation. Treat that as a memory/compute tradeoff rather than a universal speed claim.

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

Both terms are required. The projection is fail-closed: VDN must resolve the matching `adaln_basis` + `adaln_mean` pair for the loaded curve table and will not silently guess or drop unsupported AdaLN updates.

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
- `resident` registers ordinary BF16 branch weights as a real Comfy-managed additional `ModelPatcher`.

`retain_buffers=on` reuses execution-owned scan/window/activation scratch. The retained pool belongs to one VDN state and is leased for one diffusion-model execution; nested/concurrent runs that cannot acquire it use isolated transient scratch.

Under `cudaMallocAsync`, explicit `record_stream` bookkeeping is skipped because the allocator is stream ordered already. This incorporates upstream v1.4.3's torch-warning fix in this fork's state-owned prefetch implementation.

## Flow-Aligned Regenerate interoperability

VDN supports two external-sequence contracts used by MiniMax-H3 Flow-Aligned Regenerate:

- API 1 target-sparse compatibility;
- API 2 `mixed_grid_low_suffix` for a target-grid protected prefix plus a genuine low-grid generated suffix.

During an external mixed sequence VDN keeps the learned dense softmax gate active and disables only geometry-dependent local-window/linear-complement work that cannot be interpreted on the mixed lattice. The fresh target-grid stage returns to normal VDN execution automatically.

See [docs/MIXED_SEQUENCE_API.md](docs/MIXED_SEQUENCE_API.md) for the exact fail-closed contract.

## Legacy Advanced-node workflow compatibility

v1.5.0 fixes old `ApplyVDNH3Advanced` workflows that were saved before ComfyUI serialized widget values by name.

The original Advanced node had 14 positional widgets. PR #2 added `retain_buffers` and `architecture_mode`, which shifted old arrays and could restore values such as `"both"` into `window_radius`.

The included frontend migration now:

- publishes the original 14 names through `fallbackWidgetsValuesNames`;
- restores old positional values by name;
- inserts `retain_buffers="auto"`;
- inserts `architecture_mode="override"` for old workflows, preserving their historical behavior because those architecture fields used to apply unconditionally;
- preserves the short-lived 16-value positional layout without changing its semantics;
- leaves workflows that already contain `widgets_values_named` untouched.

Newly created nodes still default to `architecture_mode="checkpoint"`.

## Compatibility notes

- `grouped` is the portable attention backend and remains the default. `flex` is opt-in and falls back to grouped if unavailable.
- VDN owns the H3 attention object patch. Another extension that tries to own the same `diffusion_model.blocks.*.attn.forward` target is incompatible and is rejected rather than ambiguously stacked.
- Ordinary model weight patches/LoRAs remain under Comfy's normal weight lifecycle.
- The AIMDO malloc-graph compatibility guard is scoped only around VDN diffusion-model execution and restores the user's compiler setting afterward.
- Historical benchmark numbers in [Benchmarks.md](Benchmarks.md) predate major lifecycle changes and are not assigned to the current runtime without matched measurement.

## Validation

v1.5.0 is gated by:

- pinned ComfyUI + official OpenVDN numerical/oracle tests;
- current ComfyUI-main import/registration smoke;
- adapter, pruning-affine, lifecycle, external-sequence, retained-buffer and compiler-guard tests;
- a dedicated `cudaMallocAsync` allocator regression;
- a frontend regression using an actual old 14-value Advanced-node `widgets_values` array.

The API-2 mixed-grid path has also been exercised in the production Continuum workflow with Flow-Aligned Regenerate, Spectrum/SA-PECE, DiffAid and Untwisting RoPE across multiple chunk boundaries.

## Upstream and licensing

- OpenVDN reference implementation: https://github.com/OpenVDN/vdn-minimax-h3
- Original ComfyUI port: https://github.com/Saganaki22/ComfyUI-VDN-H3
- This maintained fork: https://github.com/xmarre/ComfyUI-VDN-H3

Repository code is distributed under [Apache-2.0](LICENSE). Third-party model weights and repositories retain their own licenses. See [NOTICE](NOTICE) for attribution details.
