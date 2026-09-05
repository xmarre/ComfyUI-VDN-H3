# Benchmarks — ComfyUI-VDN-H3

This file preserves historical measurements and defines the measurement contract for the current runtime. **Historical numbers are not performance validation of v1.5.1.**

## Current v1.5.1 execution differences

Compared with the older measurements below:

- `lora_mode=bypass` uses stack-safe activation-side Comfy `BypassForwardHook` adapters for ordinary LoRA targets;
- the v1.5.0 `add_weight_wrapper()` / `weight_function` adapter path is no longer active;
- VDN can coexist with another ordinary Comfy runtime-bypass provider on the same module by inserting underneath the existing chain and splicing itself out safely;
- fused INT8 `mlp.fc2` targets that do not call `module.forward` remain ordinary Comfy weight patches;
- full-width curve/pruned AdaLN adapters are projected through the exact pruning affine and applied as native curve weight + bias patches;
- the old private GPU branch cache is replaced by bounded streaming or a Comfy-managed additional `ModelPatcher`;
- selected upstream v1.4 performance work is retained under state-owned lifecycle rules: VRAM-aware branch selection, native INT8 ConvRot streaming under pressure, execution-leased retained scratch, and one-block streaming prefetch;
- `auto` placement reserves still-unloaded base-model bytes before assigning VDN residency.

New matched GPU measurements are required before assigning speed or VRAM numbers to v1.5.1.

## Historical RTX 5090 measurements

These measurements predate the v1.5/v1.5.1 lifecycle work.

### Rig

- GPU: NVIDIA GeForce RTX 5090 (sm_120, 32 GB)
- OS: Windows 11
- torch: 2.10.0+cu130
- triton: 3.6.0
- base: `minimax_h3_fl2va_int8_convrot.safetensors`
- VDN stage: `stage-dmd-step-250`
- sampler: Euler / simple
- sampling steps: 8
- seed: 42
- CFG: 1
- generation: t2v + audio

### 1280x736, 145 frames

Approximate layout: F=37, S=920, total packed sequence about 34,487 tokens.

| historical attention backend | s/it | sampling time | notes |
|---|---:|---:|---|
| grouped | 16.92–16.95 | ~2:15 | cold + warm runs |
| flex | 17.10 | ~2:17 | warm run after compilation |

On that workload grouped was effectively tied with FlexAttention. This is a hardware/workload-specific historical observation.

### 512x320, 56 frames

An 8-step end-to-end smoke render with audio completed successfully during earlier correctness bring-up.

## Adapter modes

### `lora_mode=merge`

Normal Comfy `ModelPatcher.add_patches()` application. This is the conservative reference path for output comparison. Quantized bases can incur a temporary dequantize -> patch -> requantize materialization cost during load.

### `lora_mode=bypass`

Ordinary adapter targets keep the resident base parameter untouched and add their low-rank activation delta through Comfy's bypass-hook mechanism. VDN's injection layer is specifically regression-tested with an independent external runtime-bypass provider in both provider insertion orders and across repeated load/unload cycles.

This path preserves the native base forward for quantized/custom layers instead of installing the v1.5.0 `weight_function` wrapper. That distinction matters because a Comfy weight function on quantized H3 can force a copied/dequantized compute-weight path.

The production reason for the v1.5.1 hotfix was a hard CUDA illegal-address abort in the stacked INT8 ConvRot H3 + VDN bypass + external runtime-DoRA + `cudaMallocAsync` workflow after v1.5.0 introduced runtime weight wrappers. CUDA failure reporting is asynchronous, so that trace does not prove which kernel originally faulted, but the v1.5.0 wrapper path was the new regression boundary and is no longer used.

## Curve/pruned AdaLN projection

For supported pruned MiniMax-H3 bases:

```text
dense(t) ≈ mean + curve(t) @ basis
```

A released full-width LoRA `B @ A` is converted once as:

```text
A_pruned   = A @ basis.T
bias_delta = B @ (A @ mean)
```

The constant term is required. v1.5.1 registers the projected native curve weight and bias terms as ordinary Comfy patches in both adapter modes; there is no reconstructed dense timestep MLP and no runtime weight wrapper for these terms.

The affine resolver accepts only the exact pruning basis/mean corresponding to the loaded curve table. A mismatched or unverifiable affine fails closed.

## Branch residency

`lora_mode` is independent of VDN linear-branch residency.

### `branch_weights=auto`

- starts from Comfy's current free-memory reading;
- subtracts bytes of the supplied base `MODEL` that are not resident yet;
- chooses ordinary BF16 `resident` only when the remaining budget satisfies the branch-size/headroom rule;
- otherwise selects `stream` and prefers the native INT8 ConvRot branch file when available.

### `branch_weights=stream`

- resolves branch tensors block-by-block;
- keeps safetensors mappings bounded to each resolution operation;
- when retained buffers are enabled on CUDA, uses one-block lookahead prefetch keyed by block/device/dtype.

### `branch_weights=resident`

- registers ordinary branch tensors as a Comfy-managed additional model;
- uses Comfy load/offload/device lifecycle;
- quantized branch files fail closed for resident mode rather than silently dequantizing them.

## Retained runtime buffers

`retain_buffers=auto|on|off` controls VDN scratch, not model weights. Retained mode can reuse bounded recurrence, grouped-window, activation-copy and prefetch scratch owned by one `VDNState`. One execution leases the pool; nested/concurrent execution falls back to isolated transient storage.

Under `cudaMallocAsync`, explicit `record_stream` bookkeeping is skipped because the allocator is already stream ordered.

## Current CI validation

CI validates implementation contracts rather than performance:

- pinned ComfyUI: `6c53f8c9a06d95f3d847009ceaae55c624169247`;
- official OpenVDN oracle: `b8cb28fbfca0266d1c7742a9f25ab8b58191de97`;
- source-to-source numerical oracle comparisons;
- adapter conversion and checkpoint/model-spec validation;
- curve/pruned affine projection including the constant bias term;
- merge-path custom/quantized weight lifecycle tests;
- stack-safe bypass lifecycle tests across independent providers and repeated cycles;
- explicit assertions that active bypass installs no `weight_wrapper_patches`;
- retained-vs-transient branch/window numerical identity;
- resource lease/isolation and prefetch placement tests;
- current ComfyUI-main import/registration smoke;
- legacy `ApplyVDNH3Advanced` positional-workflow migration.

Real decoded-media output and GPU timing remain separate release evidence, not something inferred from CPU CI.
