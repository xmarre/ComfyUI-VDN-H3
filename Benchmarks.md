# Benchmarks — ComfyUI-VDN-H3

This file preserves historical measurements and defines the measurement contract for the current runtime. **Historical numbers are not performance validation of v1.5.2.**

## Current v1.5.2 candidate execution differences

Compared with the older measurements below:

- `lora_mode=bypass` uses one VDN-owned PyTorch forward post-hook per affected module;
- VDN never replaces or splices `module.forward`, so another Comfy runtime-bypass provider retains independent ownership of its forward chain;
- the v1.5.0 `add_weight_wrapper()` / `weight_function` adapter path is not active;
- the v1.5.1 VDN `BypassForwardHook` linked-list/splicing path is not active;
- all VDN LoRA terms for one module are combined into one exact low-rank residual;
- runtime adapter factors are staged onto the intended compute device at injection time, before the first H3 forward;
- projected pruned/curve AdaLN updates stay off the base weight/bias in bypass mode and are applied as an exact projected post-forward residual plus constant bias;
- fused INT8 `mlp.fc2` targets that do not call `module.forward` remain ordinary Comfy weight patches;
- merge-mode curve/pruned AdaLN adapters are projected through the exact pruning affine and applied as native curve weight + bias patches;
- the old private GPU branch cache remains replaced by bounded streaming or a Comfy-managed additional `ModelPatcher`;
- selected upstream v1.4 performance work remains under state-owned lifecycle rules: VRAM-aware branch selection, native INT8 ConvRot streaming under pressure, execution-leased retained scratch, and one-block streaming prefetch;
- cross-stream prefetched branch tensors record the model consumer stream under both native and `cudaMallocAsync` allocators;
- `auto` placement reserves still-unloaded base-model bytes before assigning VDN residency.

New matched GPU measurements are required before assigning speed or VRAM numbers to v1.5.2. The complete stacked RTX PRO 6000 workflow is also a correctness release gate, not merely a benchmark.

## Historical RTX 5090 measurements

These measurements predate the v1.5-v1.5.2 lifecycle work.

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

Ordinary adapter targets keep the resident base parameter untouched. VDN registers a standard PyTorch forward post-hook and adds the exact low-rank residual after the module returns. VDN does not replace `module.forward`, does not enter another provider's `BypassForwardHook` chain, and does not install a Comfy `weight_function` wrapper.

The production history matters for interpreting this candidate:

- v1.5.0 weight wrappers failed the stacked INT8 ConvRot H3 + external runtime-DoRA + `cudaMallocAsync` workflow;
- v1.5.1 removed those wrappers but its VDN mutable-forward chain also failed;
- the first v1.5.2 post-hook candidate still materialized projected curve-AdaLN patches and lazily copied runtime factors in the first hook; the complete real run still failed at the first actual H3 evaluation.

The revised v1.5.2 candidate removes both remaining VDN-side boundaries: curve-AdaLN stays non-materialized in bypass mode, and runtime factors are staged before the model forward begins. CUDA reporting is asynchronous, so none of these historical Python stack locations are treated as proof of a particular originating CUDA kernel.

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

The constant term is required. In bypass mode the projected low-rank term and constant are applied after `adaln_proj.linear` without modifying its base parameters. In merge mode they remain ordinary Comfy weight/bias patches. The affine resolver accepts only the exact pruning basis/mean corresponding to the loaded curve table; a mismatched or unverifiable affine fails closed.

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
- when retained buffers are enabled on CUDA, uses one-block lookahead prefetch keyed by block/device/dtype;
- waits on the producer event and records the consumer stream on the returned storages before use.

### `branch_weights=resident`

- registers ordinary branch tensors as a Comfy-managed additional model;
- uses Comfy load/offload/device lifecycle;
- quantized branch files fail closed for resident mode rather than silently dequantizing them.

## Retained runtime buffers

`retain_buffers=auto|on|off` controls VDN scratch, not model weights. Retained mode can reuse bounded recurrence, grouped-window, activation-copy and prefetch scratch owned by one `VDNState`. One execution leases the pool; nested/concurrent execution falls back to isolated transient storage.

## Current CI validation

CI validates implementation contracts rather than performance:

- pinned ComfyUI: `6c53f8c9a06d95f3d847009ceaae55c624169247`;
- official OpenVDN oracle: `b8cb28fbfca0266d1c7742a9f25ab8b58191de97`;
- source-to-source numerical oracle comparisons;
- adapter conversion and checkpoint/model-spec validation;
- curve/pruned affine projection including the constant bias term;
- merge-path custom/quantized weight lifecycle tests;
- non-mutating post-forward bypass lifecycle tests across independent providers and repeated cycles;
- projected curve-AdaLN bypass identity without base weight/bias mutation;
- explicit assertions that active bypass installs no `weight_wrapper_patches`;
- injection-time runtime-factor staging contract;
- cross-stream branch-prefetch ownership including quantized backing tensors;
- retained-vs-transient branch/window numerical identity;
- resource lease/isolation and prefetch placement tests;
- current ComfyUI-main import/registration smoke;
- legacy `ApplyVDNH3Advanced` positional-workflow migration.

Real decoded-media output, full GPU workflow completion, VRAM, and timing remain separate release evidence, not something inferred from CPU CI.
