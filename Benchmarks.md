# Benchmarks — ComfyUI-VDN-H3

This file preserves historical measurements from earlier releases and defines the measurement contract for the current correctness/lifecycle refactor. **Historical numbers are not performance validation of the current branch.**

The current implementation differs materially from the runs below:

- the old forward-hook LoRA bypass is replaced by a Comfy-owned runtime `weight_function` path;
- VDN no longer traverses, splices or restores mutable `module.forward` LoRA chains;
- the old private GPU branch cache is replaced by either bounded streaming or a Comfy-managed additional `ModelPatcher`;
- safetensors mappings are bounded to load operations rather than retained process-global handles;
- curve/pruned AdaLN adapters are projected through the exact pruning affine (`adaln_basis` + `adaln_mean`) onto the native curve coordinates, including their constant bias term, rather than dropping weights or running a reconstructed dense timestep MLP;
- selected upstream v1.4 performance ideas are adapted under state-owned lifecycle rules: VRAM-aware branch selection, native INT8 ConvRot streaming under pressure, execution-leased retained scratch, and one-block streaming prefetch;
- `auto` placement reserves still-unloaded base-model bytes before deciding that VDN may consume free VRAM.

New matched GPU measurements are therefore required before making speed, VRAM or output-parity claims for this branch.

## Historical RTX 5090 measurements

These measurements predate the current refactor.

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

The original runs used the then-current implementation, including the old `BypassForwardHook`-based `lora_mode=bypass` and the old private `cache_gpu` branch path. Those names may still appear in serialized workflows, but the current implementations are different. Historical bypass/cache numbers therefore must not be attributed to the new runtime or managed-residency paths.

### 1280x736, 145 frames

Approximate layout: F=37, S=920, total packed sequence about 34,487 tokens.

| historical attention backend | s/it | sampling time | notes |
|---|---:|---:|---|
| grouped | 16.92–16.95 | ~2:15 | cold + warm runs |
| flex | 17.10 | ~2:17 | warm run after compilation |

On that workload the grouped implementation was effectively tied with FlexAttention. That is a hardware/workload-specific historical observation, not a current-branch result.

### 512x320, 56 frames

An 8-step end-to-end smoke render with audio completed successfully during earlier correctness bring-up.

## Historical v1.2/v1.3 validation

Earlier release validation included:

- eager vs `fast_kernels` comparisons;
- merge vs the **old forward-hook bypass** implementation;
- full and pruned MiniMax-H3 bases;
- fixed-seed 512x320 / 56-frame 8-step renders;
- INT8 ConvRot experiments.

Those runs were useful in finding lifecycle and numerical problems in the old bypass implementation. In particular, the historical Stage-DMD bypass path showed visible degradation relative to merge. The current `lora_mode=bypass` does not execute adapters through that mechanism, so those results neither validate nor condemn the new runtime mode.

## Upstream v1.4 performance work

Saganaki22's upstream v1.4 added meaningful performance work around branch-file selection, retained buffers, streaming prefetch and attention/runtime allocation behavior. This branch incorporates the useful ideas but **does not copy its ownership model verbatim**.

The current adaptation intentionally differs where correctness/lifecycle guarantees require it:

- no persistent `safe_open`/mmap handle cache;
- no process-global branch-weight GPU cache;
- native INT8 ConvRot is selected under memory pressure but remains streamed rather than being hidden in an unmanaged resident cache;
- ordinary BF16 residency is represented as a real Comfy additional `ModelPatcher`;
- retained scan/window/activation scratch is owned by one `VDNState` and leased per diffusion-model execution;
- nested/concurrent executions that cannot acquire the retained pool use isolated transient scratch;
- one-block prefetch uses one bounded process worker that owns no model tensors/cache; each VDN state owns at most one cancellable future/result;
- prefetched results are keyed by `(block, device, dtype)` to prevent stale placement reuse;
- `auto` free-VRAM calculations subtract still-unloaded bytes of the supplied base `MODEL` before branch/scratch policy decisions.

Upstream v1.4 benchmark numbers therefore remain useful motivation/reference data, not measurements of this implementation.

## Current adapter modes that need matched measurement

### `lora_mode=merge`

Normal Comfy `ModelPatcher.add_patches()` application. This is the reference/eager path for output comparison. Quantized bases can incur a temporary dequantize -> patch -> requantize VRAM spike during eager materialization.

On a curve/pruned base, a released full-width AdaLN LoRA is first transformed through the base's pruning affine. The projected low-rank weight update and its constant F32 bias term are then registered as ordinary Comfy patches.

### `lora_mode=bypass`

The current low-VRAM runtime path uses Comfy `add_weight_wrapper()` / `weight_function` ownership. It does not modify `module.forward`, does not create VDN LoRA injections, and leaves the resident base parameter unmerged. Low-rank Stage-B/Turbo terms are applied to the floating compute weight Comfy produces for the current invocation; the implementation avoids constructing a second persistent full-size patched model weight.

For curve/pruned AdaLN targets, the projected A/B factors and F32 constant bias offsets live in the same Comfy-managed additional model. Separate weight and bias wrappers apply them to the native small AdaLN projection. There is no dense timestep MLP and no AdaLN `forward` object patch in this path.

On INT8/custom-weight layers, a runtime weight wrapper can force a dequantized compute fallback for the patched invocation instead of the fused quantized kernel. The tradeoff is therefore lower patched-weight residency/load pressure versus potentially higher per-call compute cost. This must be measured rather than inferred.

## Curve/pruned AdaLN projection

For the supported pruned MiniMax-H3 lineage, the dense AdaLN input is represented by the pruning affine

```text
dense(t) ≈ mean + curve(t) @ basis
```

A released full-width LoRA `B @ A` is therefore converted once as

```text
A_pruned   = A @ basis.T
bias_delta = B @ (A @ mean)
```

The projection math is performed once in float64. Projected A and the constant bias term are stored as F32; B retains its checkpoint storage dtype until Comfy selects the invocation compute dtype. This adds no new model approximation beyond the pruned base's existing affine representation. The constant term is required; dropping it would not reproduce the affine-projected adapter.

The affine resolver accepts only the exact pruning basis/mean corresponding to the loaded curve table:

- stage-local `adaln_affine.safetensors` is the explicit companion path;
- the selected diffusion checkpoint and sibling safetensors are searched first;
- installed `models/diffusion_models` candidates are also searched;
- installed candidates must carry the matching curve table or a matching table identity; a different or unverified basis fails closed.

The repaired BF16 pruned source checkpoint can retain `adaln_basis` and `adaln_mean`; its INT8 derivatives may intentionally omit those inference-unused auxiliaries. Keeping the matching BF16 sibling installed is sufficient—the resolver reads only the tiny affine tensors/table from it. Otherwise `tools/extract_h3_adaln_affine.py` writes an approximately 97 KB sidecar.

## Current branch residency modes

`lora_mode` is independent of VDN linear-branch residency.

### `branch_weights=auto`

- starts from Comfy's current free-memory reading;
- subtracts bytes of the supplied base `MODEL` that are not resident yet;
- chooses ordinary BF16 `resident` only when the remaining budget satisfies the branch-size/headroom rule;
- otherwise selects `stream` and prefers the native INT8 ConvRot branch file when available;
- quantized branch weights are not silently promoted into an unmanaged resident cache.

### `branch_weights=stream`

- resolves branch tensors block-by-block;
- keeps safetensors mappings bounded to each resolution operation;
- when retained buffers are enabled on CUDA, uses one-block lookahead prefetch keyed by block/device/dtype.

### `branch_weights=resident`

- registers ordinary branch tensors as a Comfy-managed additional model;
- uses Comfy load/offload/device lifecycle;
- quantized branch files fail closed for resident mode rather than silently dequantizing them.

A minimum-memory configuration can combine `lora_mode=bypass` with `branch_weights=stream` and `retain_buffers=off`.

## Retained runtime buffers

`retain_buffers=auto|on|off` controls VDN scratch, not model weights.

Retained mode can reuse bounded storage for:

- forward/reverse recurrence banks;
- grouped-window plan/gather scratch;
- video/text raw Q/K/V copies;
- streaming lookahead state.

The pool is owned by the `VDNState`, not by process-global module dictionaries. One execution leases the retained pool; nested/concurrent executions fall back to transient storage so the same CUDA buffers are not raced. Cancellation clears the state-owned persistent scratch.

`auto` uses the effective free-VRAM budget (after unloaded base reservation) and the selected branch-file size. It is a heuristic that still requires real GPU validation.

## Current CI validation

CI validates implementation contracts rather than performance:

- pinned ComfyUI: `6c53f8c9a06d95f3d847009ceaae55c624169247`;
- official OpenVDN oracle: `b8cb28fbfca0266d1c7742a9f25ab8b58191de97`;
- direct reduced-dimension source-to-source oracle comparisons, including the official `HybridAttention` class;
- adapter conversion including hybrid/dense/token-refiner naming and fused variable-rank/scaled QKV patches;
- curve/pruned affine AdaLN projection tests, including the constant bias term, merge/runtime lifecycle, file replacement invalidation, wrong-table rejection and the production 51-target shape;
- direct KJ-loader-style selected-INT8 + matching-BF16-sibling affine discovery regression;
- ModelSpec and checkpoint corruption/invalidation tests;
- merge-path custom/quantized weight lifecycle tests;
- runtime-low-VRAM wrapper tests showing no forward hooks, no resident base-weight mutation and no adapter accumulation;
- retained-vs-transient VDN branch/window numerical identity tests;
- runtime resource lease/isolation tests;
- prefetch placement identity tests;
- base-residency-aware `auto` VRAM budget tests;
- repeated clone/load/unload and pseudo-Continuum conditioning/forward cycles;
- current Comfy `master` import/node-registration smoke.

The expanded suite is expected to contain **110 tests** after the KJ sibling-affine regression. Synthetic CPU tests intentionally do not stand in for real GPU VRAM, kernel selection, output quality or end-to-end performance.

## Production GPU validation status

The target RTX Pro 6000 / INT8-ConvRot workflow has already validated two important pre-sampling gates:

- `branch_weights=auto` correctly selected the quantized VDN branch and kept it streamed under the managed-lifecycle policy;
- released Stage-B/Turbo attention names are now converted to current Comfy fused `qkv_proj`/`out_proj` targets instead of failing on unfused names.

The next production attempt then reached the 51 full-width Turbo AdaLN targets and stopped before sampling because the older implementation required a dense time embedder. That requirement is what the new pruning-native affine projection replaces. The same workflow must now be rerun before any end-to-end performance or quality conclusion is drawn.

## Required real GPU matrix

Before claiming current-branch performance, output parity, or VRAM savings, collect matched fixed-seed runs for at least:

- BF16 MiniMax-H3 base;
- pruned/curve base with verified affine AdaLN projection;
- INT8 ConvRot base;
- `stage-dmd-step-250` with Turbo/DMD adapter at 8 steps;
- `stage-b-step-2000` without Turbo at the intended longer schedule;
- `lora_mode=merge` vs the **new** `lora_mode=bypass`;
- non-1.0 adapter strengths;
- `branch_weights=auto`, `stream`, and supported `resident`;
- `retain_buffers=auto`, `on`, and `off` where practical;
- grouped vs FlexAttention;
- repeated renders with the base kept resident;
- cancel then rerun without restarting ComfyUI;
- Continuum multi-chunk execution and reruns without restarting ComfyUI;
- ordinary LoRA composition and Dynamic DoRA/LoRA Loader with its Runtime Bypass OFF;
- Spectrum, DiffAid and Untwist RoPE where those are part of the target workflow.

For every matched run record:

- exact base/VDN checkpoint hashes;
- exact node commit and Comfy commit;
- resolution, frame count, sampler/scheduler, steps and seed;
- wall time and sampler time;
- peak VRAM / minimum free VRAM;
- model-load/adapter-materialization peak separately from steady-state sampling;
- selected branch file and resolved `auto` policy;
- retained/transient buffer policy;
- branch-transfer/prefetch time where measurable;
- attention time;
- compile cost separately from steady state when `torch.compile` is enabled;
- whether patched INT8 layers remained on their fused kernel or used the dequantized runtime fallback;
- fixed-seed visual/audio comparison against `merge`.

The critical lifecycle gate is a real Continuum run beyond chunk 1: Apply must resolve/project all 51 Turbo AdaLN targets, chunk 2+ must reach actual H3 transformer evaluations, repeated runs must not accumulate adapters, and no VDN LoRA `module.forward` chain should exist.