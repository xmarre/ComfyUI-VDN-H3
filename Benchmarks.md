# Benchmarks — ComfyUI-VDN-H3

This file preserves historical measurements from earlier releases. **They are not performance validation of the current lifecycle/runtime-adapter refactor.** The current branch replaces the old forward-hook LoRA bypass with a Comfy-owned runtime weight-wrapper implementation, replaces the old private GPU branch cache with Comfy-managed residency, hardens checkpoint loading, and changes curve/pruned handling. New matched GPU measurements are required before making current-branch VRAM, quality, or speed claims.

## Historical RTX 5090 measurements

These measurements were recorded before the current refactor.

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

The original runs used the then-current implementation, including the old `BypassForwardHook`-based `lora_mode=bypass` and the old `cache_gpu` branch setting. Those names may still exist for workflow compatibility, but their current implementations are different. Historical bypass numbers therefore must not be attributed to the new runtime weight-wrapper mode.

### 1280x736, 145 frames

Approximate layout: F=37, S=920, total packed sequence about 34,487 tokens.

| historical attention backend | s/it | sampling time | notes |
|---|---:|---:|---|
| grouped | 16.92–16.95 | ~2:15 | cold + warm runs |
| flex | 17.10 | ~2:17 | warm run after compilation |

On that workload the grouped implementation was effectively tied with FlexAttention. That observation is hardware/workload-specific and should not be generalized to the current branch without remeasurement.

### 512x320, 56 frames

An 8-step end-to-end smoke render with audio completed successfully during earlier correctness bring-up.

## Historical v1.2.0 validation

Earlier release validation included:

- eager vs `fast_kernels` comparisons;
- merge vs the **old forward-hook bypass** implementation;
- full and pruned MiniMax-H3 bases;
- fixed-seed 512x320 / 56-frame 8-step renders;
- INT8 ConvRot experiments.

Those runs were useful in finding both lifecycle and numerical problems in the old bypass implementation. In particular, the old Stage-DMD bypass path showed visible degradation relative to merge in historical testing. The new `lora_mode=bypass` does not execute adapters through that path, so those results neither validate nor condemn the new implementation.

## Current adapter modes that need matched measurement

### `lora_mode=merge`

Normal Comfy `ModelPatcher.add_patches()` application. This is the reference/eager path for output comparison. Quantized bases can incur a temporary dequantize -> patch -> requantize VRAM spike during eager materialization.

### `lora_mode=bypass`

The current low-VRAM runtime path uses Comfy `add_weight_wrapper()` / `weight_function` ownership. It does not modify `module.forward`, does not create VDN LoRA injections, leaves the resident base parameter unmerged, and applies low-rank terms to one transient current-layer compute weight with in-place `addmm_`.

On INT8/custom-weight layers, a runtime weight wrapper can force a dequantized compute fallback for the patched invocation instead of the fused quantized kernel. The expected tradeoff is therefore lower patched-weight residency/load pressure versus potentially higher per-call compute cost. This must be measured rather than inferred.

`lora_mode` is independent of VDN branch residency:

- `branch_weights=stream`: resolve branch tensors block-by-block;
- `branch_weights=resident`: Comfy-managed additional-model residency where supported.

A minimum-memory configuration can combine `lora_mode=bypass` with `branch_weights=stream`.

## Current CI validation

The current CI validates implementation contracts rather than performance:

- pinned ComfyUI: `6c53f8c9a06d95f3d847009ceaae55c624169247`;
- official OpenVDN oracle: `b8cb28fbfca0266d1c7742a9f25ab8b58191de97`;
- direct reduced-dimension source-to-source oracle comparisons;
- adapter conversion including fused variable-rank/scaled QKV patches;
- curve/pruned exact-AdaLN reconstruction tests;
- ModelSpec and checkpoint corruption/invalidation tests;
- merge-path custom/quantized weight lifecycle tests;
- runtime-low-VRAM wrapper tests showing no forward hooks, no resident base-weight mutation and no adapter accumulation;
- repeated clone/load/unload and pseudo-Continuum conditioning/forward cycles;
- current Comfy `master` import/node-registration smoke.

Synthetic CPU tests intentionally do not stand in for real GPU VRAM or quality measurements.

## Required real GPU matrix

Before claiming current-branch performance, output parity, or VRAM savings, collect matched fixed-seed runs for at least:

- BF16 MiniMax-H3 base;
- pruned/curve base with exact dense-AdaLN reconstruction;
- INT8 ConvRot base;
- `stage-dmd-step-250` with Turbo/DMD adapter at 8 steps;
- `stage-b-step-2000` without Turbo at the intended longer schedule;
- `lora_mode=merge` vs the **new** `lora_mode=bypass`;
- non-1.0 adapter strengths;
- `branch_weights=stream` and supported `resident`;
- grouped vs FlexAttention;
- repeated renders with the base kept resident;
- Continuum multi-chunk execution and reruns without restarting ComfyUI;
- ordinary LoRA composition and Dynamic DoRA/LoRA Loader with its Runtime Bypass OFF;
- Spectrum, DiffAid and Untwist RoPE where those are part of the target workflow.

For every matched run record:

- wall time and sampler time;
- peak VRAM / minimum free VRAM;
- model-load/adapter-materialization peak separately from steady-state sampling;
- branch-transfer time;
- attention time;
- compile cost separately from steady state when `torch.compile` is enabled;
- whether patched INT8 layers remained on their fused kernel or used the dequantized runtime fallback;
- fixed-seed visual/audio comparison against `merge`.

The critical lifecycle gate is a real Continuum run beyond chunk 1: chunk 2+ must reach actual H3 transformer evaluations, repeated runs must not accumulate adapters, and no VDN LoRA `module.forward` chain should exist.
