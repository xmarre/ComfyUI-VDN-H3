# Benchmarks — ComfyUI-VDN-H3

This file preserves historical measurements from earlier releases. **They are not performance validation of the current lifecycle/adapter refactor.** The current branch removes VDN bypass adapters, replaces the old GPU cache with Comfy-managed residency, hardens checkpoint loading, and changes curve/pruned handling. A new matched GPU benchmark should be collected after real-render validation.

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

The original runs used settings from their then-current node implementation, including modes that no longer exist. In particular, the current node exposes only native `merge` adapter application and `stream` / Comfy-managed `resident` branch ownership.

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
- merge vs then-existing bypass comparisons;
- full and pruned MiniMax-H3 bases;
- fixed-seed 512x320 / 56-frame 8-step renders;
- INT8 ConvRot experiments.

Those results were useful in identifying that the old bypass mode was not a safe model-equivalent path. They should not be read as validation of the refactored implementation because the current architecture deliberately removes that mechanism.

## Current branch validation

The current CI validates implementation contracts rather than performance:

- pinned ComfyUI: `6c53f8c9a06d95f3d847009ceaae55c624169247`;
- official OpenVDN oracle: `b8cb28fbfca0266d1c7742a9f25ab8b58191de97`;
- direct reduced-dimension source-to-source oracle comparisons;
- adapter conversion including fused variable-rank/scaled QKV patches;
- curve/pruned exact-AdaLN reconstruction tests;
- ModelSpec and checkpoint corruption/invalidation tests;
- synthetic Comfy custom-weight/quantized patch lifecycle tests;
- repeated clone/load/unload and pseudo-Continuum conditioning/forward cycles;
- current Comfy `master` import/node-registration smoke.

At the time of this refactor the pinned suite reports **70 passing tests**.

## What still needs real GPU measurement

Before claiming current-branch performance or real-render parity, collect matched runs for at least:

- BF16 MiniMax-H3 base;
- pruned/curve base with exact dense-AdaLN reconstruction;
- INT8 ConvRot base;
- `stage-dmd-step-250` with Turbo/DMD adapter;
- `stage-b-step-2000` without Turbo;
- `stream` and `resident` branch modes where supported;
- grouped vs FlexAttention;
- repeated renders with the base kept resident;
- Continuum multi-chunk execution;
- representative composition with ordinary LoRAs / other Comfy weight patches.

Record wall time, sampling time, peak VRAM, branch transfer time, attention time and—when `torch.compile` is used—compile cost separately from steady state.

Synthetic CPU tests and historical renders are deliberately not substituted for those measurements.