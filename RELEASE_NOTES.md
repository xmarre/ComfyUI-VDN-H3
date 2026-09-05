# ComfyUI-VDN-H3 v1.5.2

v1.5.2 is still under release validation. It replaces VDN's mutable-forward runtime bypass and, after the first post-hook candidate still failed the real stacked RTX PRO 6000 workflow, removes two additional VDN-side lifetime/materialization hazards exposed by that trace.

## Runtime-bypass ownership

- Ordinary VDN LoRA residuals use PyTorch forward **post-hooks**.
- VDN does **not** replace, splice, save, or restore `module.forward`.
- VDN does **not** use `ModelPatcher.weight_function` / `add_weight_wrapper`.
- One post-hook is registered per affected module, with all VDN terms fused into one exact low-rank residual for that module.
- Registration handles are generation-owned across clone-shared models: a newer VDN clone replaces the old registration, while stale ejects cannot remove the newer generation.
- Independently managed Comfy `BypassForwardHook` providers remain outside VDN's ownership; VDN never enters their mutable forward chain.
- Fused INT8 `mlp.fc2` targets retain native Comfy patch ownership because the H3 fused path bypasses `module.forward`.

## Revised pruned-AdaLN bypass

The first v1.5.2 candidate still materialized the exact projected curve-AdaLN update as native weight+bias patches in bypass mode. The full production run still aborted at the first actual H3 evaluation. CUDA reported the error when the first ordinary VDN post-hook attempted a device copy, but CUDA errors are asynchronous and MiniMax-H3 runs `adaln_proj` before the attention projection.

The revised candidate keeps the exact projection while removing that base-weight mutation:

```text
dense(t) ≈ mean + curve(t) @ basis
A_pruned   = A @ basis.T
bias_delta = B @ (A @ mean)
```

- In `bypass`, the projected low-rank curve-coordinate residual and constant bias are added by a post-forward hook on `adaln_proj.linear`.
- The pruned base AdaLN weight and bias remain untouched.
- In `merge`, the exact projected weight+bias terms continue to use ordinary native Comfy patches.

## Adapter staging

The first candidate lazily copied VDN adapter factors from CPU to GPU in the first post-hook invocation. That made the first `.to()` call a convenient asynchronous CUDA error-reporting boundary even if the actual fault occurred earlier.

The revised candidate stages all runtime VDN adapter factors synchronously onto the intended compute device **when the PatcherInjection is injected**, before any H3 module forward executes. No ordinary VDN post-hook performs its normal H2D setup during the first model call. An unexpected device/dtype change still has a synchronous correctness fallback.

## cudaMallocAsync prefetch lifetime

VDN's one-block branch prefetch uses a producer CUDA stream and hands the resulting tensors to the model's consumer stream. The fork previously inherited upstream v1.4.3's blanket `record_stream` suppression under `cudaMallocAsync`.

That is too broad for cross-stream handoff. PyTorch's `CUDAMallocAsyncAllocator` tracks recorded side-usage streams and synchronizes them before `cudaFreeAsync`. v1.5.2 therefore records the consumer stream for every prefetched tensor and its quantized backing storages under both native and `cudaMallocAsync` allocators.

## Validation status

CPU/oracle coverage includes:

- exact ordinary post-forward LoRA math with `module.forward` unchanged;
- exact projected curve-AdaLN bypass math with base weight/bias unchanged;
- coexistence with a real Comfy `BypassForwardHook` on ordinary and curve targets;
- clone-shared replacement and stale-generation ejection;
- repeated pseudo-Continuum injection/ejection cycles;
- custom/quantized-like modules retaining their native weight path;
- zero VDN weight wrappers and zero VDN mutable-forward wrappers in bypass mode;
- injection-time runtime-factor staging;
- cross-stream prefetch `record_stream` ownership including `cudaMallocAsync` and quantized backing tensors;
- merge mode remaining on normal weight patches;
- current-Comfy import smoke and official OpenVDN numerical/oracle coverage.

**Release remains blocked until the complete real RTX PRO 6000 production workflow finishes cleanly.**

# ComfyUI-VDN-H3 v1.5.1

v1.5.1 is a hotfix for a production regression introduced by v1.5.0's `lora_mode=bypass` implementation.

## Fixed: stacked runtime adapters on quantized H3

v1.5.0 moved VDN bypass adapters from Comfy's activation-side `BypassForwardHook` mechanism to `ModelPatcher.add_weight_wrapper()` / `weight_function`. On quantized MiniMax-H3 layers, a weight function forces Comfy through a copied/dequantized compute-weight path.

In the production RTX PRO 6000 workflow using:

- an INT8 ConvRot MiniMax-H3 base;
- VDN `lora_mode=bypass`;
- an independent runtime-bypass DoRA/LoRA provider on the same H3 modules;
- `cudaMallocAsync`;
- Continuum + Flow Mixed-Grid + Spectrum/SA-PECE;

the first actual H3 evaluation hard-aborted with `CUDA error: an illegal memory access was encountered`. The fatal Python stack was inside the external Comfy `BypassForwardHook`/LoRA `F.linear` chain. CUDA errors are asynchronous, so that stack alone does not identify the exact originating kernel.

v1.5.1 restored the older VDN `BypassForwardHook` architecture, including stack-safe linked-list insertion/removal. A later production run showed that this was not sufficient: the same stacked workflow still aborted. v1.5.1 is therefore superseded by the v1.5.2 work above.

## Pruned / curve AdaLN behavior

The v1.5 exact pruning-affine work was retained. Full-width released AdaLN LoRAs were projected through the resolved `adaln_basis` + `adaln_mean` pair. v1.5.1 materialized those projected curve weight/bias terms as ordinary Comfy patches in both adapter modes; the revised v1.5.2 bypass path no longer does so.

---

# ComfyUI-VDN-H3 v1.5.0

v1.5.0 was the correctness/lifecycle and Flow-interoperability release of the xmarre fork. It added the mixed-grid API used by MiniMax-H3 Flow-Aligned Regenerate, current pruned/INT8 MiniMax-H3 support, exact curve-AdaLN projection, branch/runtime lifecycle hardening, upstream v1.4.3 reconciliation, and backwards-compatible Advanced-node workflow migration.

Its `lora_mode=bypass` weight-wrapper implementation is superseded because of the stacked runtime-adapter regression described above. `merge` behavior was not implicated.
