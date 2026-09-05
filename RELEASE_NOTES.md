# ComfyUI-VDN-H3 v1.5.2

v1.5.2 replaces VDN's mutable-forward runtime bypass after the v1.5.1 hotfix
still hard-aborted in the real stacked RTX PRO 6000 workflow.

## Runtime-bypass ownership

- Ordinary VDN LoRA residuals use PyTorch forward **post-hooks**.
- VDN does **not** replace, splice, save, or restore `module.forward`.
- VDN does **not** use `ModelPatcher.weight_function` / `add_weight_wrapper`.
- One post-hook is registered per affected module, with all VDN terms fused into
  one exact low-rank residual for that module.
- Registration handles are generation-owned across clone-shared models: a newer
  VDN clone replaces the old registration, while stale ejects cannot remove the
  newer generation.
- Independently managed Comfy `BypassForwardHook` providers remain outside VDN's
  ownership; VDN never enters their mutable forward chain.
- Fused INT8 `mlp.fc2` and projected curve-AdaLN terms retain native Comfy patch
  ownership where a module post-hook is not semantically available.

The latest production failure was reported asynchronously at core
`LoRAAdapter.h` / `BypassForwardHook` during the first actual H3 call, so the
visible stack does not prove the originating CUDA kernel. This change therefore
removes the VDN-side shared mutable-forward topology rather than claiming a
specific CUDA kernel fix. Real GPU validation is required before release.

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

the first actual H3 evaluation hard-aborted with `CUDA error: an illegal memory access was encountered`. The fatal Python stack was inside the external Comfy `BypassForwardHook`/LoRA `F.linear` chain. CUDA errors are asynchronous, so that stack alone does not identify the exact originating kernel, but the regression boundary is the v1.5.0 weight-wrapper path: the previously tested stack-safe VDN bypass-hook implementation worked in the same multi-provider lifecycle.

v1.5.1 therefore restores that validated architecture for ordinary VDN LoRA targets instead of trying another private weight-wrapper variant.

### Bypass ownership in v1.5.1

- Ordinary VDN LoRA targets use Comfy's normal `BypassForwardHook` objects.
- VDN installs them through one stack-safe `PatcherInjection`.
- VDN is spliced **inside** an already-active Comfy bypass chain regardless of provider insertion order.
- VDN can remove itself from the middle of that live chain without restoring stale `module.forward` references.
- Reapplying VDN on a clone-shared inner model first removes the previous live VDN hook set, preventing accumulated deltas across reruns.
- Cyclic pre-existing bypass chains fail closed.
- Fused INT8 `mlp.fc2` targets that bypass `module.forward` remain normal Comfy weight patches.
- The v1.5.0 runtime `weight_function`/`add_weight_wrapper` path is no longer installed by `lora_mode=bypass`.

This is the same cross-provider lifetime design that was previously validated in the superseded PR #1; PR #1 itself remains closed because v1.5.1 integrates that lifecycle into the newer v1.5 architecture rather than reverting the rest of v1.5.

## Pruned / curve AdaLN behavior

The v1.5 exact pruning-affine work is retained.

Full-width released AdaLN LoRAs are still projected through the resolved `adaln_basis` + `adaln_mean` pair:

```text
A_pruned   = A @ basis.T
bias_delta = B @ (A @ mean)
```

In bypass mode, these already-native projected curve weight/bias terms now use ordinary Comfy `ModelPatcher.add_patches()` ownership. They do not use VDN forward hooks and do not use the v1.5.0 weight-wrapper path.

## Regression coverage

The v1.5.1 suite explicitly covers:

- repeated multi-hook VDN inject/eject cycles;
- VDN-first and external-provider-first injection ordering;
- same-order provider teardown in both orders;
- replacement of a live VDN hook set while an external bypass hook remains active;
- cyclic-chain fail-closed behavior;
- `apply_adapters(..., mode="bypass")` creating a VDN injection with **zero weight wrappers**;
- bypass operation on a plain linear module with no `weight_function` contract;
- projected curve AdaLN math through native weight/bias patches.

All v1.5.0 work unrelated to the regressed adapter execution mechanism remains: VDN API 2 mixed-grid support, pruned/curve AdaLN projection, branch placement and retained-buffer lifecycle, `cudaMallocAsync` prefetch handling, current Comfy smoke tests, and the legacy `ApplyVDNH3Advanced` widget migration.

---

# ComfyUI-VDN-H3 v1.5.0

v1.5.0 was the correctness/lifecycle and Flow-interoperability release of the xmarre fork. It added the mixed-grid API used by MiniMax-H3 Flow-Aligned Regenerate, current pruned/INT8 MiniMax-H3 support, exact curve-AdaLN projection, branch/runtime lifecycle hardening, upstream v1.4.3 reconciliation, and backwards-compatible Advanced-node workflow migration.

Its `lora_mode=bypass` weight-wrapper implementation is superseded by v1.5.1 because of the stacked runtime-adapter regression described above. `merge` behavior was not implicated.
