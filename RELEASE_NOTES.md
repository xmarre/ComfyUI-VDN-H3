# ComfyUI-VDN-H3 v1.5.0

v1.5.0 is the correctness/lifecycle release of the xmarre fork used by MiniMax H3 Flow-Aligned Regenerate mixed-grid Continuum. It reconciles upstream through current `b49130c26a70d12c542601c5bc4f7ee0f112ee2e`, retains the fork's stricter Comfy ownership model, and promotes external-sequence API 2 after real GPU multi-boundary validation.

## Flow mixed-grid API 2

- Keeps API 1 support for the target-sparse control path.
- Adds `topology=mixed_grid_low_suffix` to `vdn_h3_external_sequence_v1` API 2.
- Validates both the regular native low-carrier sequence and the actual mixed target-prefix/source-suffix row count.
- Uses learned gated dense attention while disabling geometry-dependent local-window/linear-complement work only for the mixed external sequence.
- Returns to ordinary VDN behavior for the fresh full target-grid stage.
- Missing, stale or malformed external contracts fail closed.

## Runtime LoRA lifecycle

- Replaces the old forward-hook bypass chain with Comfy-managed weight/bias wrappers.
- Keeps base weights unmaterialized in runtime bypass mode and bounds low-rank delta temporaries.
- Supports released Stage-B/Turbo naming families and projection of full-width Turbo AdaLN LoRAs onto the native pruned/curve representation, including the required constant bias term.
- Adapter ownership is isolated per Apply execution; clones remain equivalent and repeated execution does not accumulate adapters.

## VDN resource ownership

- `branch_weights=auto|stream|resident` uses effective free VRAM after accounting for an unloaded base model.
- Quantized INT8-ConvRot branch weights remain streamed under managed-lifecycle policy.
- Retained scan/window/activation scratch belongs to one `VDNState`; nested/concurrent execution falls back to isolated transient buffers.
- One-block lookahead uses one bounded tensor-less executor and at most one state-owned outstanding result.
- Completed prefetch results cannot be silently overwritten by a later request.
- Flex BlockMask caching is bounded and keyed by full device/layout identity.

## Current Comfy compiler and allocator compatibility

- Adopts upstream v1.4.3's AIMDO incompatibility detection but scopes the process-global disable flag only around VDN diffusion-model forwards, with nested ownership and `finally` restoration.
- Reconciles upstream `b49130c...`: when PyTorch uses `cudaMallocAsync`, VDN now skips `record_stream`, which is unnecessary for the stream-ordered allocator and produced a warning in the validated RTX Pro 6000 workflow. Native allocator paths retain explicit `record_stream`.

## Real GPU validation

The integrated RTX Pro 6000 run exercised streamed INT8-ConvRot VDN branch weights, retained buffers, runtime adapter bypass, grouped attention, Spectrum + SA-Solver-PECE, DiffAid, Untwisting RoPE, learned H3 latent transfer, Flow API 2 mixed-grid continuation, fresh ordinary full-grid VDN stages, and multi-boundary Continuum completion. The Flow-side suffix DC bridge removed the remaining visible boundary flash while VDN stayed stable through the complete sequence.

This validates the interoperability/lifecycle path; it does not claim that every VDN branch/backend setting is numerically identical or universally faster.

## Tests

- Pinned Comfy + official OpenVDN oracle suite.
- Current Comfy main import/node-registration/compiler-guard smoke.
- Static/compile checks.
- New allocator tests verify `cudaMallocAsync` skips `record_stream` while the native allocator path retains it.

## Distribution

This GitHub release belongs to the xmarre correctness fork. The inherited Comfy Registry publisher identity remains `saganaki22`, so the fork release does **not** attempt to publish under that upstream registry identity.
