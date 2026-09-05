# Upstream reconciliation

This fork keeps the released VDN-H3 architecture and tracks the original ComfyUI port while adapting lifecycle mechanisms to current Comfy ownership and the Flow-Aligned mixed-grid contract.

## Reference points

- Original ComfyUI port: `Saganaki22/ComfyUI-VDN-H3`
- Reconciled upstream head: `b49130c26a70d12c542601c5bc4f7ee0f112ee2e` (`v1.4.3`)
- Official OpenVDN numerical oracle: `OpenVDN/vdn-minimax-h3@b8cb28fbfca0266d1c7742a9f25ab8b58191de97`

## Upstream v1.4.x work retained

- VRAM-aware branch placement;
- optional native INT8 ConvRot branch selection under pressure;
- one-block stream lookahead;
- reusable scan/window/activation scratch;
- grouped/Flex attention runtime improvements;
- the AIMDO malloc-graph compatibility workaround;
- v1.4.3's `cudaMallocAsync` rule: explicit `record_stream` is skipped when the allocator is already stream ordered.

## State ownership differences

The fork does not retain process-global model-weight ownership:

- no persistent process-global safetensors handles;
- no private process-global resident GPU branch cache;
- ordinary BF16 resident branch weights are a Comfy-managed additional `ModelPatcher`;
- quantized branch weights remain streamed;
- retained scan/window/activation storage belongs to one `VDNState` and is leased per execution;
- nested/concurrent execution that cannot acquire the retained pool receives isolated transient scratch;
- one-block prefetch uses a single bounded tensor-less executor and at most one state-owned future/result;
- completed prefetch results cannot be overwritten before `take()` consumes them;
- Flex BlockMask caching is bounded and keyed by device/layout identity.

## Adapter lifecycle: v1.5.1

`merge` remains normal `ModelPatcher.add_patches()` ownership.

For ordinary LoRA targets, `lora_mode=bypass` uses Comfy `BypassForwardHook` objects but **not** the unsafe plain provider lifecycle. VDN wraps those hooks in a stack-safe `PatcherInjection` derived from the production-validated PR #1 fix:

- VDN is inserted inside an already-active independent Comfy bypass chain;
- VDN can be spliced out while another provider remains active;
- both provider insertion orders are supported;
- clone-shared reruns replace the old live VDN hook set instead of accumulating deltas;
- cyclic existing chains fail closed;
- fused INT8 `mlp.fc2` targets that bypass `module.forward` remain ordinary weight patches.

v1.5.0 temporarily replaced this with `weight_function` / `add_weight_wrapper` runtime LoRA execution. That path forced quantized H3 layers through Comfy's dequantized compute-weight route and regressed the real stacked VDN + external runtime-DoRA workflow into a CUDA illegal-memory-access abort. v1.5.1 removes that active execution path.

Full-width released AdaLN LoRAs on supported pruned/curve H3 bases are still projected through the exact pruning affine, including the required constant bias term. The already-native projected curve weight/bias terms use normal Comfy patches in both adapter modes.

## Flow-Aligned external sequence

This fork additionally defines API 2 for `mixed_grid_low_suffix`, used by `xmarre/MiniMax-H3-Flow-Aligned-Regenerate`. It keeps the learned dense gate active on the mixed sequence while disabling only geometry-dependent local/linear-complement processing. API 1 target-sparse compatibility remains accepted.

## v1.4.3 allocator change

Upstream commit `b49130c26a70d12c542601c5bc4f7ee0f112ee2e` skips `record_stream` under `cudaMallocAsync`. This fork applies the same allocator condition in `vdn_h3.runtime._StreamPrefetcher`, where this branch owns its asynchronous prefetch lifecycle. Dedicated regression coverage remains in place.
