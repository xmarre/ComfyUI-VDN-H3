# Upstream reconciliation

This fork keeps the released VDN-H3 architecture and tracks the original ComfyUI port while replacing lifecycle mechanisms that conflict with Comfy's model ownership or the Flow-Aligned mixed-grid contract.

## Reference points

- Original ComfyUI port: `Saganaki22/ComfyUI-VDN-H3`
- Reconciled upstream head: `b49130c26a70d12c542601c5bc4f7ee0f112ee2e` (`v1.4.3`)
- Official OpenVDN numerical oracle: `OpenVDN/vdn-minimax-h3@b8cb28fbfca0266d1c7742a9f25ab8b58191de97`

## Upstream v1.4.x work retained in this fork

The useful performance/runtime intent is preserved:

- VRAM-aware branch placement;
- optional native INT8 ConvRot branch selection under pressure;
- one-block stream lookahead;
- reusable scan/window/activation scratch;
- grouped/Flex attention runtime improvements;
- the AIMDO malloc-graph compatibility workaround;
- v1.4.3's `cudaMallocAsync` rule: explicit `record_stream` is skipped when the allocator is already stream ordered.

## Lifecycle differences kept intentionally

The fork does not copy upstream process-global resource ownership verbatim:

- no persistent process-global safetensors handles;
- no private process-global resident GPU branch cache;
- ordinary BF16 resident branch weights are a Comfy-managed additional `ModelPatcher`;
- quantized branch weights remain streamed;
- retained scan/window/activation storage belongs to one `VDNState` and is leased per execution;
- nested/concurrent execution that cannot acquire the retained pool receives isolated transient scratch;
- one-block prefetch uses a single bounded tensor-less executor and at most one state-owned future/result;
- completed prefetch results cannot be overwritten before `take()` consumes them;
- prefetch identity includes block, full device identity and compute dtype;
- Flex BlockMask caching is bounded and keyed by full device/layout identity.

## Adapter lifecycle differences

The old forward-hook LoRA bypass was removed. `lora_mode=bypass` now means Comfy-managed runtime weight/bias wrappers. VDN never traverses, replaces, restores, or chains LoRA target `module.forward` methods.

Full-width released AdaLN LoRAs on supported pruned/curve H3 bases are projected through the model's pruning affine, including the required constant bias term.

## Flow-Aligned external sequence

This fork additionally defines API 2 for `mixed_grid_low_suffix`, used by `xmarre/MiniMax-H3-Flow-Aligned-Regenerate`. It keeps the learned dense gate active on the mixed sequence while disabling only geometry-dependent local/linear-complement processing. API 1 target-sparse compatibility remains accepted.

## v1.4.3 allocator change

Upstream commit `b49130c26a70d12c542601c5bc4f7ee0f112ee2e` skips `record_stream` under `cudaMallocAsync`. This fork adopts the same allocator condition in `vdn_h3.runtime._StreamPrefetcher`, where this branch actually owns its asynchronous prefetch lifecycle. The behavior is covered by a dedicated regression test.
