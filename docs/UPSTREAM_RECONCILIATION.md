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
- the AIMDO malloc-graph compatibility workaround.

The v1.4.3 allocator warning suppression is **not** copied literally anymore. Upstream skips every `record_stream` call when the allocator backend is `cudaMallocAsync`. That is correct only when recording the allocation/creation stream itself. VDN's branch prefetch is a different case: weights are allocated on a dedicated prefetch stream and consumed on the model stream. PyTorch's `CUDAMallocAsyncAllocator` explicitly keeps a set of `recorded_streams` and synchronizes those side-usage streams before `cudaFreeAsync`. The fork therefore always records the consumer stream for prefetched tensors, including their quantized backing storages, under both native and `cudaMallocAsync` allocators.

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
- cross-stream prefetch ownership is recorded on the consumer stream before the producer-owned tensors can be released;
- Flex BlockMask caching is bounded and keyed by device/layout identity.

## Adapter lifecycle: v1.5.2 candidate

`merge` remains normal `ModelPatcher.add_patches()` ownership.

For ordinary LoRA targets, `lora_mode=bypass` uses VDN-owned PyTorch forward post-hooks. VDN intentionally does not use either of the two adapter-execution topologies that failed the real stacked quantized workflow:

- no `weight_function` / `add_weight_wrapper` execution from v1.5.0;
- no VDN `BypassForwardHook` linked list or custom `module.forward` splicing from v1.5.1.

Instead:

- the existing module forward executes normally;
- any independently managed runtime provider keeps sole ownership of whatever forward wrapper chain it installs;
- VDN adds its exact low-rank residual from a PyTorch post-hook after that call returns;
- all VDN terms for one module are combined into one post-hook residual;
- adapter factors are moved to the intended compute device synchronously at injection time, before the first H3 forward;
- one generation-owned `PatcherInjection` registers/removes the handles;
- a newer clone-shared VDN generation replaces the older registration;
- stale older ejects cannot remove the newer generation;
- fused INT8 `mlp.fc2` targets that bypass `module.forward` remain ordinary weight patches.

### Pruned / curve AdaLN

The exact pruning affine is still used:

```text
dense(t) ≈ mean + curve(t) @ basis
A_pruned   = A @ basis.T
bias_delta = B @ (A @ mean)
```

The first v1.5.2 candidate projected those terms correctly but then registered the projected curve weight and constant bias as ordinary native patches even in bypass mode. The real RTX PRO 6000 run invalidated that candidate: the first actual H3 evaluation still hit `cudaErrorIllegalAddress`, asynchronously reported when the first ordinary VDN post-hook attempted its lazy device copy. MiniMax-H3 evaluates `adaln_proj` before the attention projection, so the materialized VDN curve patch was the remaining VDN-specific adapter operation preceding that reporting boundary.

The revised candidate therefore keeps the exact projected AdaLN update out of the base parameter tree in `bypass` mode as well. The projected low-rank curve-coordinate residual plus its constant bias are added by a post-forward hook on `adaln_proj.linear`; the underlying pruned weight and bias remain untouched. `merge` mode continues to use native projected weight/bias patches.

This isolates bypass mode from all three VDN-side base-weight/forward mutations implicated by the production traces: runtime weight wrappers, mutable forward chains, and projected curve-AdaLN materialization.

## Flow-Aligned external sequence

This fork additionally defines API 2 for `mixed_grid_low_suffix`, used by `xmarre/MiniMax-H3-Flow-Aligned-Regenerate`. It keeps the learned dense softmax gate active on the mixed sequence while disabling only geometry-dependent local/linear-complement processing. API 1 target-sparse compatibility remains accepted.

## Release gate

The revised v1.5.2 implementation remains a candidate until the complete production RTX PRO 6000 workflow finishes without the CUDA abort. CPU/oracle CI proves numerical and ownership contracts, not GPU allocator/kernel safety.
