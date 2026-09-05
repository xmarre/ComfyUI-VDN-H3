# ComfyUI-VDN-H3 v1.5.0

v1.5.0 is the correctness/lifecycle and Flow-interop release of the xmarre fork. It reconciles the useful upstream VDN-H3 v1.4.x runtime work while preserving explicit ComfyUI ownership, fixes the Continuum-era adapter recursion path, supports current pruned/INT8 MiniMax-H3 bases, and adds the mixed-grid API used by MiniMax-H3 Flow-Aligned Regenerate.

## Highlights

- Replaces the old forward-hook `lora_mode=bypass` implementation with Comfy-managed weight/bias wrappers. VDN no longer traverses or restores LoRA target `module.forward` chains.
- Preserves `merge` as the conservative reference adapter path while keeping a low-residency runtime mode.
- Adds fail-closed projection of released full-width AdaLN LoRAs onto supported pruned/curve H3 bases, including the constant bias term from the pruning affine.
- Adds `branch_weights=auto|stream|resident`, state-owned retained buffers, bounded one-block streaming prefetch, and Comfy-managed BF16 branch residency.
- Reconciles upstream through v1.4.3. Under `cudaMallocAsync`, explicit `record_stream` bookkeeping is skipped because the allocator is already stream ordered and current torch warns on the no-op calls.
- Adds external-sequence API 2 / `mixed_grid_low_suffix` for Flow-Aligned Regenerate. Mixed-grid execution keeps the learned dense softmax gate and disables only geometry-dependent local/linear-complement processing while the external mixed sequence is active.
- Keeps API 1 target-sparse compatibility for existing integrations.

## Legacy Advanced-node workflow migration

PR #2 inserted `retain_buffers` and `architecture_mode` into `ApplyVDNH3Advanced`. Older Comfy workflows stored widget values positionally, so loading the old 14-value layout against the new 16-widget definition could shift values into the wrong widget types.

v1.5.0 ships a frontend migration specifically for those persisted workflows:

- attaches the frontend-native `fallbackWidgetsValuesNames` order for the original 14 widgets;
- restores all fourteen old values by name;
- assigns `retain_buffers="auto"`;
- assigns `architecture_mode="override"` because the old node applied its architecture controls unconditionally;
- recognizes the short-lived current 16-value positional layout without changing its semantics;
- leaves workflows that already contain `widgets_values_named` untouched.

The next normal workflow save contains named widget values, so future widget reordering no longer relies on positional interpretation.

## Validation

The release is gated by repository CI before tagging:

- pinned ComfyUI + official OpenVDN numerical/oracle suite;
- current ComfyUI-main import and compiler-guard smoke;
- Python static/compile checks;
- allocator-guard regression for `cudaMallocAsync`;
- Node.js regression using an actual old 14-value `widgets_values` array, including the `architecture_mode="override"` migration semantic.

Real production testing also exercised the API-2 mixed-grid path together with Flow-Aligned Regenerate, Spectrum/SA-PECE, DiffAid and Untwisting RoPE across multiple Continuum boundaries without the previous LoRA recursion failure.

## Distribution

This GitHub release is built from the exact CI-tested `main` commit and includes a source ZIP plus `SHA256SUMS`.

The fork is released under its own xmarre metadata. OpenVDN, the upstream ComfyUI port, and the released VDN model weights retain their respective upstream ownership and licenses; see `NOTICE` and the README.
