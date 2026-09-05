# Upstream reconciliation

This branch is not a mechanical merge of `Saganaki22/ComfyUI-VDN-H3`.  The fork has
intentional lifecycle, quantization, Continuum and Flow target-sparse changes, so
upstream deltas are reviewed by behavior and then adapted only where they preserve
those invariants.

## Reconciled upstream snapshot

Audited upstream head:

- repository: `Saganaki22/ComfyUI-VDN-H3`
- head: `fe6e6f2f26075f03dea09b5216e14db727af4b77`
- prior snapshot already reconciled by this PR: `e49edae28266bcaa9b74988ac95ef4dd035f959c`
- delta: 9 upstream commits

### Commit-by-commit disposition

| Upstream commit | Purpose | Fork disposition |
| --- | --- | --- |
| `57e92f12662b659f5be591ecb7e8f2deb58396e1` | v1.4.1 fast-kernel/default workaround | No net import: upstream itself reverted this in `29ed2a2...`. |
| `29ed2a2cdc45d97bdfb54dd6bc43b8ecb7f021b4` | Revert v1.4.1 workaround | Net state retained; there is nothing to port. |
| `103badc8eb8376d2589e4045530f9557296cab6d` | Suppress rejected-SDPA probe warnings and demote compile-fallback logging | Audited. The fork's grouped path dispatches through Comfy's `optimized_attention` / `comfy.ops.scaled_dot_product_attention` and has no upstream `_log_backend_once` probe loop, so that window change does not apply. Compile-helper fallback remains explicit because `fast_kernels` is an opt-in ablation in this fork. |
| `aa03c363a13b6b111680c6c358dd27ea7701d512` | Detect the new Comfy AIMDO compiler incompatibility | Adopted, but not as an Apply-time persistent global toggle. |
| `edc5101091518757342ddf783a57188dbae93372` | Tooltips, compiler scoping attempt, README notes, refreshed example workflow | Runtime compiler intent adopted. Fork-specific node tooltips/options supersede upstream `cache_gpu`/old-bypass text, so they are not overwritten wholesale. The example workflow is not copied blindly because the fork's node contract intentionally differs and the PNG/JSON pair must stay synchronized. |
| `e81634f37b0e650b7059e838d0dee2fdc6d24ce2` | Document compiler-off lifetime caveat | Superseded by the final per-forward guard below. |
| `dbc94d5e045d6766a47e7ef12dc19dc029b79694` | Fix compiler-stack detector | Adopted: detection follows the final upstream `comfy_aimdo.malloc_graph` stack shape. |
| `4478aa462e185282da380f9a1c6224bfc35a469c` | Fix import shadowing in compiler shim | Not reproduced: the fork guard uses module-level imports/lazy module lookup without the shadowing pattern. |
| `fe6e6f2f26075f03dea09b5216e14db727af4b77` | Scope compiler disable to VDN forwards and remove unload hook | Adopted and hardened. |

## Compiler guard adaptation

Upstream v1.4.3 established that Comfy builds with the AIMDO malloc-graph model
compiler can hard-fail on VDN-patched MiniMax-H3 forwards.  Current Comfy exposes
only the process-global `args.disable_comfy_compiler` control.

The fork therefore installs a guard around **VDN's own `DIFFUSION_MODEL` execution
wrapper**, not around Comfy core functions and not around model lifetime:

- lazy detection; old Comfy builds are a no-op;
- a user-supplied `--disable-comfy-compiler` setting is preserved;
- VDN-owned disables are reference-counted for nested/overlapping VDN wrappers;
- restoration occurs in `finally`, including exceptions/cancellation;
- no unload hook is installed;
- no Comfy function is monkey-patched;
- no flag is left disabled between VDN forwards.

Because Comfy's switch is process-global, a truly concurrent non-VDN forward in a
different executor thread cannot be perfectly isolated by any consumer-side toggle.
The normal Comfy prompt executor is serialized; the fork avoids widening that global
state beyond the VDN forward itself.

Tests cover no-op behavior on old stacks, exception restoration, preservation of a
user-disabled compiler, nested guard ownership and idempotent wrapper installation.

## Changes intentionally not reintroduced

Earlier upstream v1.4 resource work used implementation patterns that this fork has
already replaced with stricter ownership:

- private process-global branch GPU cache -> Comfy-managed resident additional model;
- quantized resident materialization -> quantized branch streaming;
- persistent/process-global scratch -> `VDNState`-owned bounded runtime buffers;
- per-state immortal prefetch thread -> one bounded tensor-less executor with
  state-owned cancellable result;
- unbounded/device-type-only Flex mask caching -> bounded full-device/layout LRU.

Those are not merge conflicts to resolve by reverting the fork; they are deliberate
lifecycle adaptations of the same performance intent.

## Validation requirement

A reconciliation is considered complete only when both CI lanes pass on the exact
final tree:

1. pinned Comfy + pinned OpenVDN direct numerical/orchestration oracle;
2. current Comfy `master` import/node-registration smoke.

Production GPU validation remains a separate gate and must still exercise the
AIMDO-era Comfy build with VDN, Continuum, Spectrum and the Flow target-sparse
external-sequence contract.
