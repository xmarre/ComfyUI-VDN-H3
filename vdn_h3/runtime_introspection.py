"""Read-only metadata bridge for VDN's non-mutating runtime adapters.

The v1.5.2 bypass path deliberately uses PyTorch forward post-hooks instead of
Comfy ``BypassForwardHook`` objects or ``ModelPatcher`` weight wrappers.  That
keeps the quantized H3 execution path isolated, but it also means consumers that
inspect ``ModelPatcher.injections`` before sampling cannot infer which LoRA terms
will become active when the injection is installed.

Spectrum H3's model-aware profiler is one such consumer.  It already understands
the ordinary Comfy ``BypassInjectionManager.adapters`` convention by inspecting a
``PatcherInjection`` closure.  Publish the same *read-only* adapter metadata shape
without changing VDN execution ownership: the real injection still installs only
the post-forward hooks created in :mod:`vdn_h3.apply`.

The metadata keeps references to the existing CPU adapter factors; it never
concatenates or copies them.  Projected curve-AdaLN low-rank terms are exposed as
ordinary LoRA metadata while their required constant offsets are exposed as an
explicit non-LoRA entry.  A profiler that understands classic LoRA can therefore
measure the low-rank part exactly and conservatively count the constant term as an
unknown patch, matching the visibility of the previous native weight+bias patch
representation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

import comfy.patcher_extension


@dataclass(frozen=True)
class _LoRAIntrospectionAdapter:
    """Minimal classic-LoRA descriptor; never participates in model execution."""

    up: torch.Tensor
    down: torch.Tensor

    name = "lora"

    @property
    def weights(self):
        # Spectrum/Comfy classic LoRA metadata convention:
        # (up, down, alpha, mid, dora_scale, reshape).
        # Runtime strength is carried separately by the registry value, so alpha
        # is rank and contributes no additional scale.
        rank = int(self.down.shape[0])
        return (self.up, self.down, float(rank), None, None, None)


@dataclass(frozen=True)
class _BiasIntrospectionAdapter:
    """Descriptor for a runtime constant offset that is not a weight LoRA."""

    offset: torch.Tensor

    name = "vdn_runtime_bias"

    @property
    def weights(self):
        return (self.offset,)


class _RuntimeAdapterRegistry:
    """Comfy-manager-shaped read-only metadata captured by an injection closure."""

    def __init__(self, adapters: dict[str, tuple[Any, float]]):
        self.adapters = adapters


def _metadata_key(
    adapters: dict[str, tuple[Any, float]],
    base_key: str,
    ordinal: int,
) -> str:
    if ordinal == 0 and base_key not in adapters:
        return base_key
    suffix = ordinal
    while True:
        candidate = f"{base_key}#vdn-runtime-{suffix}"
        if candidate not in adapters:
            return candidate
        suffix += 1


def _build_runtime_adapter_registry(
    terms_by_module,
    bias_terms_by_module=None,
) -> _RuntimeAdapterRegistry:
    """Build zero-copy introspection metadata for the active runtime residuals."""
    bias_terms_by_module = bias_terms_by_module or {}
    adapters: dict[str, tuple[Any, float]] = {}

    for module in sorted(terms_by_module):
        base_key = f"diffusion_model.{module}.weight"
        for ordinal, (down, up, scale) in enumerate(terms_by_module[module]):
            resolved_scale = float(scale)
            if resolved_scale == 0.0:
                continue
            key = _metadata_key(adapters, base_key, ordinal)
            adapters[key] = (
                _LoRAIntrospectionAdapter(up=up, down=down),
                resolved_scale,
            )

    for module in sorted(bias_terms_by_module):
        base_key = f"diffusion_model.{module}.bias"
        for ordinal, (offset, scale) in enumerate(bias_terms_by_module[module]):
            resolved_scale = float(scale)
            if resolved_scale == 0.0:
                continue
            key = _metadata_key(adapters, base_key, ordinal)
            adapters[key] = (
                _BiasIntrospectionAdapter(offset=offset),
                resolved_scale,
            )

    return _RuntimeAdapterRegistry(adapters)


def _wrap_injection_with_introspection(
    injection,
    registry: _RuntimeAdapterRegistry,
):
    """Delegate an injection unchanged while making ``registry`` introspectable."""

    def inject(model_patcher):
        # Deliberate closure capture.  Model-aware consumers can inspect this
        # manager-shaped metadata before sampling; runtime execution remains owned
        # entirely by the delegated injection.
        _ = registry
        return injection.inject(model_patcher)

    def eject(model_patcher):
        _ = registry
        return injection.eject(model_patcher)

    return comfy.patcher_extension.PatcherInjection(inject=inject, eject=eject)


def install_runtime_introspection_bridge() -> None:
    """Wrap VDN's post-forward injection installer once per Python process."""
    from . import apply as apply_module

    current = apply_module._install_post_forward_injection
    if getattr(current, "_vdn_runtime_introspection_bridge", False):
        return

    original = current

    def install(new_model, dm, terms_by_module, bias_terms_by_module=None):
        count = original(
            new_model,
            dm,
            terms_by_module,
            bias_terms_by_module,
        )
        if not count:
            return count

        registry = _build_runtime_adapter_registry(
            terms_by_module,
            bias_terms_by_module,
        )
        if not registry.adapters:
            return count

        injections = list((getattr(new_model, "injections", {}) or {}).get("vdn_lora", ()))
        if not injections:
            return count
        new_model.set_injections(
            "vdn_lora",
            [
                _wrap_injection_with_introspection(injection, registry)
                for injection in injections
            ],
        )
        return count

    install._vdn_runtime_introspection_bridge = True
    install._vdn_runtime_introspection_original = original
    apply_module._install_post_forward_injection = install
