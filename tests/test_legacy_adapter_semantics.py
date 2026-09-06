from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vdn_h3 import legacy_adapters


def test_plugin_runtime_wires_v131_adapter_entrypoint():
    root_init = (Path(__file__).resolve().parents[1] / "__init__.py").read_text()
    assert "from vdn_h3.legacy_adapters import apply_adapters" in root_init
    assert "_apply_module.apply_adapters = _legacy_apply_adapters" in root_init
    assert "mixed_passthrough" not in root_init


def test_legacy_adapter_path_does_not_use_v15_curve_projection():
    source = (Path(__file__).resolve().parents[1] / "vdn_h3" / "legacy_adapters.py").read_text()
    assert "project_curve_terms" not in source
    assert "curve_affine" not in source
    assert "_inject_adaln_egrid" in source
    assert "BypassInjectionManager" in source
    assert "_inject_hook_stack_safe" in source


def test_unique_t_keeps_reference_conditioning_rows(monkeypatch):
    monkeypatch.setattr(
        legacy_adapters.comfy.ldm.minimax.model,
        "time_shift_sigma",
        lambda sigma, _shift_v, _shift_a: torch.as_tensor(0.25, device=sigma.device),
    )
    timestep = torch.tensor([500.0])

    base = legacy_adapters._unique_t(timestep, 12.0, 3.0, {})
    image = legacy_adapters._unique_t(
        timestep,
        12.0,
        3.0,
        {"refs": [{"kind": "image"}], "visual_cond_noise_aug": 0.999},
    )
    audio = legacy_adapters._unique_t(
        timestep,
        12.0,
        3.0,
        {"refs": [{"kind": "audio", "ref_audio_t": 4}], "audio_cond_noise_aug": 1.0},
    )

    assert base == pytest.approx([0.5, 0.75])
    assert image == pytest.approx([0.5, 0.75, 0.999])
    assert audio == pytest.approx([0.5, 0.75, 1.0])


def test_egrid_forward_is_explicitly_marked_v131():
    base = SimpleNamespace(apply_silu=False)
    forward = legacy_adapters._make_adaln_forward(
        base,
        torch.empty(1, 1),
        torch.empty(1, 1),
        {"silu_temb": None},
    )
    assert forward._vdn_v131_egrid is True
