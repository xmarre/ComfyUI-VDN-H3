from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, got {count}")
    return text.replace(old, new, 1)


def regex_once(text: str, pattern: str, replacement: str, label: str) -> str:
    text2, count = re.subn(pattern, replacement, text, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, got {count}")
    return text2


apply_path = ROOT / "vdn_h3" / "apply.py"
text = apply_path.read_text()

text = regex_once(
    text,
    r'\A"""Apply released VDN adapters through ComfyUI-owned patch mechanisms\..*?"""\nfrom __future__ import annotations\n',
    '''"""Apply released VDN adapters through ComfyUI-owned patch mechanisms.

``merge`` uses normal ``ModelPatcher.add_patches`` weight ownership.

``bypass`` is the low-residency execution mode. Ordinary VDN LoRA terms are
implemented with PyTorch forward *post-hooks*, not Comfy ``BypassForwardHook``
and not ``ModelPatcher.weight_function``. VDN therefore never replaces or
splices ``module.forward`` and cannot participate in another provider's mutable
forward-wrapper linked list.

This split is deliberate for quantized MiniMax-H3. The v1.5.0 weight-wrapper
path forced custom quantized layers through a copied/dequantized weight path and
hard-aborted in the production stacked-adapter workflow. v1.5.1 restored the old
mutable-forward bypass chain, but the same real RTX PRO 6000 workflow still
hard-aborted at the first H3 evaluation. VDN now stays out of both mechanisms:
the native quantized base forward runs unchanged, an independently managed Comfy
runtime adapter may wrap that forward if it wants to, and VDN adds its exact
low-rank residual after the module returns.

Fused INT8 ``mlp.fc2`` targets whose H3 fast path bypasses ``module.forward``
remain ordinary Comfy weight patches. Full-width AdaLN LoRAs on curve/pruned H3
bases remain projected once through the exact pruning affine (basis + mean), with
the resulting native curve weight and constant-bias terms owned by normal Comfy
patches in both modes.
"""
from __future__ import annotations
''',
    "module docstring",
)

text = replace_once(
    text,
    "import logging\n\nimport torch\n",
    "import logging\nimport threading\nimport weakref\nfrom collections import defaultdict\n\nimport torch\n",
    "imports",
)
text = replace_once(
    text,
    "import comfy.lora\nimport comfy.patcher_extension\nimport comfy.utils\nimport comfy.weight_adapter\n",
    "import comfy.lora\nimport comfy.patcher_extension\nimport comfy.utils\n",
    "comfy imports",
)

runtime_block = r'''class _FrugalLoRA.*?\ndef apply_adapters\('''
runtime_replacement = r'''def _int8_fused_fc2(dm, modules):
    """Return fc2 targets whose fused quantized forward bypasses module.forward.

    A PyTorch forward hook cannot observe these H3 fused calls. Keep these terms
    under normal Comfy weight-patch ownership, matching the previously validated
    quantized path.
    """
    fused = []
    for module in modules:
        if not module.endswith(".mlp.fc2"):
            continue
        try:
            weight = comfy.utils.get_attr(dm, module + ".weight")
        except Exception:
            continue
        if (
            getattr(weight, "_layout_cls", None) == "TensorWiseINT8Layout"
            and not getattr(getattr(weight, "_params", None), "transposed", False)
        ):
            fused.append(module)
    return fused


class _PostForwardLoRA:
    """Exact additive LoRA residual without replacing ``module.forward``.

    All terms for one module are fused into one down/up pair per active
    device/dtype. Source tensors stay in their checkpoint-owned representation;
    device copies are cached only while the PatcherInjection is active and are
    dropped on eject/replacement.
    """

    def __init__(self, terms):
        self.terms = tuple(
            (down.detach(), up.detach(), float(scale))
            for down, up, scale in terms
        )
        self._cache = {}

    def _compiled_weights(self, x: torch.Tensor):
        key = (x.device.type, x.device.index, x.dtype)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        downs = []
        ups = []
        for down, up, scale in self.terms:
            down_active = down.to(device=x.device, dtype=x.dtype, non_blocking=True)
            up_active = up.to(device=x.device, dtype=x.dtype, non_blocking=True)
            if scale != 1.0:
                up_active = up_active * scale
            downs.append(down_active)
            ups.append(up_active)

        if not downs:
            raise RuntimeError("VDN post-forward LoRA hook has no adapter terms")
        down_cat = downs[0] if len(downs) == 1 else torch.cat(downs, dim=0)
        up_cat = ups[0] if len(ups) == 1 else torch.cat(ups, dim=1)
        self._cache[key] = (down_cat, up_cat)
        return down_cat, up_cat

    def __call__(self, module, inputs, output):
        if not isinstance(output, torch.Tensor):
            raise RuntimeError(
                f"VDN post-forward bypass expected Tensor output from "
                f"{type(module).__name__}, got {type(output).__name__}"
            )
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError(
                f"VDN post-forward bypass expected the first positional input to "
                f"{type(module).__name__} to be a Tensor"
            )
        x = inputs[0]
        down, up = self._compiled_weights(x)
        delta = F.linear(F.linear(x, down), up)
        return output + delta

    def clear(self):
        self._cache.clear()


# ModelPatcher clones share the same inner model. Track the currently active VDN
# registration outside the model object so a newer clone can replace an older
# registration without private attributes on the model and so an old clone's
# later eject cannot tear down the newer generation.
_ACTIVE_POST_FORWARD = weakref.WeakKeyDictionary()
_ACTIVE_POST_FORWARD_LOCK = threading.RLock()


def _remove_post_forward_registration(registration):
    for handle in reversed(registration["handles"]):
        handle.remove()
    for plan in registration["plans"]:
        plan.clear()


def _install_post_forward_injection(new_model, dm, terms_by_module):
    if not terms_by_module:
        return 0

    plans = []
    for path in sorted(terms_by_module):
        module = comfy.utils.get_attr(dm, path)
        register = getattr(module, "register_forward_hook", None)
        if not callable(register):
            raise RuntimeError(
                f"VDN bypass target {path!r} ({type(module).__name__}) does not "
                "support PyTorch forward hooks"
            )
        plans.append((path, module, _PostForwardLoRA(terms_by_module[path])))

    owner = new_model.model
    token = object()

    def inject_all(model_patcher):
        del model_patcher
        with _ACTIVE_POST_FORWARD_LOCK:
            current = _ACTIVE_POST_FORWARD.get(owner)
            if current is not None and current["token"] is token:
                return
            if current is not None:
                _remove_post_forward_registration(current)

            handles = []
            hook_plans = [plan for _, _, plan in plans]
            try:
                for _, module, plan in plans:
                    handles.append(module.register_forward_hook(plan))
            except Exception:
                for handle in reversed(handles):
                    handle.remove()
                for plan in hook_plans:
                    plan.clear()
                raise

            _ACTIVE_POST_FORWARD[owner] = {
                "token": token,
                "handles": handles,
                "plans": hook_plans,
            }

    def eject_all(model_patcher):
        del model_patcher
        with _ACTIVE_POST_FORWARD_LOCK:
            current = _ACTIVE_POST_FORWARD.get(owner)
            # A newer clone may already have replaced this registration. An old
            # eject must not remove the newer generation.
            if current is None or current["token"] is not token:
                return
            _remove_post_forward_registration(current)
            try:
                del _ACTIVE_POST_FORWARD[owner]
            except KeyError:
                pass

    injection = comfy.patcher_extension.PatcherInjection(
        inject=inject_all,
        eject=eject_all,
    )
    new_model.set_injections("vdn_lora", [injection])
    return len(plans)


def apply_adapters('''
text = regex_once(text, runtime_block, runtime_replacement, "runtime implementation")

text = replace_once(
    text,
    '    """Apply released VDN adapters through merge or stack-safe bypass mode."""',
    '    """Apply released VDN adapters through merge or forward-post-hook bypass."""',
    "apply docstring",
)
text = replace_once(
    text,
    "    curve_terms = {}\n    all_hooks = []\n",
    "    curve_terms = {}\n    runtime_terms = defaultdict(list)\n",
    "runtime accumulator",
)

old_bypass = '''            bypass_targets = _bypass(
                new_model,
                ordinary,
                bypass_modules,
                s,
                all_hooks,
            )
            if fused_terms:
'''
new_bypass = '''            for module in bypass_modules:
                down, up, term_scale = ordinary[module]
                runtime_terms[module].append(
                    (down, up, float(term_scale) * s)
                )
            bypass_targets = len(bypass_modules)
            if fused_terms:
'''
text = replace_once(text, old_bypass, new_bypass, "bypass term collection")
text = replace_once(
    text,
    '                "[vdn] adapter %s: %d native patches, %d stack-safe bypass targets, "\n',
    '                "[vdn] adapter %s: %d native patches, %d post-forward bypass targets, "\n',
    "verbose log",
)

old_report = '''    if mode == "bypass":
        _install_injection(new_model, all_hooks)
        runtime_report = {
            "mode": "stack_safe_bypass",
            "forward_hooks": len(all_hooks),
            "weight_wrappers": 0,
            "bias_wrappers": 0,
            "managed_adapter_bytes": 0,
            "delta_buffer_limit_bytes": 0,
            "owner_key": None,
            "stack_safe_cross_provider": True,
            "projected_curve_weight_patches": curve_weight_patches,
            "projected_curve_bias_patches": curve_bias_patches,
        }
'''
new_report = '''    if mode == "bypass":
        forward_hook_modules = _install_post_forward_injection(
            new_model, dm, runtime_terms
        )
        runtime_term_count = sum(len(terms) for terms in runtime_terms.values())
        runtime_report = {
            "mode": "post_forward_hook_bypass",
            "forward_hooks": forward_hook_modules,
            "pytorch_forward_post_hooks": forward_hook_modules,
            "runtime_terms": runtime_term_count,
            "mutable_forward_wrappers": 0,
            "module_forward_untouched": True,
            "weight_wrappers": 0,
            "bias_wrappers": 0,
            "managed_adapter_bytes": 0,
            "delta_buffer_limit_bytes": 0,
            "owner_key": None,
            "stack_safe_cross_provider": True,
            "projected_curve_weight_patches": curve_weight_patches,
            "projected_curve_bias_patches": curve_bias_patches,
        }
'''
text = replace_once(text, old_report, new_report, "runtime report")

apply_path.write_text(text)

# Replace the old mutable-forward regression suite with invariants for the new
# architecture. These tests intentionally coexist with a real Comfy
# BypassForwardHook to prove VDN leaves that provider's forward chain alone.
(ROOT / "tests" / "test_bypass_reinject.py").write_text(r'''"""Lifecycle regressions for VDN's non-mutating runtime bypass.

VDN must never replace ``module.forward``. Its low-residency LoRA residuals use
PyTorch forward post-hooks whose handles are owned by one PatcherInjection.
ModelPatcher clones share the inner model, so a new VDN generation replaces the
old registered handles and stale clone ejection cannot remove the newer set.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import comfy.model_management
import comfy.model_patcher
from comfy.weight_adapter.bypass import BypassForwardHook
from comfy.weight_adapter.lora import LoRAAdapter

from vdn_h3.apply import apply_adapters


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8, bias=False)
        self.use_adaln_curves = False


class Root(nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = Diffusion()
        self.device = torch.device("cpu")


def _base():
    torch.manual_seed(901)
    return comfy.model_patcher.ModelPatcher(
        Root(), torch.device("cpu"), torch.device("cpu")
    )


def _term(seed, rank=3):
    gen = torch.Generator().manual_seed(seed)
    down = torch.randn(rank, 8, generator=gen)
    up = torch.randn(8, rank, generator=gen)
    return down, up


def _apply(base, down, up, strength=1.0):
    patcher = base.clone()
    report = apply_adapters(
        patcher,
        {"default": {"linear": (down, up, 1.0)}},
        strength,
        mode="bypass",
        stage_path=None,
    )
    return patcher, report


def _delta(x, down, up, strength=1.0):
    return strength * torch.nn.functional.linear(
        torch.nn.functional.linear(x, down), up
    )


def test_vdn_bypass_never_replaces_module_forward(monkeypatch):
    monkeypatch.setattr(
        comfy.model_management, "get_torch_device", lambda: torch.device("cpu")
    )
    base = _base()
    module = base.model.diffusion_model.linear
    true_forward = module.forward
    down, up = _term(902)
    vdn, report = _apply(base, down, up, 0.75)
    injection = vdn.injections["vdn_lora"][0]
    x = torch.randn(4, 8)
    want = true_forward(x) + _delta(x, down, up, 0.75)

    assert report["runtime_bypass"]["mode"] == "post_forward_hook_bypass"
    assert report["runtime_bypass"]["mutable_forward_wrappers"] == 0
    assert report["runtime_bypass"]["module_forward_untouched"] is True
    assert module.forward == true_forward

    injection.inject(vdn)
    try:
        assert module.forward == true_forward
        assert torch.allclose(module(x), want, atol=1e-5, rtol=1e-5)
    finally:
        injection.eject(vdn)
    assert module.forward == true_forward


def test_clone_replacement_and_stale_eject_do_not_accumulate(monkeypatch):
    monkeypatch.setattr(
        comfy.model_management, "get_torch_device", lambda: torch.device("cpu")
    )
    base = _base()
    module = base.model.diffusion_model.linear
    true_forward = module.forward
    down1, up1 = _term(903)
    down2, up2 = _term(904)
    first, _ = _apply(base, down1, up1, 1.0)
    second, _ = _apply(base, down2, up2, 0.5)
    inj1 = first.injections["vdn_lora"][0]
    inj2 = second.injections["vdn_lora"][0]
    x = torch.randn(3, 8)

    inj1.inject(first)
    assert module.forward == true_forward
    assert torch.allclose(
        module(x), true_forward(x) + _delta(x, down1, up1), atol=1e-5
    )

    # Same shared model, newer clone: this must replace, not stack.
    inj2.inject(second)
    want2 = true_forward(x) + _delta(x, down2, up2, 0.5)
    assert module.forward == true_forward
    assert torch.allclose(module(x), want2, atol=1e-5)

    # Ejecting the stale generation must not tear down the current one.
    inj1.eject(first)
    assert module.forward == true_forward
    assert torch.allclose(module(x), want2, atol=1e-5)

    inj2.eject(second)
    assert module.forward == true_forward
    assert torch.allclose(module(x), true_forward(x), atol=1e-6)


def _external_hook(module, down, up, strength=1.0):
    # Core Comfy LoRA bypass hook: this deliberately DOES mutate module.forward.
    # VDN must coexist without becoming part of this linked list.
    alpha = torch.tensor(float(down.shape[0]))
    adapter = LoRAAdapter(set(), (up, down, alpha, None, None, None))
    return BypassForwardHook(module, adapter, multiplier=float(strength))


def _run_cross_provider(monkeypatch, external_first):
    monkeypatch.setattr(
        comfy.model_management, "get_torch_device", lambda: torch.device("cpu")
    )
    base = _base()
    module = base.model.diffusion_model.linear
    true_forward = module.forward
    vd, vu = _term(905)
    ed, eu = _term(906)
    vdn, _ = _apply(base, vd, vu, 0.6)
    injection = vdn.injections["vdn_lora"][0]
    external = _external_hook(module, ed, eu, 0.4)
    x = torch.randn(2, 8)
    want = (
        true_forward(x)
        + _delta(x, ed, eu, 0.4)
        + _delta(x, vd, vu, 0.6)
    )

    if external_first:
        external.inject()
        external_forward = module.forward
        injection.inject(vdn)
    else:
        injection.inject(vdn)
        assert module.forward == true_forward
        external.inject()
        external_forward = module.forward

    try:
        # VDN registration never changes the external provider's forward object.
        assert module.forward == external_forward
        assert torch.allclose(module(x), want, atol=1e-5, rtol=1e-5)

        injection.eject(vdn)
        # External provider is still live and exact after VDN removal.
        assert module.forward == external_forward
        external_only = true_forward(x) + _delta(x, ed, eu, 0.4)
        assert torch.allclose(module(x), external_only, atol=1e-5, rtol=1e-5)
    finally:
        # idempotent VDN stale eject is harmless
        injection.eject(vdn)
        external.eject()

    assert module.forward == true_forward


def test_cross_provider_external_first(monkeypatch):
    _run_cross_provider(monkeypatch, external_first=True)


def test_cross_provider_vdn_first(monkeypatch):
    _run_cross_provider(monkeypatch, external_first=False)


def test_repeated_pseudo_continuum_chunks_do_not_accumulate(monkeypatch):
    monkeypatch.setattr(
        comfy.model_management, "get_torch_device", lambda: torch.device("cpu")
    )
    base = _base()
    module = base.model.diffusion_model.linear
    true_forward = module.forward
    down, up = _term(907)
    x = torch.randn(3, 8)
    want = true_forward(x) + _delta(x, down, up)

    for chunk in range(12):
        vdn, _ = _apply(base, down, up, 1.0)
        injection = vdn.injections["vdn_lora"][0]
        injection.inject(vdn)
        try:
            assert module.forward == true_forward, chunk
            assert torch.allclose(module(x), want, atol=1e-5), chunk
        finally:
            injection.eject(vdn)
        assert module.forward == true_forward, chunk
        assert torch.allclose(module(x), true_forward(x), atol=1e-6), chunk
''')

# Existing lifecycle tests should now require an untouched forward, not a
# BypassForwardHook replacement.
quant = ROOT / "tests" / "test_quantized_patch.py"
q = quant.read_text()
q = q.replace(
    "    assert module.forward != true_forward\n",
    "    assert module.forward == true_forward\n",
)
q = q.replace(
    "            assert module.forward != true_forward, cycle\n",
    "            assert module.forward == true_forward, cycle\n",
)
q = q.replace(
    "activation-bypass lifecycles",
    "non-mutating runtime-bypass lifecycles",
)
q = q.replace(
    "v1.5.1 no longer installs a weight wrapper",
    "the runtime path installs neither a weight wrapper nor a forward replacement",
)
quant.write_text(q)

runtime = ROOT / "tests" / "test_runtime_lowvram.py"
r = runtime.read_text()
r = replace_once(
    r,
    '    assert runtime["mode"] == "stack_safe_bypass"\n',
    '    assert runtime["mode"] == "post_forward_hook_bypass"\n',
    "runtime test mode",
)
r = replace_once(
    r,
    '    assert runtime["forward_hooks"] == 1\n',
    '    assert runtime["forward_hooks"] == 1\n    assert runtime["pytorch_forward_post_hooks"] == 1\n    assert runtime["mutable_forward_wrappers"] == 0\n    assert runtime["module_forward_untouched"] is True\n',
    "runtime report assertions",
)
r = r.replace(
    "v1.5.1 must use the activation-side Comfy bypass contract instead.",
    "the VDN runtime path uses a PyTorch post-forward hook instead.",
)
r = replace_once(
    r,
    "    injection.inject(vdn)\n    try:\n        got = module(x)\n",
    "    injection.inject(vdn)\n    try:\n        assert module.forward == true_forward\n        got = module(x)\n",
    "runtime unchanged forward",
)
runtime.write_text(r)

# Curve-only bypass still has no runtime hook and remains native patches.
curve = ROOT / "tests" / "test_curve_apply.py"
c = curve.read_text().replace(
    "Curve-only bypass has nothing to activation-hook; its exact projected",
    "Curve-only bypass has nothing to post-forward-hook; its exact projected",
)
curve.write_text(c)

# Bump staging metadata and make the release notes explicit about the v1.5.1
# real-workflow failure. The final release remains gated on a real GPU rerun.
pyproject = ROOT / "pyproject.toml"
p = pyproject.read_text()
p = replace_once(p, 'version = "1.5.1"', 'version = "1.5.2"', "version")
pyproject.write_text(p)

notes = ROOT / "RELEASE_NOTES.md"
n = notes.read_text()
header = '''# ComfyUI-VDN-H3 v1.5.2\n\nv1.5.2 replaces VDN's mutable-forward runtime bypass after the v1.5.1 hotfix\nstill hard-aborted in the real stacked RTX PRO 6000 workflow.\n\n## Runtime-bypass ownership\n\n- Ordinary VDN LoRA residuals use PyTorch forward **post-hooks**.\n- VDN does **not** replace, splice, save, or restore `module.forward`.\n- VDN does **not** use `ModelPatcher.weight_function` / `add_weight_wrapper`.\n- One post-hook is registered per affected module, with all VDN terms fused into\n  one exact low-rank residual for that module.\n- Registration handles are generation-owned across clone-shared models: a newer\n  VDN clone replaces the old registration, while stale ejects cannot remove the\n  newer generation.\n- Independently managed Comfy `BypassForwardHook` providers remain outside VDN's\n  ownership; VDN never enters their mutable forward chain.\n- Fused INT8 `mlp.fc2` and projected curve-AdaLN terms retain native Comfy patch\n  ownership where a module post-hook is not semantically available.\n\nThe latest production failure was reported asynchronously at core\n`LoRAAdapter.h` / `BypassForwardHook` during the first actual H3 call, so the\nvisible stack does not prove the originating CUDA kernel. This change therefore\nremoves the VDN-side shared mutable-forward topology rather than claiming a\nspecific CUDA kernel fix. Real GPU validation is required before release.\n\n'''
if not n.startswith("# ComfyUI-VDN-H3 v1.5.2"):
    n = header + n
notes.write_text(n)

readme = ROOT / "README.md"
rd = readme.read_text()
rd = rd.replace(
    "stack-safe bypass",
    "non-mutating post-forward bypass",
)
rd = rd.replace(
    "stack-safe `PatcherInjection`",
    "generation-owned `PatcherInjection` of PyTorch forward post-hooks",
)
rd = rd.replace(
    "Comfy `BypassForwardHook`",
    "PyTorch forward post-hook",
)
readme.write_text(rd)

print("staged v1.5.2 forward-post-hook bypass changes")
