"""Scoped compatibility guard for Comfy's AIMDO model compiler.

Upstream VDN-H3 v1.4.3 identified a Comfy build family where the AIMDO
malloc-graph compiler cannot execute VDN-patched MiniMax-H3 forwards.  Comfy
currently exposes only a process-global ``args.disable_comfy_compiler`` switch,
so this module keeps the unavoidable mutation as narrow and reversible as
possible:

* detection is lazy and fail-open on older Comfy builds;
* a user-provided ``--disable-comfy-compiler`` setting is never changed;
* VDN-owned disables are reference-counted across nested/overlapping VDN
  wrappers and restored in ``finally``;
* no Comfy function is monkey-patched and no unload hook is installed.

The guard is installed around VDN's own DIFFUSION_MODEL wrapper, so the switch is
active before the native MiniMax-H3 forward asks ``model_prefetch`` whether to
start a malloc graph and is restored immediately after that wrapped forward.
"""
from __future__ import annotations

from contextlib import contextmanager
import functools
import logging
import threading

import comfy.cli_args

_log = logging.getLogger("comfy.vdn")
_lock = threading.RLock()
_active_owned_guards = 0
_warned = False


def _compiler_stack_present() -> bool:
    """Return whether this Comfy build contains the affected compiler stack."""
    try:
        args = comfy.cli_args.args
        if not hasattr(args, "disable_comfy_compiler"):
            return False
        import comfy.model_prefetch as model_prefetch

        # Match upstream v1.4.3's final detector. ``import comfy_aimdo.malloc_graph``
        # binds the package on current Comfy; older builds lack this compiler path.
        aimdo = getattr(model_prefetch, "comfy_aimdo", None)
        return aimdo is not None and hasattr(aimdo, "malloc_graph")
    except Exception:
        return False


@contextmanager
def disabled_for_vdn():
    """Temporarily disable Comfy's compiler for one VDN model forward.

    The CLI flag is process-global, so true concurrent non-VDN execution cannot be
    isolated by any consumer-side workaround.  Comfy's normal prompt executor is
    serialized; reference counting prevents nested/overlapping VDN wrappers from
    restoring the flag while another VDN forward still owns it.
    """
    global _active_owned_guards, _warned

    args = comfy.cli_args.args
    owns = False
    affected = _compiler_stack_present()
    if affected:
        with _lock:
            if _active_owned_guards > 0:
                _active_owned_guards += 1
                owns = True
            elif not bool(getattr(args, "disable_comfy_compiler", False)):
                args.disable_comfy_compiler = True
                _active_owned_guards = 1
                owns = True
                if not _warned:
                    _warned = True
                    _log.warning(
                        "[vdn] this Comfy build's AIMDO model compiler is incompatible "
                        "with VDN-H3; disabling it only for VDN model forwards")

    try:
        yield owns
    finally:
        if owns:
            with _lock:
                _active_owned_guards -= 1
                if _active_owned_guards == 0:
                    args.disable_comfy_compiler = False


def install_layout_guard() -> bool:
    """Wrap VDN's own layout-wrapper factory exactly once.

    ``vdn_h3.hybrid.apply_vdn`` resolves ``make_layout_wrapper`` from its module
    globals when Apply executes, so installing after node imports is sufficient and
    avoids modifying or wrapping any ComfyUI core callable.
    """
    from vdn_h3 import hybrid

    current = hybrid.make_layout_wrapper
    if getattr(current, "_vdn_compiler_guard_installed", False):
        return False

    @functools.wraps(current)
    def guarded_factory(state):
        inner = current(state)

        @functools.wraps(inner)
        def guarded(executor, *args, **kwargs):
            with disabled_for_vdn():
                return inner(executor, *args, **kwargs)

        return guarded

    guarded_factory._vdn_compiler_guard_installed = True
    hybrid.make_layout_wrapper = guarded_factory
    return True
