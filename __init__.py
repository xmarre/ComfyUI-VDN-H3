"""ComfyUI-VDN: Video Delta Net (VDN-H3) hybrid attention for MiniMax-H3.

Reference implementation: github.com/OpenVDN/vdn-minimax-h3 (Apache-2.0).
This package ports the released Video Delta Attention onto ComfyUI's native
MiniMax-H3 model as model patches; no ComfyUI core files are modified.
"""

import os
import sys

_PKG = os.path.dirname(__file__)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from vdn_h3 import apply as _apply_module
from vdn_h3.legacy_adapters import apply_adapters as _legacy_apply_adapters
from vdn_h3.compiler_guard import install_layout_guard as _install_layout_guard

# Production bisect: restore the adapter execution semantics that were validated
# before the v1.5/upstream reconciliation. Nodes import ``apply_adapters`` from
# vdn_h3.apply below, so replace only that public runtime entrypoint while leaving
# the newer implementation available for its isolated regression tests.
_apply_module.apply_adapters = _legacy_apply_adapters

from vdn_h3.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

_install_layout_guard()
del _install_layout_guard
del _legacy_apply_adapters
del _apply_module

# Frontend compatibility shim for legacy ApplyVDNH3Advanced positional workflows.
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
