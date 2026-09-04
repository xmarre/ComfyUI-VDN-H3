"""Pytest path setup for standalone, CI and in-place ComfyUI custom-node runs."""
import os
import sys
from pathlib import Path

_PACKAGE = Path(__file__).resolve().parents[1]
_DEFAULT_COMFY = Path(__file__).resolve().parents[3]
_COMFYUI_ROOT = Path(os.environ.get("COMFYUI_ROOT", _DEFAULT_COMFY)).resolve()
_OPENVDN_ROOT = os.environ.get("OPENVDN_ROOT")

for path in (str(_COMFYUI_ROOT), str(_PACKAGE), _OPENVDN_ROOT):
    if path and path not in sys.path:
        sys.path.insert(0, path)
