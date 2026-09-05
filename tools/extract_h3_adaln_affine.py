#!/usr/bin/env python3
"""Extract the tiny AdaLN pruning affine needed for released full-width H3 LoRAs.

The repaired pruned BF16 Comfy checkpoint can retain ``adaln_basis`` and
``adaln_mean`` even though native inference does not use them. Quantized derivatives
may omit those auxiliaries. This tool writes only the two small tensors to
``adaln_affine.safetensors`` and records the matching curve-table SHA-256 when the
source checkpoint also contains ``adaln_t_table``/``time_embedder.table``.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def tensor_hash(t: torch.Tensor) -> str:
    x = t.detach().to(device="cpu", dtype=torch.float32).contiguous()
    h = hashlib.sha256()
    h.update(str(tuple(x.shape)).encode())
    h.update(x.numpy().tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="matching pruned BF16/source safetensors")
    parser.add_argument(
        "output", type=Path, nargs="?", default=Path("adaln_affine.safetensors"))
    args = parser.parse_args()

    metadata = {}
    with safe_open(str(args.source), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        missing = [key for key in ("adaln_basis", "adaln_mean") if key not in keys]
        if missing:
            raise SystemExit(f"{args.source}: missing {missing}")
        basis = handle.get_tensor("adaln_basis").to(torch.float32).clone().contiguous()
        mean = handle.get_tensor("adaln_mean").to(torch.float32).clone().contiguous()
        table = None
        for key in ("adaln_t_table", "time_embedder.table"):
            if key in keys:
                table = handle.get_tensor(key).to(torch.float32).clone().contiguous()
                break

    if basis.ndim != 2 or mean.ndim != 1 or basis.shape[1] != mean.shape[0]:
        raise SystemExit(
            f"invalid affine shapes: basis={tuple(basis.shape)} mean={tuple(mean.shape)}")
    if table is not None:
        if table.ndim != 2 or table.shape[1] != basis.shape[0]:
            raise SystemExit(
                f"curve/affine mismatch: table={tuple(table.shape)} basis={tuple(basis.shape)}")
        metadata["adaln_table_sha256"] = tensor_hash(table)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file({"adaln_basis": basis, "adaln_mean": mean}, str(args.output), metadata=metadata)
    size = args.output.stat().st_size
    print(f"wrote {args.output} ({size} bytes)")
    if "adaln_table_sha256" in metadata:
        print(f"adaln_table_sha256={metadata['adaln_table_sha256']}")
    else:
        print("warning: source had no curve table; output has no automatic table identity")


if __name__ == "__main__":
    main()
