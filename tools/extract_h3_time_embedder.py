#!/usr/bin/env python3
"""Extract the four MiniMax-H3 dense time-embedder tensors needed for exact
full-width AdaLN LoRA evaluation on a curve/pruned H3 base.

Usage:
    python tools/extract_h3_time_embedder.py \
        /path/to/dense_h3.safetensors \
        /path/to/models/vdn/stage-dmd-.../dense_time_embedder.safetensors

The output contains model weights and therefore retains the source checkpoint's model
license/terms.  Do not redistribute it merely because this repository's code is
Apache-2.0.
"""
from __future__ import annotations

import argparse
import json
import os
import struct

import torch
from safetensors import safe_open
from safetensors.torch import save_file

KEYS = (
    "time_embedder.proj_in.weight",
    "time_embedder.proj_in.bias",
    "time_embedder.proj_out.weight",
    "time_embedder.proj_out.bias",
)


def header(path):
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise ValueError("truncated safetensors file")
        size = struct.unpack("<Q", raw)[0]
        if size <= 0 or size > (64 << 20):
            raise ValueError(f"invalid header size {size}")
        return json.loads(fh.read(size))


def find_prefix(h):
    if any(k.endswith("adaln_t_table") for k in h):
        raise ValueError("input is itself a curve/pruned H3 checkpoint; use the matching dense base")
    for key in h:
        if key.endswith(KEYS[2]):
            prefix = key[: -len(KEYS[2])]
            if all(prefix + suffix in h for suffix in KEYS):
                return prefix
    raise ValueError("could not find a complete MiniMax-H3 time_embedder in the input")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="matching dense MiniMax-H3 .safetensors checkpoint")
    parser.add_argument("output", help="dense_time_embedder.safetensors destination")
    args = parser.parse_args()

    source = os.path.realpath(args.source)
    prefix = find_prefix(header(source))
    tensors = {}
    with safe_open(source, framework="pt", device="cpu") as handle:
        for suffix in KEYS:
            tensors[suffix] = handle.get_tensor(prefix + suffix).clone()

    os.makedirs(os.path.dirname(os.path.realpath(args.output)), exist_ok=True)
    save_file(
        tensors,
        args.output,
        metadata={
            "source": source,
            "purpose": "ComfyUI-VDN-H3 exact curve AdaLN reconstruction",
        },
    )
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"wrote {args.output}: {len(tensors)} tensors, {total / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
