from __future__ import annotations

import torch

from vdn_h3 import runtime


class _Recorded:
    def __init__(self):
        self.streams = []

    def record_stream(self, stream):
        self.streams.append(stream)


def test_prefetch_records_consumer_stream_without_allocator_special_case():
    stream = object()
    tensor = _Recorded()

    runtime._StreamPrefetcher._record_stream(tensor, stream)

    assert tensor.streams == [stream]


def test_prefetch_records_consumer_stream_even_when_cuda_malloc_async(monkeypatch):
    """cudaMallocAsync still tracks non-creation usage streams in PyTorch.

    The prefetch producer stream and model consumer stream are different, so VDN must
    record the consumer regardless of allocator backend. This test deliberately makes
    the allocator query return cudaMallocAsync; _record_stream must not branch on it.
    """
    monkeypatch.setattr(
        runtime.torch.cuda,
        "get_allocator_backend",
        lambda: "cudaMallocAsync",
        raising=False,
    )
    stream = object()
    tensor = _Recorded()

    runtime._StreamPrefetcher._record_stream(tensor, stream)

    assert tensor.streams == [stream]


def test_prefetch_records_quantized_backing_storages(monkeypatch):
    stream = object()
    calls = []

    def record_tensor(self, got_stream):
        calls.append((id(self), got_stream))

    monkeypatch.setattr(torch.Tensor, "record_stream", record_tensor, raising=True)

    class Params:
        pass

    wrapper = _Recorded()
    wrapper._qdata = torch.empty(1)
    wrapper._params = Params()
    wrapper._params.scale = torch.empty(1)
    wrapper._params.orig_weight = torch.empty(1)
    wrapper._params.bias = torch.empty(1)

    expected_children = {
        id(wrapper._qdata),
        id(wrapper._params.scale),
        id(wrapper._params.orig_weight),
        id(wrapper._params.bias),
    }

    runtime._StreamPrefetcher._record_stream(wrapper, stream)

    assert wrapper.streams == [stream]
    assert {tensor_id for tensor_id, got_stream in calls if got_stream is stream} == expected_children
