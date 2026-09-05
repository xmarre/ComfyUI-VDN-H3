from __future__ import annotations

from vdn_h3 import runtime


def _reset():
    runtime._RECORD_STREAM_NEEDED = None


def test_record_stream_skipped_for_cuda_malloc_async(monkeypatch):
    _reset()
    monkeypatch.setattr(runtime.torch.cuda, "get_allocator_backend", lambda: "cudaMallocAsync")
    assert runtime._record_stream_needed() is False
    # Cached decision must not re-query or change mid-run.
    monkeypatch.setattr(runtime.torch.cuda, "get_allocator_backend", lambda: "native")
    assert runtime._record_stream_needed() is False


def test_record_stream_retained_for_native_allocator(monkeypatch):
    _reset()
    monkeypatch.setattr(runtime.torch.cuda, "get_allocator_backend", lambda: "native")
    assert runtime._record_stream_needed() is True


def test_record_stream_query_failure_fails_conservative(monkeypatch):
    _reset()

    def fail():
        raise RuntimeError("allocator API unavailable")

    monkeypatch.setattr(runtime.torch.cuda, "get_allocator_backend", fail)
    assert runtime._record_stream_needed() is True


def test_prefetch_record_stream_is_noop_under_cuda_malloc_async(monkeypatch):
    _reset()
    monkeypatch.setattr(runtime.torch.cuda, "get_allocator_backend", lambda: "cudaMallocAsync")

    class FakeTensor:
        def record_stream(self, stream):
            raise AssertionError("record_stream must not be called under cudaMallocAsync")

    runtime._StreamPrefetcher._record_stream(FakeTensor(), object())
