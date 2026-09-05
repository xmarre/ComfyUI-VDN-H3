from __future__ import annotations

from vdn_h3 import window


def test_block_mask_cache_is_bounded_lru():
    window._BM_CACHE.clear()
    values = []
    for index in range(window.MAX_BLOCK_MASK_CACHE + 3):
        value = object()
        values.append(value)
        window._block_mask_cache_put((index, "cuda:0"), value)

    assert len(window._BM_CACHE) == window.MAX_BLOCK_MASK_CACHE
    assert (0, "cuda:0") not in window._BM_CACHE
    assert (2, "cuda:0") not in window._BM_CACHE
    assert window._block_mask_cache_get((3, "cuda:0")) is values[3]

    window._block_mask_cache_put((99, "cuda:0"), object())
    assert (3, "cuda:0") in window._BM_CACHE
    assert (4, "cuda:0") not in window._BM_CACHE


def test_block_mask_cache_key_can_distinguish_cuda_devices():
    window._BM_CACHE.clear()
    zero = object()
    one = object()
    window._block_mask_cache_put((128, "cuda:0"), zero)
    window._block_mask_cache_put((128, "cuda:1"), one)
    assert window._block_mask_cache_get((128, "cuda:0")) is zero
    assert window._block_mask_cache_get((128, "cuda:1")) is one
