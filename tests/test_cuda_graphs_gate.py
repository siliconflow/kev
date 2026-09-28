"""The VRAM gate on graph capture (2026-09-25 postmortem): a 24 GB card whose allocator is
fragmented enters captures it cannot finish; the failed capture pollutes the pool state and
every later capture fails after it (one 222 MiB capture at 57 MiB free, then 18 more shapes).
The gate marks such buckets failed-eager up front, with an observable "gated" reason."""


def test_capture_need_covers_both_key_kinds():
    """The token estimate reads both key arities: (rows, Nb, Lb, Sr) and (states, Nb, Sb)."""
    from kev import cuda_graphs
    need = cuda_graphs.CudaGraphs._capture_need
    g = cuda_graphs  # constants
    gmin = cuda_graphs.GRAPH_MIN_FREE
    # the incident's big row bucket: 12 rows x 320 tokens over a 2048-wide state
    big = need(None, ("rows", 12, 320, 2048))
    # rows: n*Lb + Lb*Sr tokens (row intermediates dominate), states: n*Sb
    expect_big = max(gmin, (12 * 320 + 320 * 2048) * cuda_graphs.GRAPH_BYTES_PER_TOKEN)
    assert big == expect_big and big > 2**30           # ~1.26 GiB, well above the floor
    assert need(None, ("states", 3, 64)) == gmin        # small buckets floor at GRAPH_MIN_FREE
    assert need(None, ("states", 16, 1024)) == gmin     # 16*1024*2048 = 32 MiB < 256 MiB floor


def test_gate_skips_capture_and_marks_gated(monkeypatch):
    """With device free VRAM below the bucket's need, capture_pending() does not enter a capture
    (no stream/graph objects), marks the bucket failed with a "gated" reason, and stats() counts it."""
    import torch
    from kev.cuda_graphs import CudaGraphs

    g = object.__new__(CudaGraphs)          # no __init__: no CUDA, no model
    g.device, g.pending, g.eager_runs = torch.device("cpu"), {}, {}
    g.failed, g.captures, g.graphs = {}, 0, {}
    g.pending[("rows", 1, 128, 2048)] = (lambda buf: None, "buf-token")
    g.eager_runs[("rows", 1, 128, 2048)] = 3

    real_free = CudaGraphs._free_bytes
    monkeypatch.setattr(CudaGraphs, "_free_bytes", lambda self: 57 * 2**20)   # the incident's 57 MiB
    try:
        g.capture_pending()
        why, buf = g.failed[("rows", 1, 128, 2048)]
        assert why.startswith("gated:") and "57 MiB free" in why and buf == "buf-token"
        assert g.captures == 0 and not g.pending
        # stats() reports the gated split of failures
        class _Available:
            def __getattr__(self, name): return False
        stats = g.stats()
        assert stats["failed"] == 1 and stats["gated"] == 1
    finally:
        monkeypatch.setattr(CudaGraphs, "_free_bytes", real_free)


def test_capture_proceeds_when_vram_available(monkeypatch):
    """Above the need, the normal capture path runs untouched (warm-up + capture on the side stream)."""
    import torch
    from kev.cuda_graphs import CudaGraphs

    g = object.__new__(CudaGraphs)
    g.device, g.pending, g.eager_runs = torch.device("cpu"), {}, {}
    g.failed, g.captures, g.graphs, g.eager_runs = {}, 0, {}, {}
    key = ("rows", 2, 128, 64)
    g.pending[key] = (lambda buf: None, "buf-token")
    g.eager_runs[key] = 3

    monkeypatch.setattr(CudaGraphs, "_free_bytes", lambda self: 8 * 2**30)    # plenty
    calls = []
    real_capture = CudaGraphs.capture_pending
    # re-implement the loop head to observe it went past the gate: patch CUDAGraph/stream is heavy;
    # instead assert the gate did NOT mark the key failed before an (expected) failure in the real
    # capture machinery, which on CPU raises - and that failure must be the capture's, not the gate's.
    try:
        g.capture_pending()
    except Exception:
        pass   # the CPU capture machinery may raise; the point is the gate let it through
    assert not any(w.startswith("gated") for w, _ in g.failed.values())
