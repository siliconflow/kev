"""Overload behaviour (2026-10-28 incident): an instance whose card is full must refuse work and say so,
and an OOM batch must shed what only costs speed instead of failing forever.

The log that motivated this: 269 POST /v1/systemone over 11 minutes, 262 of them 500 (CUDA OOM on a
2 MiB allocation), free device memory pinned at 1.81 MiB for the whole window, and every probe (GET
/metrics, GET /v1/models) green. Two gaps besides the probes (tests/test_healthz.py):
  - the queue was unbounded: requests piled up behind a batch that could never allocate;
  - kev.serve._run shed only the prefix cache, never the CUDA graphs' private pool (~1.5 GiB) or their
    shared buffers (~3 GiB), so nothing recovered without a restart.
"""


def test_enqueue_refuses_when_full():
    """A full queue refuses with 503 + Retry-After instead of blocking or accepting."""

    import queue
    from fastapi import HTTPException
    from kev.serve import enqueue

    q = queue.Queue(maxsize=1)
    enqueue(q, "first")
    try:
        enqueue(q, "second")
        raise AssertionError("the second put must be refused")
    except HTTPException as e:
        assert e.status_code == 503 and "queue full" in e.detail
        assert e.headers["Retry-After"] == "1"
    assert q.qsize() == 1


def test_enqueue_unbounded_at_zero():
    """maxsize 0 keeps the old unbounded behaviour (KEV_MAX_QUEUE=0)."""

    import queue
    from kev.serve import enqueue

    q = queue.Queue(maxsize=0)
    for i in range(1000):
        enqueue(q, i)
    assert q.qsize() == 1000


def test_submit_counts_refusals():
    """Server.submit refuses (and counts) when the queue is full, instead of waiting for a batch slot."""

    import queue, threading
    from fastapi import HTTPException
    import kev.serve as S

    class Model:
        def encode(self, tok, rec, max_state=None, max_branch=None): return {"ids": [1], "seg": [0]}

    s = object.__new__(S.Server)
    s.model, s.tok, s.stopping = Model(), None, threading.Event()
    s.queue, s.refusals = queue.Queue(maxsize=1), 0
    s.queue.put_nowait("someone else's request")
    try:
        s.submit({"state": "x", "questions": {}})
        raise AssertionError("a full queue must refuse")
    except HTTPException as e:
        assert e.status_code == 503 and e.headers["Retry-After"] == "1"
    assert s.refusals == 1 and s.queue.qsize() == 1


def _graph_stub(**kw):
    """A CudaGraphs stand-in with just the state release()/admits() look at."""
    import torch
    from types import SimpleNamespace
    return SimpleNamespace(graphs={"k": 1}, pending={"k": 1}, eager_runs={"k": 3}, failed={}, captures=5,
                           bank="bank", bank_views="views", rowbuf="rowbuf", hidden="hidden",
                           device=torch.device("cpu"), sheds=0, disabled=kw.pop("disabled", False), **kw)


def test_release_sheds_graphs_then_disables_at_the_cap(monkeypatch):
    """release() drops every captured graph, pool-derived bucket state, and (with buffers) the shared
    buffers; after GRAPH_SHEDS_MAX sheds graphs stay off for good."""

    import torch
    import kev.cuda_graphs as G

    g = _graph_stub(_free_bytes=lambda: 8 * 2**30)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: None)
    monkeypatch.setattr(G, "GRAPH_SHEDS_MAX", 2)

    G.CudaGraphs.release(g, buffers=True)
    assert not g.graphs and not g.pending and not g.eager_runs and not g.failed and g.captures == 0
    assert g.bank is None and g.bank_views is None and g.rowbuf is None and g.hidden is None
    assert g.sheds == 1 and g.disabled is False          # one shed may still recapture

    G.CudaGraphs.release(g, buffers=False)
    assert g.sheds == 2 and g.disabled is True           # nothing more to shed: eager serving stays on


def test_release_without_buffers_keeps_them():
    import torch
    import kev.cuda_graphs as G

    g = _graph_stub(_free_bytes=lambda: 8 * 2**30)
    orig_sync, orig_empty = torch.cuda.synchronize, torch.cuda.empty_cache
    torch.cuda.synchronize = lambda *a, **k: None
    torch.cuda.empty_cache = lambda *a, **k: None
    try:
        G.CudaGraphs.release(g, buffers=False)
        assert g.bank == "bank" and g.rowbuf == "rowbuf" and g.hidden == "hidden"
    finally:
        torch.cuda.synchronize, torch.cuda.empty_cache = orig_sync, orig_empty


def test_ensure_buffers_disables_when_rebuild_cannot_allocate(monkeypatch):
    """A rebuild that cannot allocate turns graphs off (eager serving continues) instead of raising."""

    import torch
    import kev.cuda_graphs as G

    class LM:
        class config:
            hidden_size = 2560

    g = _graph_stub()
    g.bank = g.bank_views = g.rowbuf = g.hidden = None
    g.lm, g.device, g.dtype, g.probe = LM(), torch.device("cpu"), torch.bfloat16, None

    def boom(*a, **k):
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 MiB.")

    monkeypatch.setattr(G, "Buffers", boom)
    assert G.CudaGraphs._ensure_buffers(g) is False
    assert g.disabled is True and g.bank is None
    assert G.CudaGraphs.admits(g, 4, [8], False) is False   # kev.model.probs_batch routes eagerly from here


def test_admits_false_when_disabled():
    from types import SimpleNamespace
    import kev.cuda_graphs as G

    g = _graph_stub(disabled=True)
    assert G.CudaGraphs.admits(g, 4, [8], False) is False


def test_run_sheds_cache_then_graphs_then_serves_eagerly(monkeypatch):
    """Server._run's OOM ladder: the prefix cache first, then the graphs + their buffers, then a third
    failure reaches the client. Here the third attempt succeeds and the batch is served."""

    import torch
    import kev.serve as S

    class Graphs:
        def __init__(self): self.releases = 0
        def release(self, buffers=False): self.releases += 1; self.buffers = buffers

    class Model:
        def __init__(self, fails): self.graphs, self.fails, self.calls = Graphs(), fails, 0
        def probs_batch(self, encs, cached, keep):
            self.calls += 1
            assert cached is None or len(cached) == len(encs)
            if self.calls <= self.fails:
                raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 MiB.")
            return [[torch.tensor([0.5, 0.5])] for _ in encs], [None] * len(encs)

    def run(fails):
        s = object.__new__(S.Server)
        s.device, s.model = "cpu", Model(fails)
        s.prefix_cache, s.latency = S.PrefixCache(4, 0), S.LatencyView()
        s.batches = s.batched_requests = s.graph_sheds = 0
        enc = {"ids": [1, 2, 3], "seg": [0, 0, 0]}
        return s, s._run([enc])

    s, out = run(fails=2)                        # both sheds, then the retry serves
    assert s.model.calls == 3 and s.batches == 1 and len(out) == 1
    assert s.model.graphs.releases == 2 and s.graph_sheds == 2
    assert s.model.graphs.buffers is True        # the buffer-bearing release is the one that matters

    s2 = object.__new__(S.Server)
    s2.device, s2.model = "cpu", Model(fails=3)
    s2.prefix_cache, s2.latency = S.PrefixCache(4, 0), S.LatencyView()
    s2.batches = s2.batched_requests = s2.graph_sheds = 0
    try:
        s2._run([{"ids": [1], "seg": [0]}])
        raise AssertionError("a third OOM must reach the client")
    except torch.OutOfMemoryError:
        pass
    assert s2.model.graphs.releases == 2 and s2.batches == 0


def test_submit_refuses_after_an_unrecoverable_error():
    """Once the device context is poisoned no retry can serve: submit refuses 503 + Retry-After without
    encoding or queueing, and does not count the refusal as queue backpressure."""

    import queue, threading
    from fastapi import HTTPException
    import kev.serve as S

    class Model:
        def encode(self, *a, **k): raise AssertionError("must not encode after a fatal error")

    s = object.__new__(S.Server)
    s.model, s.tok, s.stopping = Model(), None, threading.Event()
    s.queue, s.refusals, s.fatal = queue.Queue(maxsize=4), 0, "the device context is broken"
    try:
        s.submit({"state": "x", "questions": {}})
        raise AssertionError("a fatal error must refuse the request")
    except HTTPException as e:
        assert e.status_code == 503 and "unrecoverable accelerator error" in e.detail
        assert e.headers["Retry-After"] == "1"
    assert s.refusals == 0 and s.queue.qsize() == 0


def test_image_requests_are_refused_after_an_unrecoverable_error():
    """The vision tower runs on the same poisoned context: image requests get the same 503 + Retry-After as text
    ones instead of a 500 out of the dead device."""

    from fastapi import HTTPException
    import kev.serve as S

    class Served:
        fatal = "the device context is broken"
        vision = object()

    S.app.state.server = Served()
    try:
        S._probs_images({"state": "x"}, ["data:image/png;base64,AA"])
        raise AssertionError("an image request must be refused too")
    except HTTPException as e:
        assert e.status_code == 503 and "unrecoverable accelerator error" in e.detail
        assert e.headers["Retry-After"] == "1"
    finally:
        del S.app.state.server


def test_ease_pressure_sheds_cache_then_graph_buffers(monkeypatch):
    """_run sheds up front when the device is nearly out of free memory (2026-11-04: the allocation that
    failed at 1.9 MiB free poisoned the CUDA context): the cached states first, then the graphs. Graphs go off
    for good (final=True): admits() would rebuild released buffers for this very batch and spend the headroom
    just made. A card with room keeps both; off CUDA this never runs; nothing to shed is not a shed."""

    from types import SimpleNamespace
    import kev.serve as S

    released = []
    graphs = SimpleNamespace(bank="bank", disabled=False,
                             release=lambda buffers=False, **kw: released.append((buffers, kw)))
    s = object.__new__(S.Server)
    s.device, s.prefix_cache, s.pressure_sheds = "cpu", S.PrefixCache(4, 0), 0

    def off_cuda(): raise AssertionError("free_bytes must not be read off cuda")
    monkeypatch.setattr(S, "free_bytes", off_cuda)
    S.Server._ease_pressure(s, graphs)
    assert s.pressure_sheds == 0

    s.device = "cuda"
    monkeypatch.setattr(S, "free_bytes", lambda device: 1024 * 2**20)   # room: nothing to shed
    S.Server._ease_pressure(s, graphs)
    assert s.pressure_sheds == 0 and not released

    s.prefix_cache.entries[((1, 2), False)] = "cached-prefix"
    frees, emptied = iter([10 * 2**20, 400 * 2**20]), []
    monkeypatch.setattr(S, "free_bytes", lambda device: next(frees))
    monkeypatch.setattr(S, "empty_cache", lambda device: emptied.append(device))
    S.Server._ease_pressure(s, graphs)
    assert not s.prefix_cache.entries and emptied == ["cuda"]    # the cache alone made room
    assert s.pressure_sheds == 1 and not released

    monkeypatch.setattr(S, "free_bytes", lambda device: 10 * 2**20)   # nothing cached: the graphs go
    S.Server._ease_pressure(s, graphs)
    assert released == [(True, {"final": True, "why": "low device memory"})] and s.pressure_sheds == 2
    assert s.prefix_cache.oom_retries == 0            # a proactive shed is not an OOM retry

    released.clear()
    S.Server._ease_pressure(s, SimpleNamespace(bank=None, disabled=False, release=graphs.release))
    assert not released and s.pressure_sheds == 2     # nothing left to shed: no release, no counter bump

    def gone(device): raise RuntimeError("the device went away")
    monkeypatch.setattr(S, "free_bytes", gone)
    S.Server._ease_pressure(s, graphs)                # the batch's own path reports it, not the pre-flight

    def boom(buffers=False, **kw): raise RuntimeError("the driver went away")
    s.pressure_sheds = 3
    monkeypatch.setattr(S, "free_bytes", lambda device: 10 * 2**20)
    S.Server._ease_pressure(s, SimpleNamespace(bank="bank", disabled=False, release=boom))
    assert s.pressure_sheds == 3                      # a shed that itself failed is not counted, and does not raise


def test_capture_failure_turns_graphs_off_without_killing_the_thread(monkeypatch):
    """A capture that escapes capture_pending's per-bucket handling must not kill the model thread: graphs
    go off for good, their buffers are released (eager passes never rebuild them), eager serving continues,
    and only an unrecoverable error also fails liveness."""

    import threading
    import kev.serve as S

    released = []
    class Graphs:
        disabled = False
        def capture_pending(self, limit=None): raise RuntimeError("the capture machinery broke")
        def release(self, buffers=False, **kw):
            released.append((buffers, kw))
            if kw.get("final"): self.disabled = True   # CudaGraphs.release's rule: final turns them off for good

    s = object.__new__(S.Server)
    s.lock, s.fatal = threading.Lock(), ""
    g = Graphs()
    S.Server._capture(s, g)
    assert g.disabled is True and s.fatal == ""
    assert released == [(True, {"final": True, "why": "a failed capture (RuntimeError)"})]

    monkeypatch.setattr(S, "unrecoverable", lambda e: True)
    g2 = Graphs()
    S.Server._capture(s, g2)
    assert s.fatal and "the capture machinery broke" in s.fatal

    class DeadDriver(Graphs):
        def release(self, buffers=False, **kw): raise RuntimeError("the driver is gone")
    g3 = DeadDriver()
    S.Server._capture(s, g3)                          # the release fails too: the thread still lives, graphs still off
    assert g3.disabled is True


def test_run_eases_pressure_before_the_batch(monkeypatch):
    """The pre-flight shed runs before the batch allocates, not after it fails."""

    import torch
    import kev.serve as S

    order = []
    class Model:
        graphs = None
        def probs_batch(self, encs, cached, keep):
            order.append("batch")
            return [[torch.tensor([0.5])] for _ in encs], [None] * len(encs)

    s = object.__new__(S.Server)
    s.device, s.model = "cpu", Model()
    s.prefix_cache, s.latency = S.PrefixCache(4, 0), S.LatencyView()
    s.batches = s.batched_requests = s.graph_sheds = 0
    s._ease_pressure = lambda graphs: order.append("ease")
    s._run([{"ids": [1], "seg": [0]}])
    assert order == ["ease", "batch"] and s.batches == 1


def test_run_clears_the_prefix_cache_first(monkeypatch):
    """With cached states present the first retry drops only the cache (the #75 behaviour stays first)."""

    import torch
    import kev.serve as S

    class Graphs:
        def __init__(self): self.releases = 0
        def release(self, buffers=False): self.releases += 1

    class Model:
        graphs = None
        def __init__(self): self.calls, self.graphs = 0, Graphs()
        def probs_batch(self, encs, cached, keep):
            self.calls += 1
            if self.calls == 1:
                raise torch.OutOfMemoryError("CUDA out of memory")
            return [[torch.tensor([1.0])] for _ in encs], [None] * len(encs)

    s = object.__new__(S.Server)
    s.device, s.model = "cpu", Model()
    s.prefix_cache, s.latency = S.PrefixCache(4, 0), S.LatencyView()
    s.batches = s.batched_requests = s.graph_sheds = 0
    enc = {"ids": [1, 2], "seg": [0, 0]}
    s.prefix_cache.entries[((1, 2), False)] = "cached-prefix"
    out = s._run([enc])
    assert s.model.calls == 2 and len(out) == 1
    assert not s.prefix_cache.entries and s.prefix_cache.oom_retries == 1
    assert s.model.graphs.releases == 0          # the cache alone was enough
