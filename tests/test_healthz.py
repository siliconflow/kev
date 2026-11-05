
# --- healthz: liveness + real-inference readiness --------------------------------
# The 2026-09-25 production OOM was invisible to every existing check: /metrics and
# /v1/models (the only probes) do not traverse the model thread, so a 5% hard-error
# window on /v1/systemone coexisted with 4,288 green /metrics polls. Two endpoints
# close the gap:
#   GET /healthz       — process + model-thread liveness (cheap, for liveness probes)
#   GET /healthz/ready — one real inference through the same path traffic takes
#                        (encode -> queue -> model thread -> batch); 503 + the error
#                        when the instance cannot serve. Point readiness probes here.


def test_healthz_no_server_503():
    """Before main() there is no server: /healthz must 503, never 500-throw."""
    from fastapi.testclient import TestClient
    import kev.serve as S
    client = TestClient(S.app)
    if hasattr(S.app.state, "server"):
        del S.app.state.server   # a prior test may have set one
    r = client.get("/healthz")
    assert r.status_code == 503 and r.json()["ok"] is False
    assert "model thread" in r.json()["why"]


def test_healthz_reflects_model_thread_liveness():
    """/healthz is 200 exactly when the model thread exists and is alive."""
    from fastapi.testclient import TestClient
    import kev.serve as S
    import threading

    class FakeThread:
        def __init__(self, alive): self._a = alive
        def is_alive(self): return self._a

    class FS:  # minimal: healthz reads only .thread
        thread = None

    fs = FS()
    for label, alive, want in (("spawned", True, 200), ("dead-but-present", False, 503)):
        fs.thread = FakeThread(alive)
        S.app.state.server = fs
        try:
            client = TestClient(S.app)
            r = client.get("/healthz")
            assert r.status_code == want, f"{label}: {r.status_code} {r.text[:120]}"
            assert r.json()["ok"] is (want == 200)
        finally:
            del S.app.state.server


def test_healthz_ready_503s_when_inference_fails():
    """Readiness reflects the serving capability: a model thread that fails every batch
    (simulate: answer() raising, like the OOM batches of the incident) must surface as
    503 + a reason, and bump METRICS['errors'] — while nothing here touches /metrics."""
    from fastapi.testclient import TestClient
    import kev.serve as S

    class FailingServer:
        class _T:
            def is_alive(self): return True
        thread = _T()
        def submit(self, rec): raise RuntimeError("CUDA out of memory. Tried to allocate 1.10 GiB.")
    class FailingQueue:
        def qsize(self): return 37

    fs, fs_before_q = FailingServer(), None
    fs.queue = FailingQueue()
    S.app.state.server = fs
    errors_before = S.METRICS["errors"]
    try:
        client = TestClient(S.app, raise_server_exceptions=False)
        r = client.get("/healthz/ready")
        assert r.status_code == 503, r.text[:200]
        body = r.json()
        assert body["ok"] is False and "CUDA out of memory" in body["why"]
        assert body["queued"] == 37 and body["errors"] == errors_before + 1
        assert S.METRICS["errors"] == errors_before + 1   # counted at the source, visible in /metrics
    finally:
        S.METRICS["errors"] = errors_before
        del S.app.state.server


def test_healthz_ready_ok_contract():
    """A serving instance answers 200 with the batch/queue/latency context. Weightless:
    a stub server whose answer() returns the documented body shape."""
    from fastapi.testclient import TestClient
    import kev.serve as S

    class OKServer:
        from concurrent.futures import Future
        class _T:
            def is_alive(self): return True
        thread = _T()
        queue = __import__("queue").Queue()
        batches = 12
        def submit(self, rec):
            f = OKServer.Future(); f.set_result((
                [[0.001, 0.999]], {"tokens": 9, "state_tokens": 4, "latency_ms": 21.4, "prefix_cache_hit": False}))
            return f
        def _body(self, req, meta, ps, m):
            return {"model": req.model,
                    "answers": {"probe": {"type": "score", "score": 1.0, "confidence": 1.0,
                                          "legend": {"0": "no", "1": "yes"},
                                          "probabilities": {"0": 0.001, "1": 0.999}}},
                    "usage": {"input_tokens": 9, "output_tokens": 8}, "latency_ms": m["latency_ms"]}
    S.app.state.server = OKServer()
    try:
        client = TestClient(S.app)
        r = client.get("/healthz/ready")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True and body["latency_ms"] == 21.4 and body["batches"] == 12
        assert "queued" in body and "errors" in body and "probe_ms" in body
    finally:
        del S.app.state.server


def test_systemone_errors_counter_counts_inference_failures():
    """/v1/systemone: a batch-level error (the incident's 500s) increments kev_inference_errors_total
    at the source; HTTPException validation (422) must NOT count as an inference error."""
    from fastapi.testclient import TestClient
    import kev.serve as S

    class ErrServer:
        class _T:
            def is_alive(self): return True
        thread = _T()
        class _Q:
            def qsize(self): return 0
        queue = _Q()
        batches = 1
        async def answer_async(self, req): raise RuntimeError("batch failed")
    S.app.state.server = ErrServer()
    e0 = S.METRICS["errors"]
    try:
        client = TestClient(S.app, raise_server_exceptions=False)
        r = client.post("/v1/systemone", json={"state": "x", "model": "kev-latest",
                                               "questions": {"q": {"type": "choice", "instructions": "?", "criteria": {"a": "x", "b": "y"}}}})
        assert r.status_code == 500
        assert S.METRICS["errors"] == e0 + 1
        # a pydantic-422 (request shape) does not count: it never reaches the model layer
        r = client.post("/v1/systemone", json={"model": "kev-latest", "questions": {}})   # missing state + empty questions
        assert r.status_code == 422
        assert S.METRICS["errors"] == e0 + 1   # unchanged by the 422
    finally:
        S.METRICS["errors"] = e0
        del S.app.state.server


def test_healthz_fails_on_an_unrecoverable_error():
    """/healthz (liveness) turns 503 once the device context is poisoned: no retry in this process can serve
    again, so the platform restarting the container is the recovery (2026-11-04: 117 minutes of 503 readiness
    with a 200 liveness). The reason is the recorded error."""
    from fastapi.testclient import TestClient
    import kev.serve as S

    class Dead:
        class _T:
            def is_alive(self): return True
        thread = _T()
        fatal = "the device context is broken"
        busy_since = unready_since = None

    S.app.state.server = Dead()
    try:
        r = TestClient(S.app).get("/healthz")
        assert r.status_code == 503, r.text[:200]
        assert r.json()["ok"] is False and "unrecoverable accelerator error" in r.json()["why"]
        assert "the device context is broken" in r.json()["why"]
    finally:
        del S.app.state.server


def test_healthz_fails_when_nothing_is_served_for_the_unready_window():
    """The 600 s rule: readiness failing for KEV_UNREADY_MAX_S with no completed batch in that window fails
    liveness whatever the cause — while an instance still completing batches (pure overload) stays 200."""
    import time
    from fastapi.testclient import TestClient
    import kev.serve as S

    class Idle:
        class _T:
            def is_alive(self): return True
        thread = _T()
        fatal = ""
        busy_since = None

    s, now = Idle(), time.monotonic()
    S.app.state.server = s
    try:
        client = TestClient(S.app)
        s.unready_since = s.last_batch_at = now - S.UNREADY_MAX_S - 30   # unready, and nothing completed since
        r = client.get("/healthz")
        assert r.status_code == 503, r.text[:200]
        assert "no batch completed" in r.json()["why"]

        s.unready_since, s.last_batch_at = now - S.UNREADY_MAX_S - 30, now - 5   # unready, but batches still complete
        r = client.get("/healthz")
        assert r.status_code == 200, r.text[:200]
        assert r.json()["unready_s"] > S.UNREADY_MAX_S - 60             # the clock is exposed for diagnosis

        s.unready_since = None                                          # readiness passes again
        r = client.get("/healthz")
        assert r.status_code == 200 and r.json()["unready_s"] is None
    finally:
        del S.app.state.server


def test_healthz_fails_when_one_batch_is_wedged():
    """A model thread stuck inside one batch cannot report its own death: an in-flight batch older than
    KEV_STALL_S fails liveness (the batch is then killed by the container restart)."""
    import time
    from fastapi.testclient import TestClient
    import kev.serve as S

    class Wedged:
        class _T:
            def is_alive(self): return True
        thread = _T()
        fatal = ""
        unready_since = None
        last_batch_at = 0.0

    s = Wedged()
    S.app.state.server = s
    try:
        client = TestClient(S.app)
        s.busy_since = time.monotonic() - S.STALL_S - 10
        r = client.get("/healthz")
        assert r.status_code == 503 and "in flight" in r.json()["why"]
        s.busy_since = time.monotonic() - 1
        assert client.get("/healthz").status_code == 200
    finally:
        del S.app.state.server


def test_healthz_ready_short_circuits_after_an_unrecoverable_error():
    """With the context poisoned, readiness fails immediately without queueing a probe pass behind it, and it
    does not inflate the error counter (the cause is already on /healthz)."""
    from fastapi.testclient import TestClient
    import kev.serve as S

    class Dead:
        class _T:
            def is_alive(self): return True
        thread = _T()
        fatal = "the device context is broken"
        unready_since = None
        class _Q:
            def qsize(self): return 4
        queue = _Q()
        def submit(self, rec): raise AssertionError("readiness must not submit after a fatal error")

    s = Dead()
    S.app.state.server = s
    e0 = S.METRICS["errors"]
    try:
        r = TestClient(S.app, raise_server_exceptions=False).get("/healthz/ready")
        assert r.status_code == 503 and "unrecoverable accelerator error" in r.json()["why"]
        assert r.json()["queued"] == 4 and S.METRICS["errors"] == e0
        assert s.unready_since is not None
    finally:
        S.METRICS["errors"] = e0
        del S.app.state.server


def test_healthz_ready_starts_and_clears_the_unready_clock():
    """Every readiness 503 starts the clock /healthz reads; a passing probe clears it."""
    from fastapi.testclient import TestClient
    import kev.serve as S

    class Failing:
        class _T:
            def is_alive(self): return True
        thread = _T()
        fatal = ""
        unready_since = None
        class _Q:
            def qsize(self): return 0
        queue = _Q()
        def submit(self, rec): raise RuntimeError("CUDA out of memory. Tried to allocate 1.10 GiB.")

    f = Failing()
    S.app.state.server = f
    q0 = __import__("queue").Queue()
    e0 = S.METRICS["errors"]
    try:
        client = TestClient(S.app, raise_server_exceptions=False)
        assert client.get("/healthz/ready").status_code == 503
        assert f.unready_since is not None

        class OK:
            from concurrent.futures import Future
            class _T:
                def is_alive(self): return True
            thread = _T()
            fatal = ""
            unready_since = 1.0       # was unready; this pass clears it
            queue = q0
            batches = 1
            def submit(self, rec):
                x = OK.Future(); x.set_result(([[0.0, 1.0]],
                    {"tokens": 9, "state_tokens": 4, "latency_ms": 3.0, "prefix_cache_hit": False}))
                return x
            def _body(self, req, meta, ps, m):
                return {"model": req.model, "answers": {"probe": {"type": "score", "score": 1.0, "confidence": 1.0,
                        "legend": {"0": "no", "1": "yes"}, "probabilities": {"0": 0.0, "1": 1.0}}},
                        "usage": {"input_tokens": 9, "output_tokens": 8}, "latency_ms": m["latency_ms"]}
        ok = OK()
        S.app.state.server = ok
        assert client.get("/healthz/ready").status_code == 200
        assert ok.unready_since is None
    finally:
        S.METRICS["errors"] = e0
        del S.app.state.server


def test_healthz_ready_queue_full_starts_the_unready_clock():
    """A full queue is a 503 readiness too: a wedged thread fills the queue, and if that were not recorded the
    liveness watchdog would never see the instance as unready."""
    from fastapi.testclient import TestClient
    from fastapi import HTTPException
    import kev.serve as S

    class Full:
        class _T:
            def is_alive(self): return True
        thread = _T()
        fatal = ""
        unready_since = None
        class _Q:
            def qsize(self): return 64
        queue = _Q()
        def submit(self, rec):
            raise HTTPException(503, "queue full: 64 requests already waiting; retry shortly", headers={"Retry-After": "1"})

    s = Full()
    S.app.state.server = s
    e0 = S.METRICS["errors"]
    try:
        r = TestClient(S.app, raise_server_exceptions=False).get("/healthz/ready")
        assert r.status_code == 503 and "queue full" in r.json()["why"]
        assert s.unready_since is not None
    finally:
        S.METRICS["errors"] = e0
        del S.app.state.server


def test_metrics_exposes_errors_queue_and_gpu_dimensions():
    """/metrics gains the dimensions the incident needed: error counter, queue depth, and
    (on cuda) allocator/GPU-memory gauges. CPU hosts skip the gauges rather than fail."""
    from fastapi.testclient import TestClient
    import kev.serve as S

    class Mq:
        def qsize(self): return 3
    class MS:
        class _T:
            def is_alive(self): return True
        class _Ck:
            requested = "jaredpalmer/kev-4b"
            class meta:
                base = "Qwen/Qwen3.5-4B-Base"
        checkpoint = _Ck()
        model = None
        class _Mod:
            backend = "torch"
        model = _Mod()
        device = "cpu"
        thread = _T()
        queue = Mq()
        batches = 2
        batched_requests = 7
        lock = __import__("threading").Lock()
        prefix_cache = __import__("types").SimpleNamespace(hits=1, misses=2, size=4, min_tokens=384, entries=[])

    MS.latency = S.LatencyView()
    MS.latency.observe(21.4, 2)          # one batch of 2 requests @21.4ms
    S.app.state.server = MS()
    try:
        client = TestClient(S.app)
        txt = client.get("/metrics").text
        assert "kev_inference_errors_total " in txt
        assert "kev_queue_depth 3" in txt
        assert "kev_latency_ms_count 2" in txt
        assert "kev_latency_ms_sum 42.8" in txt
        assert f'kev_latency_ms_bucket{{le="25"}} 2' in txt      # 21.4 <= 25
        assert "kev_inferences_total 2" in txt                    # request-level count from the view
        assert "kev_fatal 0" in txt and "kev_unready_seconds 0" in txt
        assert "kev_pressure_sheds_total 0" in txt
        # cpu device: NO gpu gauges (guard by device, not by import success)
        assert "kev_gpu_memory_allocated_bytes" not in txt
    finally:
        del S.app.state.server


def test_metrics_exposes_the_graph_and_health_dimensions():
    """/metrics carries what only /v1/models JSON had: the CUDA graphs' resident buffer bytes, their capture
    counters and whether they are off — the 2026-11-04 card held ~3 GiB of buffers no scraper could see."""
    from fastapi.testclient import TestClient
    from types import SimpleNamespace
    import kev.serve as S

    stats = {"captured": 4, "kept": 3, "pending": 2, "failed": 5, "gated": 5,
             "sheds": 1, "disabled": True, "buffers_mib": 1024, "free_mib": 29}
    class MS:
        class _T:
            def is_alive(self): return True
        class _Ck:
            requested = "jaredpalmer/kev-4b"
            class meta:
                base = "Qwen/Qwen3.5-4B-Base"
        checkpoint = _Ck()
        model = SimpleNamespace(backend="torch", graphs=SimpleNamespace(stats=lambda: stats))
        device = "cpu"
        thread = _T()
        class _Q:
            def qsize(self): return 0
            maxsize = 64
        queue = _Q()
        batches = 5
        batched_requests = 9
        lock = __import__("threading").Lock()
        prefix_cache = __import__("types").SimpleNamespace(hits=0, misses=0, size=4, min_tokens=384, entries=[])
        pressure_sheds = 2
        fatal = "the device context is broken"
        unready_since = None

    m = MS()
    m.latency = S.LatencyView()
    S.app.state.server = m
    try:
        txt = TestClient(S.app).get("/metrics").text
        assert "kev_graph_buffers_bytes 1073741824" in txt          # 1024 MiB
        assert "kev_graph_captured 4" in txt and "kev_graph_pending 2" in txt
        assert "kev_graph_failed 5" in txt and "kev_graph_gated_total 5" in txt
        assert "kev_graph_disabled 1" in txt
        assert "kev_pressure_sheds_total 2" in txt and "kev_fatal 1" in txt
        assert "kev_unready_seconds 0" in txt
    finally:
        del S.app.state.server


def test_metrics_survives_a_failing_graph_stats():
    """/metrics is the one endpoint that must keep answering through the incident its health gauges exist for:
    a stats() that raises (the driver query on a destroyed context) costs the graph dimensions, not the scrape."""

    from fastapi.testclient import TestClient
    from types import SimpleNamespace
    import kev.serve as S

    def broken(): raise RuntimeError("CUDA error: the context is destroyed")
    class MS:
        class _T:
            def is_alive(self): return True
        class _Ck:
            requested = "jaredpalmer/kev-4b"
            class meta:
                base = "Qwen/Qwen3.5-4B-Base"
        checkpoint = _Ck()
        model = SimpleNamespace(backend="torch", graphs=SimpleNamespace(stats=broken))
        device = "cpu"
        thread = _T()
        class _Q:
            def qsize(self): return 0
            maxsize = 64
        queue = _Q()
        batches = 5
        batched_requests = 9
        prefix_cache = __import__("types").SimpleNamespace(hits=0, misses=0, size=4, min_tokens=384, entries=[])
        pressure_sheds = 2
        fatal = "the device context is broken"
        unready_since = None

    m = MS()
    m.latency = S.LatencyView()
    S.app.state.server = m
    try:
        r = TestClient(S.app).get("/metrics")
        assert r.status_code == 200
        assert "kev_fatal 1" in r.text and "kev_pressure_sheds_total 2" in r.text
        assert "kev_graph_buffers_bytes" not in r.text
    finally:
        del S.app.state.server


def test_healthz_ready_fails_on_a_malformed_probe_answer():
    """A probe pass that answers something other than a well-formed score is not ready: 503, and the unready clock
    keeps running (a 200 here would clear it, so the UNREADY_MAX_S restart could never fire for an instance whose
    every answer is nonsense)."""

    from concurrent.futures import Future
    from fastapi.testclient import TestClient
    import kev.serve as S

    class Nonsense:
        class _T:
            def is_alive(self): return True
        thread = _T()
        fatal = ""
        unready_since = None
        queue = __import__("queue").Queue()
        batches = 1
        def submit(self, rec):
            f = Future(); f.set_result(([[0.5, 0.5]], {"tokens": 9, "state_tokens": 4, "latency_ms": 3.0, "prefix_cache_hit": False}))
            return f
        def _body(self, req, meta, ps, m):
            return {"model": req.model, "answers": {"probe": {"type": "noul", "legend": {"0": "no"},
                    "probabilities": {"0": 0.5}}}, "usage": {}, "latency_ms": 3.0}

    s = Nonsense()
    S.app.state.server = s
    try:
        r = TestClient(S.app).get("/healthz/ready")
        assert r.status_code == 503, r.text[:200]
        assert r.json()["ok"] is False and "well-formed score" in r.json()["why"]
        assert s.unready_since is not None
    finally:
        del S.app.state.server
