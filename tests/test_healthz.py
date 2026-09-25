
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
        def answer(self, req): raise RuntimeError("CUDA out of memory. Tried to allocate 1.10 GiB.")
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
        class _T:
            def is_alive(self): return True
        thread = _T()
        queue = __import__("queue").Queue()
        batches = 12
        def answer(self, req):
            return {"model": req.model,
                    "answers": {"probe": {"type": "score", "score": 1.0, "confidence": 1.0,
                                          "legend": {"0": "no", "1": "yes"},
                                          "probabilities": {"0": 0.001, "1": 0.999}}},
                    "usage": {"input_tokens": 9, "output_tokens": 8}, "latency_ms": 21.4}
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

    S.app.state.server = MS()
    try:
        client = TestClient(S.app)
        txt = client.get("/metrics").text
        assert "kev_inference_errors_total " in txt
        assert "kev_queue_depth 3" in txt
        # cpu device: NO gpu gauges (guard by device, not by import success)
        assert "kev_gpu_memory_allocated_bytes" not in txt
    finally:
        del S.app.state.server
