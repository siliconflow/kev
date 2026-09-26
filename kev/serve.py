"""FastAPI sidecar for the playground: loads one checkpoint, exposes prefill-only decisions.

Run: uv run --extra serve python -m kev.serve --run runs/kev --port 8008

TypeSafe-compatible: POST /v1/systemone, GET /v1/models, the `x-typesafe-request-id` response header, and bearer auth
when KEV_API_KEY is set (unset = open server, the local default). Demo extras: POST /v1/systemone/permute (one Choice
under several option orders) and POST /v1/systemone/separate (each question in its own pass, for the packed-vs-separate
comparison). KEV_PREFIX_CACHE / KEV_PREFIX_MIN_TOKENS size the state-prefix cache; KEV_DATE_FACTS=1 opts into the
date preprocessing (api.with_date_facts). Backend and precision follow LoadOptions (KEV_BACKEND, KEV_DTYPE, ...): on Apple
Silicon the hybrid Qwen3.5 checkpoints run on MLX by default, elsewhere on torch in bf16.
"""
import argparse, asyncio, atexit, hmac, os, queue, random, sys, threading, time, uuid
from concurrent.futures import Future
import torch
from dataclasses import dataclass, field, replace
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from .api import SystemOneRequest, to_record, to_answers, output_tokens, with_date_facts
from .checkpoint import Checkpoint, LoadOptions, is_hub_id
from .device import default_device, sync
from .model import SERVE_MAX_BRANCH, SERVE_MAX_STATE

PREFIX_CACHE_SIZE = int(os.environ.get("KEV_PREFIX_CACHE", "4"))          # states kept (KV + hidden); 0 disables
PROBE_WAIT_S = float(os.environ.get("KEV_PROBE_WAIT_S", "5"))             # /healthz/ready caps its wait (a wedged model thread cannot hang the probe)
PREFIX_MIN_TOKENS = os.environ.get("KEV_PREFIX_MIN_TOKENS")               # states shorter than this are not cached; default = the model's prefix_min_tokens (0 for hybrid backbones and MLX, 384 for attention-only torch models)
DATE_FACTS = os.environ.get("KEV_DATE_FACTS", "0") == "1"
API_KEY = os.environ.get("KEV_API_KEY")                                  # unset = open server; set = require Authorization: Bearer <key>, as the TypeSafe clients always send
MAX_BATCH = 64                                                           # requests the model thread takes at once (kev.cuda_graphs splits them to fit its buffers)
MODEL_NAMES = ("kev-latest", "jev-latest")                               # both names serve this checkpoint; jev-latest is the TypeSafe SDK default model, so an unconfigured client works


@dataclass
class PrefixCache:
    """State prefixes kept across requests, least recently used first: (state token ids, option_isolation) -> prefix.
    States shorter than min_tokens are not cached. A batch keeps (copies) only the new states that will still be here
    after it, its last `size` distinct ones: the rest would be evicted by the batch itself."""
    size: int
    min_tokens: int
    entries: dict = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def plan(self, encs):
        """-> (key per request, None when its state is not cached; its cached prefix or None; whether to keep a new one)."""
        lengths = [enc["seg"].count(0) for enc in encs]
        keys = [(tuple(enc["ids"][:n]), bool(enc.get("option_isolation"))) if self.size and n >= self.min_tokens else None
                for enc, n in zip(encs, lengths)]
        survivors = set(list(dict.fromkeys(k for k in reversed(keys) if k is not None))[:self.size])
        return keys, [self.entries.get(k) if k is not None else None for k in keys], [k in survivors for k in keys]

    def store(self, keys, cached, prefixes):
        """Record hits and misses, and (re)insert the batch's prefixes in order: most recently used last."""
        for key, old, new in zip(keys, cached, prefixes):
            if key is None: continue
            self.hits += old is not None; self.misses += old is None
            if new is None: continue
            self.entries.pop(key, None); self.entries[key] = new
            while len(self.entries) > self.size: self.entries.pop(next(iter(self.entries)))

    def clear(self):
        self.entries.clear()


@dataclass
class Server:
    """The loaded checkpoint, the state-prefix cache (PrefixCache), and the one model thread that runs every forward pass.

    Request threads encode their record and queue it; the model thread takes everything queued when it becomes free and
    runs it as one batch (model.probs_batch: with CUDA graphs, shared state and row passes; otherwise one request at a
    time), then answers each request. It also captures pending CUDA graphs when the graphs say so (capture_due). `lock` is
    held around each batch and capture: hold it to use the model directly."""
    checkpoint: Checkpoint
    tok: object
    model: object
    device: str
    lock: threading.Lock = field(default_factory=threading.Lock)
    batches: int = 0
    batched_requests: int = 0
    release_date: str = field(default="")   # for the TypeSafe model card; resolved once (may ask the Hub)
    vision: object = None   # kev.vision.VisionHook under KEV_VISION=1 (set in main); None keeps image requests a 422
    latency: object = None   # LatencyView: request-level model-time observations for kev_latency_ms_*

    def __post_init__(self):
        self.release_date = self.release_date or self.checkpoint.release_date()
        self.latency = self.latency or LatencyView()
        self.prefix_cache = PrefixCache(PREFIX_CACHE_SIZE, int(PREFIX_MIN_TOKENS) if PREFIX_MIN_TOKENS else self.model.prefix_min_tokens)
        self.queue, self.stopping = queue.Queue(), threading.Event()
        # the model thread gives up the GIL at every CUDA sync and waits to get it back while the event loop parses and
        # answers requests; at Python's default 5 ms switch interval those waits stretched a batch's model time ~2x.
        # Process-wide, so close() puts it back.
        self.switch_interval = sys.getswitchinterval()
        sys.setswitchinterval(0.0005)
        self.thread = threading.Thread(target=self._work, name="kev-model", daemon=True)
        self.thread.start()
        atexit.register(self.close)   # a daemon thread killed inside a CUDA call at interpreter exit aborts the process

    def close(self):
        """Stop the model thread after its current batch; requests still queued fail, and so do later ones (submit)."""
        if self.stopping.is_set(): return
        self.stopping.set(); self.thread.join()
        while not self.queue.empty():
            self.queue.get_nowait()[1].set_exception(RuntimeError("the server stopped")); self.queue.task_done()
        sys.setswitchinterval(self.switch_interval)

    def submit(self, rec):
        """Queue one record for the model thread. -> a Future of (probabilities, stats). The state prefix (tokens up to the
        first question) is cached across requests, so a repeated state only pays for its question rows. latency_ms is the
        model time of the batch the request ran in (not its wait in the queue)."""
        if self.stopping.is_set(): raise HTTPException(503, "the server is stopping")
        try: enc = self.model.encode(self.tok, rec, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH)
        except ValueError as e: raise HTTPException(422, str(e))
        done = Future()
        self.queue.put((enc, done))
        return done

    def probs(self, rec):
        return self.submit(rec).result()

    def _work(self):
        graphs = getattr(self.model, "graphs", None)
        while not self.stopping.is_set():
            try: batch = [self.queue.get(timeout=0.05)]
            except queue.Empty:
                if graphs is not None and graphs.capture_due(idle=True):
                    with self.lock: graphs.capture_pending(limit=1)
                continue
            while len(batch) < MAX_BATCH:
                try: batch.append(self.queue.get_nowait())
                except queue.Empty: break
            try:
                with self.lock: results = self._run([enc for enc, _ in batch])
            except Exception as e:   # every request of the batch gets the error; the thread lives on
                results = [e] * len(batch)
            for (_, done), result in zip(batch, results):
                (done.set_exception if isinstance(result, Exception) else done.set_result)(result)
                self.queue.task_done()
            if graphs is not None and graphs.capture_due(idle=False):
                with self.lock: graphs.capture_pending(limit=1)

    def _run(self, encs):
        """One batch through model.probs_batch, with the prefix cache. -> per request (probs, stats)."""
        keys, cached, keep = self.prefix_cache.plan(encs)
        sync(self.device); t = time.time()
        ps, prefixes = self.model.probs_batch(encs, cached, keep)
        sync(self.device); dt = round((time.time() - t) * 1000, 1)
        self.prefix_cache.store(keys, cached, prefixes)
        self.batches += 1; self.batched_requests += len(encs)
        # per-request latency observation for kev_latency_ms_*: a request's share of the batch is
        # an honest request-level view (the batch runs as one fused pass; without separate streams
        # there is no per-row timer to read). Same state -> same number, which is exactly what a
        # latency histogram of "how long does serving this kind of request take" wants.
        self.latency.observe(dt, len(encs))
        return [([q.tolist() for q in p], {"tokens": len(enc["ids"]), "state_tokens": enc["seg"].count(0), "latency_ms": dt, "prefix_cache_hit": c is not None})
                for enc, p, c in zip(encs, ps, cached)]

    def wait_idle(self):
        """Block until every submitted request is answered and no CUDA graph waits to be captured (benchmarks, warm-up)."""
        self.queue.join()
        graphs = getattr(self.model, "graphs", None)
        while graphs is not None and graphs.capture_due(idle=True): time.sleep(0.01)
        with self.lock: pass                                   # a capture in progress finishes

    def answer(self, req):
        """The /v1/systemone response body for one request."""
        rec, meta = to_record(prepare(req))
        return self._answer_rec(rec, req, meta)

    async def answer_async(self, req):
        """answer() for the event loop: a request waiting on the model thread holds no worker thread, so a container takes
        as many concurrent requests as its batches can absorb (FastAPI runs sync endpoints on a 40-thread pool)."""
        rec, meta = to_record(prepare(req))
        if rec.get("images"):
            return self._answer_rec(rec, req, meta)   # the vision hook is synchronous; not one to hold a worker for
        return self._body(req, meta, *await asyncio.wrap_future(self.submit(rec)))

    def _answer_rec(self, rec, req, meta):
        """Image records (rec["images"], set by api.to_record from state.images) take the vision hook under
        this Server's lock; text records go through the batched model thread. One dispatch for every caller
        (systemone, permute, separate, direct answer()) - the hook answers 422 when no tower is attached."""
        if rec.get("images"):
            ps, m = _probs_images({k: v for k, v in rec.items() if k != "images"}, rec["images"])
            return self._body(req, meta, ps, m)
        return self._body(req, meta, *self.probs(rec))

    def _body(self, req, meta, ps, m):
        answers = to_answers(ps, meta)
        return {"model": req.model, "answers": answers, "usage": {"input_tokens": m["tokens"], "output_tokens": output_tokens(self.tok, answers)}, "latency_ms": m["latency_ms"]}


def prepare(req):
    """Opt-in preprocessing applied to every request before the model sees it."""
    return req.model_copy(update={"state": with_date_facts(req.state)}) if DATE_FACTS else req


app = FastAPI(title="kev")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], expose_headers=["x-typesafe-request-id", "server-timing"])


@app.middleware("http")
async def typesafe(request, call_next):
    """Bearer auth (when API_KEY is set) and the request id every TypeSafe client reads off the response."""
    started = time.perf_counter()
    if API_KEY and request.url.path.startswith("/v1") and not hmac.compare_digest(request.headers.get("authorization", ""), f"Bearer {API_KEY}"):
        resp = JSONResponse({"detail": "missing or invalid API key; send Authorization: Bearer <KEV_API_KEY>"}, 401, {"www-authenticate": "Bearer"})
    else:
        resp = await call_next(request)
    resp.headers["x-typesafe-request-id"] = request.headers.get("x-typesafe-request-id") or uuid.uuid4().hex
    resp.headers["server-timing"] = f"app;dur={(time.perf_counter() - started) * 1000:.1f}"   # time inside this process, for telling it from the network
    return resp


def server() -> Server:
    return app.state.server


METRICS = {"latency_ms": [], "requests": 0, "errors": 0}   # process-wide serving counters for /metrics (image path + API gateway view); errors counts failed inferences (5xx at the source)

_LAT_BUCKETS = [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]   # ms; bracket kev's ~30ms-1s prefill range


class LatencyView:
    """Request-level latency observations for /metrics (kev_latency_ms_*): every served request
    contributes its (batch-share) model time — the text batch path records one batch observation
    for each request it carried, the image path one per call. Keeps count/sum/buckets so the
    exposition needs no scan of a growth-unbounded list."""

    def __init__(self):
        self.count, self.sum_ms, self.buckets = 0, 0.0, [0] * len(_LAT_BUCKETS)

    def observe(self, ms, n=1):
        self.count += n; self.sum_ms += ms * n
        for i, b in enumerate(_LAT_BUCKETS):
            if ms <= b: self.buckets[i] += n

    def expose(self, prefix):
        if not self.count: return []
        lines = [f"{prefix}_count {self.count}", f"{prefix}_sum {self.sum_ms:.1f}"]
        lines += [f'{prefix}_bucket{{le="{b}"}} {c}' for b, c in zip(_LAT_BUCKETS, self.buckets)]
        lines.append(f'{prefix}_bucket{{le="+Inf"}} {self.count}')
        return lines


def _probs_images(rec, refs):
    """Image path (rec has no "images" key here): decode refs -> PIL, forward through the
    attached hook (kev.vision.VisionHook.probs_with_images). Independent of the Server's text
    machinery - the batched model thread and the state-prefix cache do not apply (the state
    segment now holds tower rows, so a cached text-state KV would be wrong). Runs under the
    Server's lock so it never interleaves with a text batch."""
    s = server()
    if s.vision is None:
        raise HTTPException(422, "image requests need KEV_VISION=1 and a base with a vision tower")
    from .vision import decode_image
    try:
        images = [decode_image(r) for r in refs[:4]]
    except ValueError as e:
        raise HTTPException(422, str(e))
    with s.lock:
        sync(s.device); t = time.time()
        try:
            ps, m = s.vision.probs_with_images(rec, images, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH)
        except ValueError as e:
            raise HTTPException(422, str(e))
        sync(s.device); dt = time.time() - t
    s.latency.observe(dt * 1000)   # one request-level observation per image call (s = server() above)
    return [p.tolist() for p in ps], {"tokens": m["tokens"], "state_tokens": m["state_tokens"], "latency_ms": round(dt * 1000, 1), "prefix_cache_hit": False}


@app.get("/healthz")
def healthz():
    """Liveness: the process is up and the model thread exists. Deep checks live at /healthz/ready.

    A probe on /healthz alone cannot see an inference-path failure: the batched model
    thread can be erroring every forward pass (OOM, a broken fused kernel) while every
    GET endpoint stays 200 — exactly the 2026-09-25 production incident, where 4,288
    /metrics polls returned 200 across a 5% hard-error window on /v1/systemone. The
    platform's probes must therefore cover both."""
    s = app.state.server if hasattr(app, "state") and hasattr(app.state, "server") else None
    if s is None or not getattr(s, "thread", None) or not s.thread.is_alive():
        return JSONResponse({"ok": False, "why": "model thread not running"}, status_code=503)
    return {"ok": True, "model_thread_alive": True}


@app.get("/healthz/ready")
def healthz_ready():
    """Readiness: one real inference through the same path every request takes (encode -> queue ->
    model thread -> batch). This is the check that "reflects the instance's serving capability": a
    CUDA OOM, a fused-kernel crash or a wedged model thread surfaces here as 503 + the error,
    while /metrics and /v1/models stay green.

    The probe record is minimal and fixed (a short state, one two-option score question); it
    waits in the same queue as traffic, so it also reflects queue backlog. It costs one small
    forward pass (~20 ms on a 4B) — probing every 10-30 s keeps that well under 1% of capacity."""
    if not hasattr(app, "state") or not hasattr(app.state, "server"):
        return JSONResponse({"ok": False, "why": "no server"}, status_code=503)
    req = SystemOneRequest.model_validate({"state": "Health probe: readiness check.",
                                           "questions": {"probe": {"type": "score", "instructions": "healthy?",
                                                                   "criteria": ["no", "yes"]}}})
    t0 = time.perf_counter()
    try:
        s = app.state.server
        # one real inference through the same dispatch every request takes; bounded wait so a dead
        # model thread (future never completing) turns into a 503 probe failure, not a hung probe
        rec, meta = to_record(prepare(req))
        ps, m = s.submit(rec).result(timeout=PROBE_WAIT_S)
        body = s._body(req, meta, ps, m)
        answers = body["answers"]["probe"]
        ok = answers["type"] == "score" and answers["legend"] == {"0": "no", "1": "yes"} \
             and sum(answers["probabilities"].values()) > 0.99
        return {"ok": ok, "latency_ms": body["latency_ms"], "probe_ms": round((time.perf_counter() - t0) * 1000, 1),
                "queued": s.queue.qsize(), "batches": s.batches, "errors": METRICS["errors"]}
    except Exception as e:
        METRICS["errors"] += 1
        s = getattr(app.state, "server", None)
        q = getattr(getattr(s, "queue", None), "qsize", lambda: -1)()
        return JSONResponse({"ok": False, "why": repr(e)[:300], "queued": q,
                             "errors": METRICS["errors"]}, status_code=503)


@app.get("/metrics")
def metrics():
    """Prometheus text exposition (the platform's prometheusScraper polls this); no client dependency.
    kev_latency_ms_* is the request-level latency view (every served request: text batches and the
    image path); kev_batches_*/kev_prefix_cache_* describe the Server that produced them."""
    from fastapi.responses import PlainTextResponse
    s = app.state.server if hasattr(app, "state") and hasattr(app.state, "server") else None
    lines = [f"kev_requests_total {METRICS['requests']}"]
    # request-level latency: every served request (text batch + image path) contributes its
    # model time — the revival of the latency metric under its request-level name. The old
    # kev_inference_latency_ms_* (pre-batching, image-path-only by the end) is superseded.
    if s is not None:
        lines += s.latency.expose("kev_latency_ms")
        lines.append(f"kev_inferences_total {s.latency.count}")
    else:
        lines.append("kev_inferences_total 0")
    if s is not None:
        lines += [f"kev_batches_total {s.batches}", f"kev_batched_requests_total {s.batched_requests}",
                  f"kev_prefix_cache_hits_total {s.prefix_cache.hits}", f"kev_prefix_cache_misses_total {s.prefix_cache.misses}"]
    model_info = {"run": (s.checkpoint.requested if s else "") or "", "base": (s.checkpoint.meta.base if s else "") or "",
                  "device": (s.device if s else "") or "", "backend": (getattr(s, "model", None) and s.model.backend or "") if s else ""}
    lines.append("kev_model_info{" + ", ".join(f'{k}="{v}"' for k, v in model_info.items()) + "} 1")
    lines.append(f"kev_inference_errors_total {METRICS['errors']}")
    if s is not None:
        lines.append(f"kev_queue_depth {s.queue.qsize()}")
        if s.device == "cuda":
            a, r = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
            lines += [f"kev_gpu_memory_allocated_bytes {a}", f"kev_gpu_memory_reserved_bytes {r}",
                      f"kev_gpu_memory_free_bytes {torch.cuda.get_device_properties(0).total_memory - r}"]
    return PlainTextResponse("\n".join(lines) + "\n")


@app.post("/v1/systemone")
async def systemone(req: SystemOneRequest):
    """TypeSafe-compatible endpoint: typed questions in, typed answers out, one prefill pass."""
    METRICS["requests"] += 1
    try:
        return await server().answer_async(req)
    except HTTPException:
        raise
    except Exception as e:
        # a failed batch lands each of its requests here (OOM, a broken kernel): count it at the
        # source so /metrics and /healthz can see an inference-path failure the same moment the
        # client does. The 500 to the client stays untouched (uvicorn raises it after we return None
        # only never — re-raise keeps FastAPI's default 500 with our counter incremented first).
        METRICS["errors"] += 1
        raise


class PermuteSystemOne(BaseModel):
    request: SystemOneRequest
    question: str
    n_perm: int = Field(default=6, ge=1, le=64)   # each order is a forward pass; 0 divided by nothing, unbounded counts ran forever (#30)
    seed: int = 0


@app.post("/v1/systemone/permute")
def systemone_permute(r: PermuteSystemOne):
    """Re-run one Choice question under n_perm option orders. Returns per-order probabilities keyed by option name."""
    q = r.request.questions.get(r.question)
    if q is None or q.type != "choice": raise HTTPException(422, "question must be an existing choice question")
    rng = random.Random(r.seed); keys = list(q.criteria); runs = []
    for i in range(r.n_perm):
        order = list(keys)
        if i > 0: rng.shuffle(order)
        one = r.request.model_copy(update={"questions": {r.question: q.model_copy(update={"criteria": {k: q.criteria[k] for k in order}})}})
        resp = server().answer(one); a = resp["answers"][r.question]
        runs.append({"order": order, "probabilities": a["probabilities"], "choice": a["choice"], "latency_ms": resp["latency_ms"]})
    spread = {k: max(x["probabilities"][k] for x in runs) - min(x["probabilities"][k] for x in runs) for k in keys}
    return {"runs": runs, "argmax_stable": len({x["choice"] for x in runs}) == 1, "spread": spread}


@app.post("/v1/systemone/separate")
def systemone_separate(req: SystemOneRequest):
    """Answer each question in its own request against the same state (N passes). For packed-vs-separate comparison."""
    parts = [server().answer(req.model_copy(update={"questions": {qid: q}})) for qid, q in req.questions.items()]
    answers = {qid: a for p in parts for qid, a in p["answers"].items()}
    return {"model": req.model, "answers": answers,
            "usage": {"input_tokens": sum(p["usage"]["input_tokens"] for p in parts), "output_tokens": output_tokens(server().tok, answers)},
            "latency_ms": round(sum(p["latency_ms"] for p in parts), 1)}


@app.get("/v1/models")
def models():
    """One TypeSafe model card (name, description, release_date) per accepted model name, plus the Kev serving details
    a client may ignore: the run, the base, the device, the backend and precision, the temperature, prefix-cache stats."""
    s = server()
    ck, meta = s.checkpoint, s.checkpoint.meta
    card = {"description": f"Kev pointer head on {meta.base}, serving {ck.requested} at temperature {s.model.head.temperature:.2f}",
            "release_date": s.release_date,
            "run": ck.requested, "base": meta.base, "lora": meta.lora, "device": s.device, "backend": s.model.backend, "dtype": s.model.dtype,
            "temperature": s.model.head.temperature,
            "cuda_graphs": graphs.stats() if (graphs := getattr(s.model, "graphs", None)) else None,
            "prefix_cache": {"size": s.prefix_cache.size, "min_state_tokens": s.prefix_cache.min_tokens, "hits": s.prefix_cache.hits,
                             "misses": s.prefix_cache.misses, "cached_states": len(s.prefix_cache.entries)},
            "batches": {"count": s.batches, "requests": s.batched_requests, "queued": s.queue.qsize()}}
    return {"models": [{"name": name, **card} for name in MODEL_NAMES]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/kev")
    ap.add_argument("--fallback", default="runs/smoke")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--host", default="0.0.0.0")   # 0.0.0.0: K8s probes/gateway reach the pod IP; override with 127.0.0.1 for local use
    a = ap.parse_args()
    run = a.run if is_hub_id(a.run) or os.path.exists(f"{a.run}/head.pt") else a.fallback
    if run != a.run: print(f"{a.run} not found, falling back to {run}")
    dev = default_device()
    opts = LoadOptions.from_env()
    if dev == "mps" and opts.attn is None: opts = replace(opts, attn="sdpa")   # serving default on Apple GPUs (parity measured)
    if dev != "cpu" and opts.dtype is None: opts = replace(opts, dtype=torch.bfloat16)   # serving default: 2-4.5x faster than fp32 on an L4, same answers (LoadOptions.dtype); KEV_DTYPE=fp32 for the exact path
    if dev == "cuda" and opts.cuda_graphs is None: opts = replace(opts, cuda_graphs=True)   # serving default: a pass is ~2,000 kernel launches, so replaying graphs cuts warm latency several-fold (kev.cuda_graphs); KEV_CUDA_GRAPHS=0 to decline
    if dev == "cuda" and opts.fused is None: opts = replace(opts, fused=True)   # serving default: fused Qwen3.5 kernels, ~1/3 less GPU time per batch (kev.fused_qwen35); KEV_FUSED=0 to decline
    if opts.backend is None: opts = replace(opts, backend="auto")   # serving default: MLX for the hybrid Qwen3.5 checkpoints on Apple Silicon (LoadOptions.backend); KEV_BACKEND=torch to decline
    ck = Checkpoint(run)
    tok, model = ck.load(dev, opts)
    app.state.server = Server(ck, tok, model, dev)
    if os.environ.get("KEV_VISION") == "1":
        from .vision import attach
        app.state.server.vision = attach(model, tok, ck.meta.base,
                                         ck.meta.base_revision if os.environ.get("KEV_BASE_HUB") != "modelscope" else None)
    else:
        app.state.server.vision = None
    print(f"serving {ck.requested} ({ck.path}) on {dev} via {model.backend} ({model.dtype}) :{a.port}")   # /v1/models reports the run as given, not the resolved cache path
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
