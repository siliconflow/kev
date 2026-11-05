"""FastAPI sidecar for the playground: loads one checkpoint, exposes prefill-only decisions.

Run: uv run --extra serve python -m kev.serve --run runs/kev --port 8008

TypeSafe-compatible: POST /v1/systemone, GET /v1/models, the `x-typesafe-request-id` response header, and bearer auth
when KEV_API_KEY is set (unset = open server, the local default). Demo extras: POST /v1/systemone/permute (one Choice
under several option orders) and POST /v1/systemone/separate (each question in its own pass, for the packed-vs-separate
comparison). KEV_PREFIX_CACHE / KEV_PREFIX_MIN_TOKENS / KEV_PREFIX_MAX_TOKENS size the state-prefix cache; KEV_MAX_QUEUE bounds the
waiting requests (default 64, 0 = unbounded, beyond it 503 + Retry-After); KEV_UNREADY_MAX_S is how long /healthz/ready may keep
failing with no completed batch before /healthz fails too (600 s, 0 = off); KEV_MIN_FREE_MIB makes a batch shed the caches and
graph buffers up front when the device is that empty (256, 0 = off); KEV_DATE_FACTS=1 opts into the
date preprocessing (api.with_date_facts). A state over kev.model.SERVE_MAX_STATE tokens is refused with a 422 (kev.model.admit);
KEV_TRUNCATE_STATES=1 reads its first SERVE_MAX_STATE tokens instead, and then every response says whether it did. Backend
and precision follow LoadOptions (KEV_BACKEND, KEV_DTYPE, ...): on Apple Silicon a hybrid checkpoint runs on MLX
by default, without a torch MPS probe, elsewhere on torch in bf16.
"""
import argparse, asyncio, atexit, hmac, os, queue, random, sys, threading, time, traceback, uuid
from concurrent.futures import Future
import torch
from dataclasses import dataclass, field, replace
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from .api import SystemOneRequest, to_record, to_answers, output_tokens, with_date_facts
from .checkpoint import Checkpoint, LoadOptions, fused_available, is_hub_id, mlx_available
from .device import DEVICE_HELP, DEVICES, empty_cache, free_bytes, kind, out_of_memory, select, sync, unrecoverable
from .model import SERVE_MAX_STATE, ContextOverflow, admit

PREFIX_CACHE_SIZE = int(os.environ.get("KEV_PREFIX_CACHE", "4"))          # states kept (KV + DeltaNet states; attention-only backbones also the state's hidden states); 0 disables
PROBE_WAIT_S = float(os.environ.get("KEV_PROBE_WAIT_S", "5"))             # /healthz/ready caps its wait (a wedged model thread cannot hang the probe)
PREFIX_MIN_TOKENS = os.environ.get("KEV_PREFIX_MIN_TOKENS")               # states shorter than this are not cached; default = the model's prefix_min_tokens (0 for hybrid backbones and MLX, 384 for attention-only torch models)
PREFIX_MAX_TOKENS = int(os.environ.get("KEV_PREFIX_MAX_TOKENS", "65536"))  # state tokens the cache holds in all (least recently used evicted first); a longer state is not cached.
                                                                         # One 64k state (Kev-27B: ~1.3 GB of keys, values and DeltaNet states), or four 16k ones, not four 64k ones
DATE_FACTS = os.environ.get("KEV_DATE_FACTS", "0") == "1"
TRUNCATE_STATES = os.environ.get("KEV_TRUNCATE_STATES", "0") == "1"    # unset = a state over SERVE_MAX_STATE tokens gets a 422; 1 = read its first SERVE_MAX_STATE tokens, and every response says whether it did (truncated, usage.state_tokens / state_tokens_used)
API_KEY = os.environ.get("KEV_API_KEY")                                  # unset = open server; set = require Authorization: Bearer <key>, as the TypeSafe clients always send
MAX_BATCH = 64                                                           # requests the model thread takes at once (kev.cuda_graphs splits them to fit its buffers)
MAX_QUEUE = int(os.environ.get("KEV_MAX_QUEUE", "64"))                   # requests allowed to wait at once; 0 = unbounded. Beyond it a request is refused 503 + Retry-After
                                                                         # instead of queueing forever behind a batch that cannot allocate (2026-10-28: 269 requests
                                                                         # queued against a full card, every one of them a 500 minutes later)
UNREADY_MAX_S = float(os.environ.get("KEV_UNREADY_MAX_S", "600"))        # /healthz/ready failing this long with no completed batch fails /healthz too, whatever the
                                                                         # cause: nothing in this process recovers an instance that has served nothing for 10 minutes
                                                                         # (2026-11-04: 117 minutes unready, /healthz green, no restart). 0 disables the watchdog
STALL_S = float(os.environ.get("KEV_STALL_S", "600"))                    # one batch in flight longer than this means the model thread is wedged (a wedged thread
                                                                         # cannot report its own failure); long enough that a legitimate pass never comes near it
MIN_FREE_MIB = int(os.environ.get("KEV_MIN_FREE_MIB", "256"))            # with less device-free VRAM than this a batch sheds caches/buffers before it runs: the
                                                                         # allocation that fails at ~2 MiB free poisons the context (2026-11-04). 0 disables
MODEL_NAMES = ("kev-latest", "jev-latest")                               # both names serve this checkpoint; jev-latest is the TypeSafe SDK default model, so an unconfigured client works


@dataclass
class PrefixCache:
    """State prefixes kept across requests, least recently used first: (state token ids, option_isolation) -> prefix.
    At most `size` states and `max_tokens` state tokens in all; states shorter than min_tokens or longer than max_tokens
    are not cached. A batch keeps (copies) only the new states that will still be here after it, its last distinct ones
    within both bounds: the rest would be evicted by the batch itself."""
    size: int
    min_tokens: int
    max_tokens: int = PREFIX_MAX_TOKENS
    entries: dict = field(default_factory=dict)
    hits: int = 0
    misses: int = 0
    oom_retries: int = 0   # batches that ran out of device memory with states cached, dropped them and ran again (Server._run)

    def plan(self, encs):
        """-> (key per request, None when its state is not cached; its cached prefix or None; whether to keep a new one)."""
        lengths = [enc["seg"].count(0) for enc in encs]
        keys = [(tuple(enc["ids"][:n]), bool(enc.get("option_isolation"))) if self.size and self.min_tokens <= n <= self.max_tokens else None
                for enc, n in zip(encs, lengths)]
        survivors, tokens = set(), 0
        for key in dict.fromkeys(k for k in reversed(keys) if k is not None):   # most recent first, as store() keeps them
            if len(survivors) == self.size or tokens + len(key[0]) > self.max_tokens: break
            survivors.add(key); tokens += len(key[0])
        return keys, [self.entries.get(k) if k is not None else None for k in keys], [k in survivors for k in keys]

    def over(self, keys):
        """Whether these keys (states) exceed either bound: more than `size` states or `max_tokens` state tokens."""
        return len(keys) > self.size or sum(len(k[0]) for k in keys) > self.max_tokens

    def make_room(self, keys, cached, keep):
        """Drop now, before the batch runs, the entries this batch's store() will evict anyway, so an old state does not
        stay resident through the passes of the new one that replaces it. store() (re)inserts every hit and every kept new
        state as most recent and evicts the least recently used while over a bound; evicting the same keys first leaves
        the same cache (tests/test_unit.py). A hit dropped here is still held by `cached` for this batch. Kev-4B on MLX:
        a 64k state's cache is 2.2 GB, so a new 64k state after another one peaked with both (runs/mlx-long-states)."""
        order = dict.fromkeys(self.entries)                     # least recently used first, as store() evicts
        for key, old, k in zip(keys, cached, keep):
            if key is not None and (old is not None or k):
                order.pop(key, None); order[key] = None
        while self.over(order):
            key = next(iter(order)); del order[key]; self.entries.pop(key, None)

    def store(self, keys, cached, prefixes):
        """Record hits and misses, and (re)insert the batch's prefixes in order: most recently used last."""
        for key, old, new in zip(keys, cached, prefixes):
            if key is None: continue
            self.hits += old is not None; self.misses += old is None
            if new is None: continue
            self.entries.pop(key, None); self.entries[key] = new
            while self.over(self.entries): self.entries.pop(next(iter(self.entries)))

    def clear(self):
        self.entries.clear()


def enqueue(q, item):
    """Put `item` in the model thread's queue, refusing with 503 (not blocking) when it is full.

    Backpressure is what keeps an overloaded instance honest: unbounded, a queue turns into a pile of
    requests that all fail together once the batch they wait for cannot allocate, and the platform's
    probes (which see only GETs) never learn (2026-10-28). A refusal frees the gateway to retry or fail
    fast, and /healthz/ready goes through the same queue, so a full queue also marks the instance
    unready."""
    try: q.put_nowait(item)
    except queue.Full:
        raise HTTPException(503, f"queue full: {q.maxsize} requests already waiting; retry shortly",
                            headers={"Retry-After": "1"})


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
    refusals: int = 0        # requests refused because the queue was full (enqueue)
    graph_sheds: int = 0     # batches that shed the CUDA graphs' memory after an OOM (_run)
    pressure_sheds: int = 0  # batches that shed the caches/graph buffers up front because the device was nearly full (_ease_pressure)
    truncate_states: bool = field(default_factory=lambda: TRUNCATE_STATES)   # KEV_TRUNCATE_STATES; off: an over-length state is refused (kev.model.admit)
    # Health bookkeeping the probes read. Single-word assignments of monotonic times from the model thread and the
    # event loop, no lock: Server.lock is held across a whole batch, and a probe that waited on it would time out
    # (and restart a healthy instance) whenever a batch ran long.
    fatal: str = ""                # an unrecoverable accelerator error this process cannot serve through (observe_error)
    unready_since: float = None    # monotonic time /healthz/ready started failing; None while it passes
    busy_since: float = None       # monotonic time the in-flight batch started; None between batches (_work)
    last_batch_at: float = field(default_factory=time.monotonic)   # last batch that completed (_run)

    def __post_init__(self):
        self.release_date = self.release_date or self.checkpoint.release_date()
        self.latency = self.latency or LatencyView()
        self.prefix_cache = PrefixCache(PREFIX_CACHE_SIZE, int(PREFIX_MIN_TOKENS) if PREFIX_MIN_TOKENS else self.model.prefix_min_tokens)
        self.queue, self.stopping = queue.Queue(maxsize=MAX_QUEUE), threading.Event()   # maxsize 0 = unbounded
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

    def observe_error(self, e):
        """Record a failure no retry inside this process can clear (kev.device.unrecoverable: a poisoned CUDA context).
        The first cause wins. Every later request is refused and /healthz fails, so the liveness probe restarts the
        container - the only thing that serves again. A recoverable error (OOM, a shape error) is not recorded."""
        if self.fatal or not unrecoverable(e): return
        self.fatal = repr(e)[:300]
        print(f"kev.serve: unrecoverable accelerator error, this process cannot serve again: {self.fatal}", flush=True)

    def _capture(self, graphs):
        """capture_pending() on the model thread. A capture that escapes its own per-bucket handling (a poisoned
        context, an allocator that cannot map the pool) must not kill the thread: graphs stay off for good and eager
        serving continues; an unrecoverable one also fails /healthz. Their shared buffers are released too - eager
        passes never rebuild them, so an instance that just failed a capture for want of VRAM does not keep holding
        the VRAM it was short of."""
        try:
            with self.lock: graphs.capture_pending(limit=1)
        except Exception as e:
            traceback.clear_frames(e.__traceback__)   # the traceback would hold the capture's tensors
            try: graphs.release(buffers=True, final=True, why=f"a failed capture ({type(e).__name__})")
            except Exception: graphs.disabled = True   # a release that fails too (a dead driver) still leaves graphs off
            self.observe_error(e)
            print(f"kev.serve: capturing a CUDA graph failed, every later pass runs eagerly: {type(e).__name__}: {e}", flush=True)

    def submit(self, rec):
        """Queue one record for the model thread. -> a Future of (probabilities, stats). The state prefix (tokens up to the
        first question) is cached across requests, so a repeated state only pays for its question rows. latency_ms is the
        model time of the batch the request ran in (not its wait in the queue). A record kev.model.admit refuses (a state
        over SERVE_MAX_STATE tokens unless truncate_states, a question row over SERVE_MAX_BRANCH) is a 422 here, before
        it reaches the model."""
        if self.stopping.is_set(): raise HTTPException(503, "the server is stopping")
        if self.fatal:
            raise HTTPException(503, f"unrecoverable accelerator error: {self.fatal}", headers={"Retry-After": "1"})
        try: enc = admit(self.model, self.tok, rec, truncate=self.truncate_states)
        except ValueError as e:   # ContextOverflow, or a record the encoder cannot take
            state = isinstance(e, ContextOverflow) and e.max_state is not None
            raise HTTPException(422, str(e) + (f"; or start the server with KEV_TRUNCATE_STATES=1 to read only its first {e.max_state:,} tokens (responses then say truncated: true)" if state else ""))
        done = Future()
        try: enqueue(self.queue, (enc, done))
        except HTTPException:
            self.refusals += 1
            raise
        return done

    def probs(self, rec):
        return self.submit(rec).result()

    def _work(self):
        graphs = getattr(self.model, "graphs", None)
        while not self.stopping.is_set():
            try: batch = [self.queue.get(timeout=0.05)]
            except queue.Empty:
                if graphs is not None and graphs.capture_due(idle=True):
                    self._capture(graphs)
                continue
            while len(batch) < MAX_BATCH:
                try: batch.append(self.queue.get_nowait())
                except queue.Empty: break
            try:
                with self.lock:   # inside the lock: a batch waiting on the image path's forward is not a wedged batch
                    self.busy_since = time.monotonic()
                    try: results = self._run([enc for enc, _ in batch])
                    finally: self.busy_since = None
            except Exception as e:   # every request of the batch gets the error; the thread lives on
                traceback.clear_frames(e.__traceback__)   # the batch's futures keep the exception, and its frames held the failed pass's tensors (142 MiB, a 3,578-token state on Kev-0.8B); the traceback keeps its lines
                self.observe_error(e)                    # a poisoned context: /healthz fails, the platform restarts
                results = [e] * len(batch)
            for (_, done), result in zip(batch, results):
                (done.set_exception if isinstance(result, Exception) else done.set_result)(result)
                self.queue.task_done()
            if graphs is not None and graphs.capture_due(idle=False):
                self._capture(graphs)

    def _run(self, encs):
        """One batch through model.probs_batch, with the prefix cache. -> per request (probs, stats).

        A pass out of device memory sheds what only costs speed and runs again, in two steps: the cached
        states first (#75, @vtxyer: kept resident, they failed every later batch of that size), then the
        CUDA graphs' private pool and their shared buffers (2026-10-28: a 24 GB card held ~1.5 GiB of
        graph pools and ~3 GiB of shared buffers, so every batch of 269 failed on a 2 MiB output
        allocation and nothing recovered until the container restarted). After the second shed the model
        routes these passes eagerly (kev.cuda_graphs.admits), so serving continues on a card that cannot
        hold the graphs. Only a third OOM reaches the client."""
        graphs = getattr(self.model, "graphs", None)
        self._ease_pressure(graphs)   # before the batch allocates: one that fails asynchronously poisons the context first
        sync(self.device); t = time.time()
        for step in ("cache", "graphs", None):
            keys, cached, keep = self.prefix_cache.plan(encs)
            self.prefix_cache.make_room(keys, cached, keep)
            try: ps, prefixes = self.model.probs_batch(encs, cached, keep); break
            except Exception as e:   # states held: the cache's, and this batch's hits make_room already dropped from it
                if step is None or not out_of_memory(e): raise
                held = self.prefix_cache.entries or any(c is not None for c in cached)
                cached = None        # the retry re-plans; the failed pass's prefixes are not reused
                if step == "cache" and held:
                    self.prefix_cache.clear(); self.prefix_cache.oom_retries += 1
                elif graphs is not None:
                    graphs.release(buffers=True); self.graph_sheds += 1
                else: raise          # nothing left to shed: the card cannot serve this batch
            empty_cache(self.device)   # after the except: its traceback holds the failed pass's tensors
        sync(self.device); dt = round((time.time() - t) * 1000, 1)
        self.prefix_cache.store(keys, cached, prefixes)
        self.batches += 1; self.batched_requests += len(encs)
        self.last_batch_at = time.monotonic()
        # per-request latency observation for kev_latency_ms_*: a request's share of the batch is
        # an honest request-level view (the batch runs as one fused pass; without separate streams
        # there is no per-row timer to read). Same state -> same number, which is exactly what a
        # latency histogram of "how long does serving this kind of request take" wants.
        self.latency.observe(dt, len(encs))
        return [([q.tolist() for q in p], {"tokens": len(enc["ids"]), "state_tokens": enc["state_tokens"], "state_tokens_used": enc["seg"].count(0),
                                           "latency_ms": dt, "prefix_cache_hit": c is not None})   # state_tokens: the request's state; _used: what the model read (less only when truncated)
                for enc, p, c in zip(encs, ps, cached)]

    def _ease_pressure(self, graphs):
        """Make room before a batch if the device is nearly out of free memory, shedding what only costs speed: the
        cached states, then the CUDA graphs and their shared buffers (eager serving continues without them).

        Proactive, because a full card's next allocation can fail asynchronously: on 2026-11-04 a 24 GB instance at
        1.9 MiB free took a 719 launch failure from a host copy and the sticky error ended the instance's
        serving for good. An OOM raised inside the batch still takes _run's ladder; this only removes the common cause.

        The graphs go off for good (release(..., final=True)) rather than merely dropping their buffers: admits()
        rebuilds released buffers for this very batch (_ensure_buffers), which would spend the headroom just made.
        No-op off CUDA and with KEV_MIN_FREE_MIB=0."""
        if MIN_FREE_MIB <= 0 or kind(self.device) != "cuda": return
        try: free = free_bytes(self.device)
        except Exception: return   # the device itself is unhappy: the batch's own path reports it
        if free >= MIN_FREE_MIB * 2**20: return
        before, shed = free, []
        if self.prefix_cache.entries:
            self.prefix_cache.clear(); empty_cache(self.device); shed.append("the cached states")
            try: free = free_bytes(self.device)
            except Exception: free = None
        if (free is None or free < MIN_FREE_MIB * 2**20) and graphs is not None \
                and getattr(graphs, "bank", None) is not None and not getattr(graphs, "disabled", False):
            try: graphs.release(buffers=True, final=True, why="low device memory")
            except Exception: pass   # a device that fails the shed itself: the batch's own path reports it
            else: shed.append("the CUDA graphs")
        if shed:   # nothing to shed is not a shed: no counter bump, no log line on every batch of a full card
            self.pressure_sheds += 1
            print(f"kev.serve: {before // 2**20} MiB device-free < {MIN_FREE_MIB} MiB, shed {' and '.join(shed)} before this batch", flush=True)

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
        """The TypeSafe body. A server that may truncate (truncate_states) also says, on every response, whether it did:
        `truncated`, and usage.state_tokens (the request's state) / state_tokens_used (what the model read), both counting
        the <state> token. The TypeSafe SDKs ignore fields they do not model, so clients keep parsing."""
        answers = to_answers(ps, meta)
        body = {"model": req.model, "answers": answers, "usage": {"input_tokens": m["tokens"], "output_tokens": output_tokens(self.tok, answers)}, "latency_ms": m["latency_ms"]}
        if self.truncate_states:
            body["usage"].update(state_tokens=m["state_tokens"], state_tokens_used=m["state_tokens_used"])
            body["truncated"] = m["state_tokens"] > m["state_tokens_used"]
        return body


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
    if s.fatal:   # the vision tower runs on the same poisoned context: refuse like submit does, before any work
        raise HTTPException(503, f"unrecoverable accelerator error: {s.fatal}", headers={"Retry-After": "1"})
    if s.vision is None:
        raise HTTPException(422, "image requests need KEV_VISION=1 and a base with a vision tower")
    from .vision import decode_image
    try:
        images = [decode_image(r) for r in refs[:4]]
    except ValueError as e:
        raise HTTPException(422, str(e))
    try:   # the same admission as a text request: an over-length state is refused (or truncated on request), never cut silently
        enc = admit(s.model, s.tok, rec, truncate=s.truncate_states)
    except ValueError as e:
        state = isinstance(e, ContextOverflow) and e.max_state is not None
        raise HTTPException(422, str(e) + (f"; or start the server with KEV_TRUNCATE_STATES=1 to read only its first {e.max_state:,} tokens (responses then say truncated: true)" if state else ""))
    with s.lock:
        s._ease_pressure(getattr(s.model, "graphs", None))   # the vision pass allocates on the same nearly-full device
        sync(s.device); t = time.time()
        try:
            ps, m = s.vision.probs_with_images(rec, images, enc=enc)
        except ValueError as e:
            raise HTTPException(422, str(e))
        except Exception as e:
            s.observe_error(e)   # the image path runs under the same device context as the batched one
            raise
        sync(s.device); dt = time.time() - t
    s.latency.observe(dt * 1000)   # one request-level observation per image call (s = server() above)
    return [p.tolist() for p in ps], {"tokens": m["tokens"], "state_tokens": m["state_tokens"], "latency_ms": round(dt * 1000, 1), "prefix_cache_hit": False}


@app.get("/healthz")
def healthz():
    """Liveness: the process is up, the model thread is running, and the instance can still serve.

    Self-healing failures stay at /healthz/ready, where the platform only takes the instance out of
    traffic: a full queue (backpressure), an OOM batch that shed memory and retried, a transient error.
    A failure no retry inside this process can clear fails HERE, so the liveness probe restarts the
    container — the only recovery:
      - an unrecoverable accelerator error (kev.device.unrecoverable: a sticky CUDA failure poisons the
        whole context; every later call fails in-process);
      - one batch in flight longer than KEV_STALL_S (a wedged model thread cannot report its own death);
      - /healthz/ready failing for KEV_UNREADY_MAX_S with no batch completed in that window, whatever
        the cause (a wedged thread, a broken kernel); last_batch_at keeps pure overload out of this
        rule, because a busy instance is still completing batches.

    2026-11-04: a 24 GB instance went from 29 MiB device-free to a sticky launch failure at
    07:31, /healthz/ready answered 503 for the next 117 minutes and /healthz stayed 200 (the thread was
    alive), so the platform neither served nor restarted it."""
    s = app.state.server if hasattr(app, "state") and hasattr(app.state, "server") else None
    if s is None or not getattr(s, "thread", None) or not s.thread.is_alive():
        return JSONResponse({"ok": False, "why": "model thread not running"}, status_code=503)
    now, why = time.monotonic(), ""
    fatal = getattr(s, "fatal", "")
    busy_since, unready_since = getattr(s, "busy_since", None), getattr(s, "unready_since", None)
    if fatal:
        why = f"unrecoverable accelerator error: {fatal}"
    elif STALL_S > 0 and busy_since is not None and now - busy_since > STALL_S:
        why = f"one batch has been in flight for {now - busy_since:.0f}s"
    elif UNREADY_MAX_S > 0 and unready_since is not None and now - max(unready_since, getattr(s, "last_batch_at", unready_since)) > UNREADY_MAX_S:
        why = (f"unready for {now - unready_since:.0f}s with no batch completed for "
               f"{now - max(unready_since, getattr(s, 'last_batch_at', unready_since)):.0f}s")
    if why: return JSONResponse({"ok": False, "why": why}, status_code=503)
    return {"ok": True, "model_thread_alive": True,
            "unready_s": None if unready_since is None else round(now - unready_since, 1)}


@app.get("/healthz/ready")
def healthz_ready():
    """Readiness: one real inference through the same path every request takes (encode -> queue ->
    model thread -> batch). This is the check that "reflects the instance's serving capability": a
    CUDA OOM, a fused-kernel crash or a wedged model thread surfaces here as 503 + the error,
    while /metrics and /v1/models stay green.

    Every 503 here starts the unready clock that /healthz turns into a restart after KEV_UNREADY_MAX_S,
    including a full queue: a wedged model thread fills the queue, so the probe would otherwise only
    ever see "queue full" and the watchdog would never fire. Pure overload does not restart anything —
    /healthz compares the clock against the last completed batch. An answer that is not a well-formed score is a
    503 too, not a 200 with ok: false: the platform routes on the status code, and a probe that keeps answering
    nonsense must not clear the clock.

    The probe record is minimal and fixed (a short state, one two-option score question); it
    waits in the same queue as traffic, so it also reflects queue backlog. It costs one small
    forward pass (~20 ms on a 4B) — probing every 10-30 s keeps that well under 1% of capacity."""
    if not hasattr(app, "state") or not hasattr(app.state, "server"):
        return JSONResponse({"ok": False, "why": "no server"}, status_code=503)
    req = SystemOneRequest.model_validate({"state": "Health probe: readiness check.",
                                           "questions": {"probe": {"type": "score", "instructions": "healthy?",
                                                                   "criteria": ["no", "yes"]}}})
    t0 = time.perf_counter()
    s = app.state.server
    ready_since = getattr(s, "unready_since", None)
    if getattr(s, "fatal", ""):   # the inference path is gone until a restart: fail now, queue no probe pass behind it
        if ready_since is None: s.unready_since = time.monotonic()
        q = getattr(getattr(s, "queue", None), "qsize", lambda: -1)()
        return JSONResponse({"ok": False, "why": f"unrecoverable accelerator error: {s.fatal}", "queued": q,
                             "errors": METRICS["errors"]}, status_code=503)
    try:
        # one real inference through the same dispatch every request takes; bounded wait so a dead
        # model thread (future never completing) turns into a 503 probe failure, not a hung probe
        rec, meta = to_record(prepare(req))
        ps, m = s.submit(rec).result(timeout=PROBE_WAIT_S)
        body = s._body(req, meta, ps, m)
        answers = body["answers"]["probe"]
        ok = answers["type"] == "score" and answers["legend"] == {"0": "no", "1": "yes"} \
             and sum(answers["probabilities"].values()) > 0.99
        if not ok:   # a probe pass that answers nonsense is not ready either: 200 here would clear the liveness clock
            if getattr(s, "unready_since", None) is None: s.unready_since = time.monotonic()
            return JSONResponse({"ok": False, "why": "the probe's answer was not a well-formed score", "queued": s.queue.qsize(),
                                 "errors": METRICS["errors"]}, status_code=503)
        s.unready_since = None
        return {"ok": ok, "latency_ms": body["latency_ms"], "probe_ms": round((time.perf_counter() - t0) * 1000, 1),
                "queued": s.queue.qsize(), "capacity": getattr(s.queue, "maxsize", 0),
                "batches": s.batches, "errors": METRICS["errors"], "refused": getattr(s, "refusals", 0)}
    except Exception as e:
        METRICS["errors"] += 1
        s = getattr(app.state, "server", None)
        if s is not None and getattr(s, "unready_since", None) is None: s.unready_since = time.monotonic()
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
        lines.append(f"kev_queue_capacity {getattr(s.queue, 'maxsize', 0)}")
        lines.append(f"kev_queue_refusals_total {getattr(s, 'refusals', 0)}")
        lines.append(f"kev_graph_sheds_total {getattr(s, 'graph_sheds', 0)}")
        lines.append(f"kev_pressure_sheds_total {getattr(s, 'pressure_sheds', 0)}")
        # health as gauges: 2026-11-04 had an instance serving nothing for 117 minutes while every
        # scraped metric said "no traffic". kev_unready_seconds is 0 exactly while readiness passes.
        lines.append(f"kev_fatal {1 if getattr(s, 'fatal', '') else 0}")
        since = getattr(s, "unready_since", None)
        lines.append(f"kev_unready_seconds {0 if since is None else round(time.monotonic() - since, 1)}")
        # the CUDA graphs' own memory is the biggest non-model allocation and was invisible to a
        # scraper before (only /v1/models JSON carried it) — the 2026-11-04 card held ~3 GiB of it
        graphs = getattr(getattr(s, "model", None), "graphs", None)
        st = None
        if graphs is not None and hasattr(graphs, "stats"):
            # stats() reads the driver (mem_get_info) and the buffers being released: a scrape during the incident
            # these gauges exist for must not 500 and lose kev_fatal/kev_unready_seconds with it
            try: st = graphs.stats()
            except Exception: st = None
        if st is not None:
            lines += [f"kev_graph_buffers_bytes {st['buffers_mib'] * 2**20}",
                      f"kev_graph_captured {st['captured']}", f"kev_graph_pending {st['pending']}",
                      f"kev_graph_failed {st['failed']}", f"kev_graph_gated_total {st['gated']}",
                      f"kev_graph_disabled {int(bool(st['disabled']))}"]
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
    return truncation_marks({"runs": runs, "argmax_stable": len({x["choice"] for x in runs}) == 1, "spread": spread}, resp)


@app.post("/v1/systemone/separate")
def systemone_separate(req: SystemOneRequest):
    """Answer each question in its own request against the same state (N passes). For packed-vs-separate comparison."""
    parts = [server().answer(req.model_copy(update={"questions": {qid: q}})) for qid, q in req.questions.items()]
    answers = {qid: a for p in parts for qid, a in p["answers"].items()}
    body = {"model": req.model, "answers": answers,
            "usage": {"input_tokens": sum(p["usage"]["input_tokens"] for p in parts), "output_tokens": output_tokens(server().tok, answers)},
            "latency_ms": round(sum(p["latency_ms"] for p in parts), 1)}
    return truncation_marks(body, parts[0])


def truncation_marks(body, part):
    """A demo endpoint built from several /v1/systemone answers on one state carries a truncating server's marks
    (Server._body: `truncated`, usage.state_tokens / state_tokens_used) from any one of them: each read the same tokens."""
    if "truncated" in part:
        body["truncated"] = part["truncated"]
        body.setdefault("usage", {}).update({k: part["usage"][k] for k in ("state_tokens", "state_tokens_used")})
    return body


@app.get("/v1/models")
def models():
    """One TypeSafe model card (name, description, release_date) per accepted model name, plus the Kev serving details
    a client may ignore: the run, the base, the device, the backend and precision, the temperature, prefix-cache stats."""
    s = server()
    ck, meta = s.checkpoint, s.checkpoint.meta
    card = {"description": f"Kev pointer head on {meta.base}, serving {ck.requested} at temperature {s.model.head.temperature:.2f}",
            "release_date": s.release_date,
            "run": ck.requested, "base": meta.base, "lora": meta.lora, "device": s.device, "backend": s.model.backend, "dtype": s.model.dtype,
            "temperature": s.model.head.temperature, "max_state_tokens": SERVE_MAX_STATE, "truncate_states": s.truncate_states,
            "cuda_graphs": graphs.stats() if (graphs := getattr(s.model, "graphs", None)) else None,
            "prefix_cache": {"size": s.prefix_cache.size, "min_state_tokens": s.prefix_cache.min_tokens, "max_tokens": s.prefix_cache.max_tokens, "hits": s.prefix_cache.hits,
                             "misses": s.prefix_cache.misses, "cached_states": len(s.prefix_cache.entries), "oom_retries": s.prefix_cache.oom_retries},
            "batches": {"count": s.batches, "requests": s.batched_requests, "queued": s.queue.qsize(),
                        "capacity": s.queue.maxsize, "refused": getattr(s, "refusals", 0)}}
    return {"models": [{"name": name, **card} for name in MODEL_NAMES]}


def _serving_device(requested, opts, run):
    """Device and checkpoint for this process. A hybrid checkpoint on Apple Silicon with the MLX backend does not consult
    the torch MPS probe. Everywhere else the probe runs before the checkpoint is opened, so a driver crash stays in the child."""
    probe_dtype = opts.dtype or torch.bfloat16
    mlx_candidate = (sys.platform == "darwin" and requested in ("auto", "mps") and opts.backend in ("auto", "mlx")
                     and opts.dtype is not torch.float32)
    if mlx_candidate:
        ck = Checkpoint(run)
        if mlx_available() and ck.hybrid_base():
            return "mps", ck
        return select(requested, dtype=probe_dtype), ck
    return select(requested, dtype=probe_dtype), Checkpoint(run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/kev")
    ap.add_argument("--fallback", default="runs/smoke")
    ap.add_argument("--host", default="127.0.0.1", help="interface to bind; 0.0.0.0 to serve beyond this machine (a container, a VM behind a proxy)")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--device", choices=("auto", *DEVICES), default="auto", help=DEVICE_HELP)
    a = ap.parse_args()
    run = a.run if is_hub_id(a.run) or os.path.exists(f"{a.run}/head.pt") else a.fallback
    if run != a.run: print(f"{a.run} not found, falling back to {run}")
    opts = LoadOptions.from_env()
    if opts.backend is None: opts = replace(opts, backend="auto")   # serving default: MLX for the hybrid Qwen3.5 checkpoints on Apple Silicon (LoadOptions.backend); KEV_BACKEND=torch to decline
    dev, ck = _serving_device(a.device, opts, run)
    if dev == "mps" and opts.attn is None: opts = replace(opts, attn="sdpa")   # serving default on Apple GPUs (parity measured)
    if dev != "cpu" and opts.dtype is None: opts = replace(opts, dtype=torch.bfloat16)   # serving default: 2-4.5x faster than fp32 on an L4, same answers (LoadOptions.dtype); KEV_DTYPE=fp32 for the exact path
    cuda_optimizations = dev == "cuda" and torch.version.hip is None   # these serving paths are measured on NVIDIA
    if cuda_optimizations and opts.cuda_graphs is None: opts = replace(opts, cuda_graphs=True)   # serving default: a pass is ~2,000 kernel launches, so replaying graphs cuts warm latency several-fold (kev.cuda_graphs); KEV_CUDA_GRAPHS=0 to decline
    fused_default = cuda_optimizations and opts.fused is None
    if fused_default: opts = replace(opts, fused=fused_available())   # serving default: fused Qwen3.5 kernels, ~1/3 less GPU time per batch (kev.fused_qwen35), when fla is installed; KEV_FUSED=0 to decline, KEV_FUSED=1 to insist
    tok, model = ck.load(dev, opts)
    if fused_default and not opts.fused and model.hybrid: print("fused Qwen3.5 kernels off: install the flash-linear-attention version kev/fused_qwen35.py pins (FLA_VERSION) to turn them on")
    # Image channel (opt-in, KEV_VISION=1): attach the base's untrained vision tower to the
    # loaded model. base/base_revision come from the checkpoint's head.pt meta; on the
    # ModelScope path (KEV_BASE_HUB=modelscope, a deploy-pack switch resolved inside
    # Checkpoint) the revision is a local dir and does not apply.
    if os.environ.get("KEV_VISION") == "1":
        from .vision import attach
        vision = attach(model, tok, ck.meta.base,
                        None if (isinstance(ck.meta.base, str) and os.path.isdir(ck.meta.base)) else ck.meta.base_revision)
    else:
        vision = None
    app.state.server = Server(ck, tok, model, dev, vision=vision)
    print(f"serving {ck.requested} ({ck.path}) on {dev} via {model.backend} ({model.dtype}) {a.host}:{a.port}; states over {SERVE_MAX_STATE:,} tokens "
          f"{'truncated (KEV_TRUNCATE_STATES=1)' if app.state.server.truncate_states else 'refused (422)'}")   # /v1/models reports the run as given, not the resolved cache path
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
