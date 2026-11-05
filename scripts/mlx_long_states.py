"""Long states served on Apple Silicon (the MLX backend): latency, memory and correctness at 8k-64k-token states.

    uv run --extra serve python scripts/mlx_long_states.py --run jaredpalmer/kev-0.8b --torch-reference 8192,16384 --out runs/mlx-long-states/0.8b.json
    uv run --extra serve python scripts/mlx_long_states.py --run jaredpalmer/kev-4b --out runs/mlx-long-states/4b.json
    uv run python scripts/mlx_long_states.py --summarize runs/mlx-long-states        # the table (summary.md)
    uv run --extra serve python scripts/mlx_long_states.py --run jaredpalmer/kev-4b --prefill-ab --lengths 8192,16384 --out runs/mlx-long-states/prefill-ab-4b.json
    uv run --extra serve python scripts/mlx_long_states.py --run jaredpalmer/kev-0.8b --prefill-ab --precision fp32-cpu --lengths 1500,3000 --hard 1 \
        --out runs/mlx-long-states/prefill-ab-0.8b-fp32-cpu.json     # chunking is exact: differences are float reassociation

The request is the one runs/kev-deploy-2ea5660/make_long_state.py builds (a one-line billing ticket, then this commit's
docs/model-cards/*.md repeated in order as filler) with one more sentence planted at 60% depth: an internal note naming
the office every letter about the order must go to. Three questions: the team (billing, from the ticket), whether the
customer reports a duplicate charge (yes, from the ticket) and the office (only the planted note says). Each state has
exactly `n` tokens as kev.model.encode counts them (`<state>` included, what admission compares with SERVE_MAX_STATE).

Per length, through kev.serve.Server (the model thread /v1/systemone uses, prefix cache on), three new states, in order:
  start     the Lisbon note right after the ticket (the depth control: same text, shallow): the length's first request, `first_ms`
  swapped   the note at 60% depth naming Osaka: the office answer must move with it (the note is read, not guessed)
  planted   the note at 60% depth naming Lisbon
(`cold_ms` = their median; planted goes last because the cache holds 65,536 state tokens in all, so at 32k and 64k the
earlier states are evicted), then the planted request `--reps` more times (a cached state: only the question rows run, `cached_ms`), the last of them
over HTTP (FastAPI TestClient on kev.serve.app, the body's latency_ms). Memory: MLX's allocator peak for the length
(`mx.get_peak_memory`, reset per length; the prefix cache is cleared first so one length's cached states do not count in
the next), the process's peak physical footprint and RSS (sampled every 20 ms; RSS misses the Metal buffers), and the system's swap-outs during the length (psutil `sout`, 0 =
nothing was pushed to swap). --torch-reference lengths also run the planted record on the fp32 torch path on the CPU
(SDPA: the state's causal pass does not materialise L x L; loaded first and released before MLX loads) and report max
|dp| and argmax agreement per question. Finally a 70,000-token state is POSTed to /v1/systemone: admission must refuse it
with a 422.

Memory guard: a size is skipped when its bf16 backbone (from the Hub's safetensors metadata) plus 1.5 GB of working set and a 1.5 GB margin exceed
the memory the system reports available (the fp32 torch reference: twice that). A length is skipped when its projected
working set would not fit either: the keys and values per state token (measured on a 2,048-token warm-up request) times
the tokens a new state's pass holds (its cache, the branch pass's copy of it, and the older cached states make_room keeps:
at most KEV_PREFIX_MAX_TOKENS - n, and at most n here) plus the transient the lengths so far measured beyond that (at
least 1.5 GB); and every later length once one pushed more than 0.5 GB to swap. Before each
length the script waits (up to 30 min) while another GPU job of this repo runs (`others`: found by command line, since an
MLX job's Metal memory is not in its RSS). Skips and waits are written to the report.
"""
import argparse, gc, glob, json, platform, statistics, subprocess, sys, threading, time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.api import SystemOneRequest, to_record  # noqa: E402
from kev.checkpoint import Checkpoint, LoadOptions  # noqa: E402
from kev.model import SERVE_MAX_STATE, load_tokenizer, user_tokens  # noqa: E402
from kev.suite import write_json  # noqa: E402

LENGTHS = (8192, 16384, 32768, 65000)
TICKET = "Customer message: I was charged twice for order 8812. Please refund the duplicate charge.\n\n"   # make_long_state.py's
NOTE = "\n\nInternal note from the billing team: every letter about order 8812 must go to our {city} office, and to no other office.\n\n"
DEPTH = 0.6
OFFICES = {"lisbon": "The Lisbon office", "denver": "The Denver office", "osaka": "The Osaka office", "nairobi": "The Nairobi office"}
QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
             "criteria": {"returns": "Exchanges, refunds for returned items", "shipping": "Delivery, delays", "billing": "Charges, payments, duplicate charges"}},
    "duplicate": {"type": "noul", "instructions": "Does the customer report being charged twice?"},
    "office": {"type": "choice", "instructions": "According to the internal note from the billing team, which office must every letter about order 8812 go to?",
               "criteria": OFFICES},
}
EXPECTED = {"team": "billing", "duplicate": "true"}   # plus office = the planted city
VARIANTS = {"start": ("Lisbon", 0.0), "swapped": ("Osaka", DEPTH), "planted": ("Lisbon", DEPTH)}   # in request order: planted last, so it is the state the cache still holds
ADMISSION_TOKENS = 70000
MARGIN = 1.5e9
TRANSIENT = 1.5e9   # working set beyond the cached states, before a length has measured it (Kev-4B, 8,192 tokens: 1.2 GB)
QUIET_WAIT = 1800   # seconds to wait for another heavy Python job on this machine to finish before a length (AGENTS.md: one at a time)


def filler_ids(tok, n):
    """make_long_state.py's filler: the model cards of this commit repeated in order until there are n tokens."""
    docs = [open(p, encoding="utf-8").read() for p in sorted(glob.glob(str(Path(__file__).resolve().parents[1] / "docs/model-cards/*.md")))]
    ids, i = [], 0
    while len(ids) < n:
        ids += user_tokens(tok, docs[i % len(docs)] + "\n\n"); i += 1
    return ids


def build_state(tok, filler, n, city=None, depth=DEPTH):
    """State text of exactly n tokens as encode counts them (<state> + user_tokens): ticket, filler, the note (if a city)
    starting at depth * n tokens, filler. The filler is cut on token boundaries and decoded; the tail is trimmed until the
    re-tokenized total is exact (a cut can merge with its neighbour)."""
    note = NOTE.format(city=city) if city else ""
    head = len(user_tokens(tok, TICKET))
    a = max(0, int(depth * n) - head) if city else 0
    b = n - 1 - head - a - len(user_tokens(tok, note))
    for _ in range(20):
        text = TICKET + tok.decode(filler[:a]) + note + tok.decode(filler[a:a + b])
        got = len(user_tokens(tok, text)) + 1
        if got == n: return text
        b -= got - n
    raise RuntimeError(f"could not build a {n}-token state (last {got})")


def request(state):
    return SystemOneRequest.model_validate({"state": state, "model": "kev-latest", "questions": QUESTIONS})


def named(probs, meta):
    """[[p, ...] per question] -> {question: {option key: p}} (raw probabilities, not the body's 4-decimal ones)."""
    return {m["id"]: dict(zip(m["keys"], map(float, p))) for p, m in zip(probs, meta)}


def correct(answers, city):
    top = {q: max(p, key=p.get) for q, p in answers.items()}
    want = {**EXPECTED, "office": city.lower()}
    return {q: top[q] == want[q] for q in want}


def agreement(a, b):
    """{question: {key: p}} pairs -> max |dp| over every option and the questions whose argmax differs."""
    dp = max(abs(a[q][k] - b[q][k]) for q in a for k in a[q])
    return {"max_dp": round(dp, 5), "argmax_flips": [q for q in a if max(a[q], key=a[q].get) != max(b[q], key=b[q].get)]}


def footprint():
    """This process's physical footprint (proc_pid_rusage's ri_phys_footprint, what Activity Monitor calls Memory): unlike
    RSS it counts the Metal buffers MLX allocates."""
    import ctypes, os
    buf = (ctypes.c_uint64 * 32)()   # rusage_info_v0: uuid (2 words), user, system, idle wakeups, interrupt wakeups, pageins, wired, resident, phys_footprint, ...
    if ctypes.CDLL("/usr/lib/libproc.dylib").proc_pid_rusage(os.getpid(), 0, ctypes.byref(buf)) != 0: return 0
    return buf[9]


class RSSPeak:
    """Peak resident set size and physical footprint of this process while the block runs (sampled every 20 ms)."""
    def __enter__(self):
        self.proc, self.peak, self.footprint, self.done = psutil.Process(), 0, 0, threading.Event()
        def sample():
            while not self.done.is_set():
                self.peak = max(self.peak, self.proc.memory_info().rss); self.footprint = max(self.footprint, footprint()); self.done.wait(0.02)
        self.thread = threading.Thread(target=sample, daemon=True); self.thread.start(); return self

    def __exit__(self, *exc):
        self.done.set(); self.thread.join()


def gb(x): return round(x / 1e9, 2)


HEAVY = ("scripts/", "kev.train", "kev.benchmark", "kev.evaluate", "kev.experiment", "test_mlx", "test_model")


def others():
    """Other Python jobs on this machine that compete for the GPU: benchmark / parity / training scripts and the weight-
    backed test suites by command line (an MLX job's Metal memory is not in its RSS, so size does not find them), and a
    server only while it is working (an idle one holds memory, which the available-memory check already sees)."""
    me, found = psutil.Process().pid, []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or [])
            if p.info["pid"] == me or "python" not in (p.info["name"] or "").lower(): continue
            if any(k in cmd for k in HEAVY) or ("kev.serve" in cmd and p.cpu_percent(interval=1.0) > 20):
                found.append(f"{p.info['pid']}: {cmd[:160]}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return found


def wait_quiet():
    """Block (up to QUIET_WAIT) while another heavy job runs; -> (seconds waited, what was running)."""
    t, seen = time.time(), []
    while (busy := others()) and time.time() - t < QUIET_WAIT:
        seen = busy; time.sleep(15)
    return round(time.time() - t), seen


def backbone_bytes(ck):
    """Bytes of the checkpoint's base weights as stored (Hub safetensors metadata; no download)."""
    from huggingface_hub import HfApi
    meta = HfApi().get_safetensors_metadata(ck.meta.base, revision=ck.meta.base_revision)
    size = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1}
    return sum(n * size.get(dt, 2) for dt, n in meta.parameter_count.items())


def machine():
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    from importlib.metadata import version
    return {"chip": chip, "memory_gb": gb(psutil.virtual_memory().total), "macos": platform.mac_ver()[0],
            "mlx": version("mlx"), "mlx_lm": version("mlx-lm"), "torch": version("torch")}


def torch_reference(ck, tok, records, lengths):
    """fp32 torch on the CPU (the exact path, SDPA) for the planted record at each length; the model is released on return."""
    import torch
    _, ref = ck.load("cpu", LoadOptions(backend="torch", attn="sdpa"))
    from kev.model import admit
    out = {}
    for n in lengths:
        rec, meta = records[n]
        t = time.perf_counter()
        with torch.no_grad(): ps = ref.probs(admit(ref, tok, rec))
        out[n] = {"answers": named([p.tolist() for p in ps], meta), "seconds": round(time.perf_counter() - t, 1)}
        print("torch fp32 cpu", n, out[n], flush=True)
    del ref; gc.collect()
    return out


PRECISIONS = ("bf16", "fp32-gpu", "fp32-cpu")


def cache_error(cache, ref, chunk):
    """A chunked prefix's cache [(attention?, array)] against one pass's: the largest relative error of a whole array
    (norm) and of one element (against that array's largest value); for the attention keys and values per state position,
    the median, the largest, and the largest at a pass's first token (where a boundary bug would show), all relative to that
    position's norm."""
    import numpy as np
    whole = max(float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)) for (_, a), (_, b) in zip(cache, ref))
    element = max(float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30)) for (_, a), (_, b) in zip(cache, ref))
    norm = lambda x: np.sqrt((x.astype(np.float64) ** 2).sum(axis=(0, 1, 3)))   # [1, heads, positions, dim] -> per position  # noqa: E731
    per = np.max([norm(a - b) / np.maximum(norm(b), 1e-30) for (att, a), (_, b) in zip(cache, ref) if att], axis=0)
    starts = per[chunk::chunk] if chunk < len(per) else per[:0]
    return {"array_rel": whole, "element_rel": element, "kv_position_rel": {"median": float(np.median(per)), "max": float(per.max()), "argmax": int(per.argmax()),
            "max_at_pass_starts": float(starts.max()) if len(starts) else None}, "identical": all(np.array_equal(a, b) for (_, a), (_, b) in zip(cache, ref))}


def prefill_ab(ck, lengths, chunks, precision, hard, out, layers=None):
    """--prefill-ab: the state pass alone (MLXDecisionModel.prefix) per PREFILL_CHUNK against one pass, on the planted state
    at each of --lengths and on the `hard` longest hard-v1 long_policy development records (real states, whose lengths are
    no multiple of a chunk): seconds, MLX peak above the weights, the answers' max |dp| and argmax flips, and the largest
    difference of the prompt cache (attention keys/values, DeltaNet conv and recurrent states) relative to its largest
    value (cache_error). --precision fp32-gpu casts the backbone to fp32, fp32-cpu also runs it on the CPU: an exact chunked
    prefill differs from one pass only by float reassociation, so its differences shrink with the precision (to ~1e-6 on
    the CPU; Metal's fp32 matmul is a reduced-precision fast path) where a boundary bug (conv window, rotary offset,
    recurrent state) would not. --layers N keeps the backbone's first N layers (the answers are then meaningless, the
    cache comparison is not): the 4B's own weights in fp32 fit a 32 GB Mac that way."""
    import mlx.core as mx
    import numpy as np
    import kev.mlx_model as MM
    from kev.data import api_request
    from kev.model import admit
    from kev.suite import load_split
    tok, m = ck.load("mps", LoadOptions(backend="mlx")); served = MM.PREFILL_CHUNK
    if layers: m.text.layers = m.text.layers[:layers]
    if precision != "bf16": m.lm.set_dtype(mx.float32); mx.eval(m.lm.parameters())
    if precision == "fp32-cpu": mx.set_default_device(mx.cpu)
    mx.clear_cache(); weights = mx.get_active_memory()
    filler = filler_ids(tok, max(lengths + [2048]) + 1000)
    cases = {f"planted-{n}": to_record(request(build_state(tok, filler, n, "Lisbon"))) for n in lengths}
    long = sorted((r for r in load_split("evals/hard-v1", "development") if r["_meta"]["family"] == "long_policy"), key=lambda r: -r["_meta"]["state_tokens"])
    cases.update({r["_meta"]["id"]: to_record(SystemOneRequest.model_validate(api_request(r))) for r in long[:hard]})
    report = {"run": ck.requested, "path": str(ck.path), "machine": machine(), "precision": precision, "dtype": m.dtype, "device": str(mx.default_device()),
              "layers": len(m.text.layers), "weights_gb": gb(weights), "cases": {}}
    m.probs(admit(m, tok, to_record(request(build_state(tok, filler, 2048)))[0]))   # warm-up
    flat = lambda cache: [(hasattr(c, "offset"), np.asarray(x.astype(mx.float32))) for c in cache for x in c.state]   # noqa: E731
    for name, (rec, meta) in cases.items():
        enc = admit(m, tok, rec); n = enc["seg"].count(0); rows = {}
        for chunk in (10 ** 9, *[c for c in chunks if c < n]):
            MM.PREFILL_CHUNK = chunk; gc.collect(); mx.clear_cache(); mx.reset_peak_memory(); t = time.perf_counter()
            prefix = m.prefix(enc); seconds = time.perf_counter() - t; peak = mx.get_peak_memory() - weights
            answers, cache = named([p.tolist() for p in m.probs_with_prefix(enc, prefix)], meta), flat(prefix[1]); del prefix
            if chunk == 10 ** 9: one, one_cache = answers, cache
            rows["one pass" if chunk == 10 ** 9 else str(chunk)] = {"seconds": round(seconds, 2), "peak_above_weights_gb": gb(peak),
                                                                    "vs_one_pass": agreement(answers, one), "cache": cache_error(cache, one_cache, chunk)}
        report["cases"][name] = {"state_tokens": n, "chunks": rows}; print(name, n, rows, flush=True)
        write_json(Path(out), report)
    MM.PREFILL_CHUNK = served


def summarize(folder):
    """runs/mlx-long-states/*.json -> summary.md next to them: one row per size x length."""
    rows = ["| size | state tokens | cold (first / median of 3) | cached (median) | MLX peak (weights) | footprint peak | swap out | planted fact (p) | swapped / start | vs fp32 torch |",
            "|---|---:|---:|---:|---:|---:|---:|---|---|---|"]
    notes = []
    for path in sorted(Path(folder).glob("*.json")):
        if path.name == "summary.json" or path.name.startswith("prefill-ab"): continue
        r = json.loads(path.read_text(encoding="utf-8")); size = r["run"].split("kev-")[-1].upper(); m = r["machine"]
        if "skipped" in r:
            notes.append(f"- Kev-{size}: {r['skipped']}"); continue
        if isinstance(r.get("torch_fp32_cpu"), str): notes.append(f"- Kev-{size} fp32 torch reference {r['torch_fp32_cpu']}")
        elif any("torch_fp32_cpu" in x for x in r["lengths"].values()):
            notes.append(f"- Kev-{size}: the fp32 torch reference ran first in the same process, so its footprint and RSS include host memory torch freed but kept; the MLX peak does not")
        for n, x in r["lengths"].items():
            if "skipped" in x or "error" in x:
                rows.append(f"| Kev-{size} | {int(n):,} | {x.get('skipped') or x.get('error')} | | | | | | | |"); continue
            p = x["office_p_true"]; ok = lambda v: "ok" if all(x["variants"][v]["correct"].values()) else "WRONG"   # noqa: E731
            ref = x.get("torch_fp32_cpu", {}).get("vs_mlx")
            rows.append(f"| Kev-{size} | {int(n):,} | {x['first_ms'] / 1e3:.1f} / {x['cold_ms'] / 1e3:.1f} s | {x['cached_ms']:.0f} ms | {x['mlx_peak_gb']:.1f} GB ({r['resident_gb']:.1f}) | "
                        f"{x.get('footprint_peak_gb', 0):.1f} GB | {x['swap_out_gb']:.2f} GB | {ok('planted')} ({p['planted']:.2f}) | {ok('swapped')} ({p['swapped']:.2f}) / {ok('start')} ({p['start']:.2f}) | "
                        + (f"max \\|dp\\| {ref['max_dp']:.4f}, {len(ref['argmax_flips'])} flips" if ref else "-") + " |")
        notes.append(f"- Kev-{size}: {r['path'].rsplit('/', 1)[-1][:7]}, {r['dtype']}, load {r['load_seconds']} s, admission of a {r['admission']['state_tokens']:,}-token state: {r['admission']['status']}")
    for path in sorted(Path(folder).glob("prefill-ab-*.json")):
        r = json.loads(path.read_text(encoding="utf-8")); size = r["run"].split("kev-")[-1].split("@")[0].upper()
        for name, x in r["cases"].items():
            one, rest = x["chunks"]["one pass"], {c: v for c, v in x["chunks"].items() if c != "one pass"}
            notes.append(f"- Kev-{size} {r['precision']}, {r['layers']} layers, the state pass alone, {name} ({x['state_tokens']:,} tokens, {path.name}): one pass {one['seconds']} s / {one['peak_above_weights_gb']} GB above the weights; "
                         + ", ".join(f"{c}-token chunks {v['seconds']} s / {v['peak_above_weights_gb']} GB (max |dp| {v['vs_one_pass']['max_dp']:.2g}, cache {v['cache']['array_rel']:.1g}, "
                                     f"keys/values per position: median {v['cache']['kv_position_rel']['median']:.1g}, max {v['cache']['kv_position_rel']['max']:.1g}, max at pass starts {v['cache']['kv_position_rel']['max_at_pass_starts'] or 0:.1g})" for c, v in rest.items()))
    text = f"MLX long-state serving, {m['chip']} {m['memory_gb']:.0f} GB, macOS {m['macos']}, mlx {m['mlx']} / mlx-lm {m['mlx_lm']} (scripts/mlx_long_states.py)\n\n" + "\n".join(rows) + "\n\n" + "\n".join(notes) + "\n"
    (Path(folder) / "summary.md").write_text(text, encoding="utf-8"); print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="jaredpalmer/kev-4b")
    ap.add_argument("--lengths", default=",".join(map(str, LENGTHS)))
    ap.add_argument("--reps", type=int, default=3, help="cached repeats of the planted request")
    ap.add_argument("--torch-reference", default="", help="comma-separated lengths to also run on fp32 torch (CPU)")
    ap.add_argument("--out", help="the report to write (runs/mlx-long-states/<size>.json)")
    ap.add_argument("--summarize", metavar="DIR", help="only write DIR/summary.md from the reports in DIR")
    ap.add_argument("--prefill-ab", action="store_true", help="only the state pass per PREFILL_CHUNK at --lengths and on --hard real records, against one pass (prefill_ab)")
    ap.add_argument("--chunks", default="256,512,1024,2048,4096", help="--prefill-ab: the PREFILL_CHUNK values to compare with one pass")
    ap.add_argument("--precision", default="bf16", choices=PRECISIONS, help="--prefill-ab: bf16 as served, or the backbone in fp32 on the GPU / on the CPU")
    ap.add_argument("--layers", type=int, help="--prefill-ab: keep only the backbone's first N layers (the 4B in fp32 on the CPU)")
    ap.add_argument("--hard", type=int, default=3, help="--prefill-ab: the longest hard-v1 long_policy development records to add (~4.9k tokens each)")
    a = ap.parse_args()
    if a.summarize: return summarize(a.summarize)
    if not a.out: ap.error("--out is required")
    if a.prefill_ab: return prefill_ab(Checkpoint(a.run), [int(x) for x in a.lengths.split(",") if x], [int(x) for x in a.chunks.split(",")], a.precision, a.hard, a.out, a.layers)
    lengths = [int(x) for x in a.lengths.split(",")]
    ck = Checkpoint(a.run)
    report = {"run": a.run, "path": str(ck.path), "base": ck.meta.base, "base_revision": ck.meta.base_revision, "machine": machine(),
              "request": {"ticket": TICKET, "note": NOTE, "depth": DEPTH, "questions": QUESTIONS, "variants": VARIANTS, "filler": "docs/model-cards/*.md, sorted, repeated"},
              "lengths": {}}
    save = lambda: write_json(Path(a.out), report)   # noqa: E731
    report["waited_before_load_s"], report["busy_before_load"] = wait_quiet()
    need = backbone_bytes(ck); avail = psutil.virtual_memory().available
    report["memory_before"] = {"backbone_gb": gb(need), "available_gb": gb(avail), "swap_used_gb": gb(psutil.swap_memory().used)}
    if need + TRANSIENT + MARGIN > avail:
        report["skipped"] = f"not run: the {gb(need)} GB backbone + {gb(TRANSIENT)} GB working set + {gb(MARGIN)} GB margin exceeds the {gb(avail)} GB available; it would swap"
        print(report["skipped"]); save(); return

    tok = load_tokenizer(ck.meta.base, revision=ck.meta.base_revision)
    filler = filler_ids(tok, max(lengths + [ADMISSION_TOKENS]) + 1000)
    states = {n: {v: build_state(tok, filler, n, city, depth) for v, (city, depth) in VARIANTS.items()} for n in lengths}
    records = {n: {v: to_record(request(s)) for v, s in by.items()} for n, by in states.items()}
    ref_lengths = [int(x) for x in a.torch_reference.split(",") if x]
    reference = {}
    if ref_lengths and 2 * need + MARGIN > psutil.virtual_memory().available:   # fp32 = twice the bf16 backbone
        report["torch_fp32_cpu"] = f"skipped: the fp32 backbone ({gb(2 * need)} GB) + {gb(MARGIN)} GB margin exceeds the {gb(psutil.virtual_memory().available)} GB available"
        print(report["torch_fp32_cpu"], flush=True)
    elif ref_lengths:
        reference = torch_reference(ck, tok, {n: records[n]["planted"] for n in ref_lengths}, ref_lengths)

    import mlx.core as mx
    from fastapi.testclient import TestClient
    import kev.serve as serve
    t, sout = time.perf_counter(), psutil.swap_memory().sout
    tok, m = ck.load("mps", LoadOptions(backend="mlx"))
    assert m.backend == "mlx", m.backend
    report["load_seconds"] = round(time.perf_counter() - t, 1); report["load_swap_out_gb"] = gb(psutil.swap_memory().sout - sout)
    mx.clear_cache(); weights = mx.get_active_memory()
    report["resident_gb"] = gb(weights); report["footprint_after_load_gb"] = gb(footprint()); report["dtype"] = m.dtype; report["temperature"] = float(m.head.temperature)
    server = serve.Server(ck, tok, m, "mps", release_date="-")
    serve.app.state.server = server
    client = TestClient(serve.app)
    warm, _ = to_record(request(build_state(tok, filler, 2048)))
    server.probs(warm)   # Metal kernels compiled outside the timed requests; its cached state measures the keys and values per token
    with server.lock:
        kv = sum(x.nbytes for c in next(reversed(server.prefix_cache.entries.values()))[1] if hasattr(c, "offset") for x in c.state) / 2048
    report["kv_kb_per_token"] = round(kv / 1e3, 1)

    transient, stop = TRANSIENT, None
    for n in lengths:
        waited, busy = wait_quiet()
        avail = psutil.virtual_memory().available
        # a new state's pass holds its own cache, the branch pass's copy of it and the older cached states make_room keeps (at most max_tokens - n tokens, up to n of them here), plus the chunk passes' transient
        resident = 2 * n + min(n, server.prefix_cache.max_tokens - n)
        projected = transient + kv * resident
        if stop or projected + MARGIN > avail:
            why = stop or f"projected working set {gb(projected)} GB ({gb(transient)} GB transient + {kv / 1e3:.0f} KB x {resident} cached tokens) + {gb(MARGIN)} GB margin exceeds the {gb(avail)} GB available"
            report["lengths"][n] = {"skipped": why, "waited_s": waited, "busy": busy}
            print(n, report["lengths"][n], flush=True); save(); continue
        with server.lock: server.prefix_cache.clear()
        gc.collect(); mx.clear_cache(); mx.reset_peak_memory()
        sout = psutil.swap_memory().sout
        row = {"available_gb_before": gb(avail), "waited_s": waited, "busy_before": busy}
        try:
            with RSSPeak() as rss:
                cold = {}
                for v, (rec, meta) in records[n].items():
                    ps, stats = server.probs(rec)
                    assert stats["state_tokens"] == n and not stats["prefix_cache_hit"], stats
                    cold[v] = {"ms": stats["latency_ms"], "answers": named(ps, meta), "correct": correct(named(ps, meta), VARIANTS[v][0])}
                rec, meta = records[n]["planted"]
                cached = [server.probs(rec)[1] for _ in range(max(0, a.reps - 1))]
                assert all(s["prefix_cache_hit"] for s in cached), cached
                body = client.post("/v1/systemone", json=request(states[n]["planted"]).model_dump(exclude_none=True))
                assert body.status_code == 200, body.text[:500]
                http = body.json()
            row.update({
                "state_tokens": n, "tokens": stats["tokens"],
                "first_ms": cold["start"]["ms"], "cold_ms": statistics.median(c["ms"] for c in cold.values()), "cold_ms_all": {v: c["ms"] for v, c in cold.items()},
                "cached_ms": statistics.median([s["latency_ms"] for s in cached] + [http["latency_ms"]]),
                "cached_ms_all": [s["latency_ms"] for s in cached] + [http["latency_ms"]],
                "http_office": http["answers"]["office"]["choice"], "http_input_tokens": http["usage"]["input_tokens"],
                "mlx_peak_gb": gb(mx.get_peak_memory()), "mlx_working_gb": gb(mx.get_peak_memory() - weights), "rss_peak_gb": gb(rss.peak), "footprint_peak_gb": gb(rss.footprint),
                "swap_out_gb": gb(psutil.swap_memory().sout - sout),
                "all_correct": all(all(c["correct"].values()) for c in cold.values()),
                "variants": cold,
                "office_p_true": {v: round(c["answers"]["office"][VARIANTS[v][0].lower()], 4) for v, c in cold.items()},
                "planted_vs_start": agreement(cold["planted"]["answers"], cold["start"]["answers"]),
            })
            if n in reference:
                row["torch_fp32_cpu"] = {**reference[n], "vs_mlx": agreement(cold["planted"]["answers"], reference[n]["answers"])}
            row["busy_after"] = others()
            transient = max(transient, mx.get_peak_memory() - weights - kv * resident)
            if row["swap_out_gb"] > 0.5: stop = f"not run: the {n}-token length pushed {row['swap_out_gb']} GB to swap"
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {str(error)[:500]}"
        report["lengths"][n] = row
        print(n, {k: v for k, v in row.items() if k not in ("variants",)}, flush=True); save()

    state = build_state(tok, filler, ADMISSION_TOKENS)
    r = client.post("/v1/systemone", json=request(state).model_dump(exclude_none=True))
    report["admission"] = {"state_tokens": ADMISSION_TOKENS, "limit": SERVE_MAX_STATE, "status": r.status_code, "detail": r.json().get("detail")}
    print("admission", report["admission"], flush=True)
    with server.lock: server.prefix_cache.clear()
    server.close(); save()


if __name__ == "__main__":
    main()
