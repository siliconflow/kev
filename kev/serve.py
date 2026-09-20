"""FastAPI sidecar for the playground: loads one checkpoint, exposes prefill-only decisions.

Run: uv run --extra serve python -m kev.serve --run runs/kev --port 8008
"""
import argparse, json, os, random, re, threading, time
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .api import SystemOneRequest, to_record, to_answers, output_tokens
from .data import DISTRACTORS, NONE
from .evaluate import load
from .model import encode

# inference limits (training used 384/640); per-branch cap mirrors Jev's ~32k, bounded by the base model window
INFER_MAX_STATE, INFER_MAX_BRANCH = 8192, 8192

app = FastAPI(title="kev")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
STATE = {"run": None, "tok": None, "model": None, "dev": None, "lock": threading.Lock()}


class Question(BaseModel):
    instr: str
    options: list[str]


class Record(BaseModel):
    state: str
    questions: list[Question]


class PermuteReq(BaseModel):
    state: str
    question: Question
    n_perm: int = 6
    seed: int = 0


def _rec(r: Record):
    return {"state": r.state, "questions": [{"instr": q.instr, "options": q.options, "label": 0} for q in r.questions]}


def _probs(rec):
    tok, model, dev = STATE["tok"], STATE["model"], STATE["dev"]
    try: enc = model.encode(tok, rec, max_state=INFER_MAX_STATE, max_branch=INFER_MAX_BRANCH)
    except ValueError as e: raise HTTPException(422, str(e))
    with STATE["lock"]:
        if dev == "mps": torch.mps.synchronize()
        t = time.time(); ps = model.probs(enc)
        if dev == "mps": torch.mps.synchronize()
        dt = time.time() - t
    METRICS["latency_ms"].append(dt * 1000)
    return [p.tolist() for p in ps], {"tokens": len(enc["ids"]), "state_tokens": enc["seg"].count(0), "latency_ms": round(dt * 1000, 1)}


METRICS = {"latency_ms": [], "requests": 0}

_LAT_BUCKETS = [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]   # ms; bracket kev's ~30ms-1s prefill range


@app.get("/metrics")
def metrics():
    """Prometheus text exposition (the platform's prometheusScraper polls this); no client dependency."""
    from fastapi.responses import PlainTextResponse
    lats = METRICS["latency_ms"]
    def bucket_vals():
        # cumulative Prometheus buckets: le=b counts every x <= b, so the values are non-decreasing in b
        return [(b, sum(1 for x in lats if x <= b)) for b in _LAT_BUCKETS]
    lines = [f"kev_requests_total {METRICS['requests']}", f"kev_inferences_total {len(lats)}"]
    if lats:
        lines += [f"kev_inference_latency_ms_count {len(lats)}", f"kev_inference_latency_ms_sum {sum(lats):.1f}"]
        lines += [f'kev_inference_latency_ms_bucket{{le="{b}"}} {c}' for b, c in bucket_vals()]
        lines.append(f'kev_inference_latency_ms_bucket{{le="+Inf"}} {len(lats)}')
    lines.append(f'kev_model_info{{run="{STATE.get("run") or ""}",base="{STATE.get("base") or ""}",device="{STATE.get("dev") or ""}"}} 1')
    return PlainTextResponse("\n".join(lines) + "\n")


@app.post("/v1/systemone")
def systemone(req: SystemOneRequest):
    """TypeSafe-compatible endpoint: typed questions in, typed answers out, one prefill pass."""
    METRICS["requests"] += 1
    rec, meta = to_record(req)
    ps, m = _probs(rec)
    answers = to_answers(ps, meta)
    return {"model": req.model, "answers": answers, "usage": {"input_tokens": m["tokens"], "output_tokens": output_tokens(STATE["tok"], answers)}, "latency_ms": m["latency_ms"]}


class PermuteSystemOne(BaseModel):
    request: SystemOneRequest
    question: str
    n_perm: int = 6
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
        req = r.request.model_copy(update={"questions": {r.question: q.model_copy(update={"criteria": {k: q.criteria[k] for k in order}})}})
        rec, meta = to_record(req); ps, m = _probs(rec)
        a = to_answers(ps, meta)[r.question]
        runs.append({"order": order, "probabilities": a["probabilities"], "choice": a["choice"], "latency_ms": m["latency_ms"]})
    spread = {k: max(x["probabilities"][k] for x in runs) - min(x["probabilities"][k] for x in runs) for k in keys}
    return {"runs": runs, "argmax_stable": len({x["choice"] for x in runs}) == 1, "spread": spread}


@app.post("/v1/systemone/separate")
def systemone_separate(req: SystemOneRequest):
    """Answer each question in its own request against the same state (N passes). For packed-vs-separate comparison."""
    answers, tokens, ms = {}, 0, 0.0
    for qid, q in req.questions.items():
        rec, meta = to_record(req.model_copy(update={"questions": {qid: q}})); ps, m = _probs(rec)
        answers.update(to_answers(ps, meta)); tokens += m["tokens"]; ms += m["latency_ms"]
    return {"model": req.model, "answers": answers, "usage": {"input_tokens": tokens, "output_tokens": output_tokens(STATE["tok"], answers)}, "latency_ms": round(ms, 1)}


@app.get("/v1/models")
def models():
    return {"models": [{"id": "kev-latest", "aliases": ["jev-latest"], "run": STATE["run"], "base": STATE["base"]}]}


@app.get("/api/info")
def info():
    ev = f"{STATE['run']}/eval.json"
    return {"run": STATE["run"], "device": STATE["dev"], "base": STATE["base"], "lora": STATE["lora"],
            "none_option": NONE, "distractors": DISTRACTORS, "has_eval": os.path.exists(ev)}


@app.get("/api/eval")
def eval_json():
    p = f"{STATE['run']}/eval.json"
    if not os.path.exists(p): raise HTTPException(404, "no eval.json for this run")
    return json.load(open(p))


@app.post("/api/predict")
def predict(r: Record):
    """All questions in one block-causal pass (shared state prefix)."""
    ps, meta = _probs(_rec(r))
    return {"probs": ps, **meta}


@app.post("/api/predict_separate")
def predict_separate(r: Record):
    """Each question alone against the same state (N passes). For packed-vs-separate comparison."""
    rec = _rec(r); out, tokens, ms = [], 0, 0.0
    for q in rec["questions"]:
        ps, meta = _probs({"state": rec["state"], "questions": [q]})
        out.append(ps[0]); tokens += meta["tokens"]; ms += meta["latency_ms"]
    return {"probs": out, "tokens": tokens, "latency_ms": round(ms, 1)}


@app.post("/api/permute")
def permute(r: PermuteReq):
    """Shuffle the option order n_perm times; return each ordering's probs mapped back to original indices."""
    rng = random.Random(r.seed); K = len(r.question.options); runs = []
    for i in range(r.n_perm):
        perm = list(range(K))
        if i > 0: rng.shuffle(perm)
        q = {"instr": r.question.instr, "options": [r.question.options[j] for j in perm], "label": 0}
        ps, meta = _probs({"state": r.state, "questions": [q]})
        orig = [0.0] * K
        for pos, j in enumerate(perm): orig[j] = ps[0][pos]
        runs.append({"perm": perm, "probs": orig, "argmax": perm[max(range(K), key=lambda i: ps[0][i])], "latency_ms": meta["latency_ms"]})
    argmaxes = {x["argmax"] for x in runs}
    spread = [max(x["probs"][j] for x in runs) - min(x["probs"][j] for x in runs) for j in range(K)]
    return {"runs": runs, "argmax_stable": len(argmaxes) == 1, "n_distinct_argmax": len(argmaxes), "spread": spread}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/kev")
    ap.add_argument("--fallback", default="runs/smoke")
    ap.add_argument("--port", type=int, default=8008)
    a = ap.parse_args()
    from .evaluate import resolve_run
    is_hub_id = re.fullmatch(r"[\w.-]+/[\w.-]+", a.run) and not os.path.isdir(a.run)
    run = a.run if is_hub_id or os.path.exists(f"{a.run}/head.pt") else a.fallback
    if run != a.run: print(f"{a.run} not found, falling back to {run}")
    label = run                       # what /v1/models reports: the Hub id or run path as given, not the resolved cache path
    run = resolve_run(run)
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    meta = torch.load(f"{run}/head.pt", map_location="cpu")
    tok, model = load(run, dev)
    STATE.update(run=label, tok=tok, model=model, dev=dev, base=meta["base"], lora=meta["lora"])
    print(f"serving {label} ({run}) on {dev} :{a.port}")
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=a.port)


if __name__ == "__main__":
    main()
