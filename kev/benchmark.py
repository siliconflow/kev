import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from kev.data import materialize
from kev.evaluate import ece, load, resolve_run
from kev.model import encode
from kev.suite import digest, load_split, record_digest, write_json

EPSILON = 1e-9


def default_device():
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def api_request(record):
    return {"state": record["state"], "questions": {
        qid: {k: v for k, v in q.items() if k in ("type", "instructions", "criteria")}
        for qid, q in record["questions"].items()}}


def labels(q):
    if q["type"] == "choice":
        keys = list(q["criteria"])
        return keys, keys.index(q["label"])
    if q["type"] == "noul":
        return ["false", "true"], int(q["label"])
    return [str(i) for i in range(len(q["criteria"]))], int(q["label"])


def validate_distribution(raw, keys):
    if set(raw) != set(keys):
        raise ValueError("probability keys do not match requested options")
    p = np.array([raw[k] for k in keys], dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("non-finite or out-of-range probabilities")
    total = float(p.sum())
    if total <= 0 or abs(total - 1) > max(1e-5, len(keys) * 0.005 + 1e-8):
        raise ValueError(f"invalid probability sum: {total}")
    return p / total, total


def prediction_rows(record, prediction):
    if set(prediction["probabilities"]) != set(record["questions"]):
        raise ValueError("answer IDs differ from request IDs")
    meta = record["_meta"]
    rows = []
    for qid, q in record["questions"].items():
        keys, y = labels(q)
        p, total = validate_distribution(prediction["probabilities"][qid], keys)
        row = {"id": meta["id"], "group": meta["group_id"], "question": qid,
               "source": meta["source"], "task": q["src"], "type": q["type"],
               "variant": meta["variant"], "keys": keys, "label": y, "control_id": meta.get("control_id"),
               "pair_id": meta.get("pair_id"), "sibling": meta.get("sibling"), # suites frozen before parent_id existed stored the parent's id in group_id for variants
               "parent": meta.get("parent_id") or (meta["id"] if meta["variant"] == "clean" else meta["group_id"]),
               "p": p.tolist(), "raw_probability_sum": total, "zero_count": int((p == 0).sum())}
        rows.append(row)
    return rows


def metrics(rows, temperature=1.0):
    if not rows:
        raise ValueError("cannot score an empty population")
    nll, acc, conf, brier, mae, rps = [], [], [], [], [], []
    for row in rows:
        p = np.array(row["p"])
        if temperature != 1:
            z = np.log(np.maximum(p, EPSILON)) / temperature
            p = np.exp(z - z.max()); p /= p.sum()
        y = row["label"]
        target = np.eye(len(p))[y]
        nll.append(-math.log(max(float(p[y]), EPSILON)))
        acc.append(int(p.argmax() == y)); conf.append(float(p.max()))
        brier.append(float(((p - target) ** 2).sum()))
        if row["type"] == "score":
            mae.append(abs(float(p @ np.arange(len(p))) - y))
            rps.append(float(((p.cumsum()[:-1] - target.cumsum()[:-1]) ** 2).mean()))
    result = {"n": len(rows), "nll": float(np.mean(nll)), "acc": float(np.mean(acc)),
              "ece": ece(conf, acc), "brier": float(np.mean(brier)), "mean_conf": float(np.mean(conf))}
    confidence, correct = np.asarray(conf), np.asarray(acc, dtype=bool)
    high = confidence >= 0.9
    result.update(confident_error_rate=float(np.mean(high & ~correct)), coverage_at_0_9=float(high.mean()),
                  accuracy_at_0_9=float(correct[high].mean()) if high.any() else None,
                  coverage_at_5pct_error=coverage_at_error(confidence, correct, 0.05), coverage_at_1pct_error=coverage_at_error(confidence, correct, 0.01),
                  # signed over-confidence (mean top probability minus accuracy) and errors within the top confidence bins;
                  # the sign is diagnostic: untrained readouts run positive, outcome-trained ones near zero or negative
                  confidence_bias=float(confidence.mean() - correct.mean()),
                  top_bins={str(t): {"n": int((confidence >= t).sum()), "errors": int(((confidence >= t) & ~correct).sum()),
                                     "error_rate": float((~correct[confidence >= t]).mean()) if (confidence >= t).any() else None} for t in (0.9, 0.95, 0.99)})
    result["selective"] = {}
    for fraction in (0.5, 0.8):
        cutoff = np.sort(confidence)[-max(1, math.ceil(len(rows) * fraction))]
        selected = confidence >= cutoff
        result["selective"][str(fraction)] = {"coverage": float(selected.mean()), "accuracy": float(correct[selected].mean()),
                                            "confidence_cutoff": float(cutoff)}
    if mae:
        result.update(score_mae=float(np.mean(mae)), ranked_probability_score=float(np.mean(rps)))
    return result


def coverage_at_error(confidence, correct, budget):
    """Selective automation: the largest share of decisions that can be accepted, in descending confidence order, while
    the empirical error among the accepted stays <= budget (jev-benchmarks' "coverage at a fixed error budget"). A model
    whose probabilities are honest gets high coverage; one that is confidently wrong gets little, whatever its accuracy."""
    order = np.argsort(-np.asarray(confidence), kind="stable"); wrong = np.cumsum(~np.asarray(correct, dtype=bool)[order])
    accepted = np.arange(1, len(order) + 1)
    ok = np.nonzero(wrong <= budget * accepted)[0]
    return float(accepted[ok[-1]] / len(order)) if len(ok) else 0.0


def unknowable_report(rows):
    """Confidence on records whose deciding evidence was removed (source 'unknowable') against their intact controls.
    Accuracy on the unknowable records is meaningless by construction; what is scored is whether the model knows it
    cannot know: mean max-probability and the share of records answered at >= 0.9."""
    unk = [r for r in rows if r["source"] == "unknowable"]; ctl = [r for r in rows if r["source"] == "unknowable_control"]
    if not unk: return None
    conf = lambda rs: [float(max(r["p"])) for r in rs]
    by_id = {r["id"]: r for r in ctl}
    paired = [(max(r["p"]), max(by_id[r["control_id"]]["p"])) for r in unk if r.get("control_id") in by_id]
    return {"n": len(unk), "mean_max_p": float(np.mean(conf(unk))), "share_at_0_9": float(np.mean([c >= 0.9 for c in conf(unk)])),
            "control_mean_max_p": float(np.mean(conf(ctl))) if ctl else None, "control_share_at_0_9": float(np.mean([c >= 0.9 for c in conf(ctl)])) if ctl else None,
            "control_acc": float(np.mean([int(np.argmax(r["p"]) == r["label"]) for r in ctl])) if ctl else None,
            "paired_confidence_drop": float(np.mean([c - u for u, c in paired])) if paired else None,
            "share_less_confident_than_control": float(np.mean([u < c for u, c in paired])) if paired else None}


def grouped_metrics(rows, key, temperature=1.0):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return {name: metrics(group, temperature) for name, group in sorted(groups.items())}


def fit_temperature(rows):
    clean = [row for row in rows if row["variant"] == "clean"]
    candidates = np.exp(np.linspace(np.log(0.25), np.log(4), 81))
    losses = [np.mean([m["nll"] for m in grouped_metrics(clean, "task", float(t)).values()]) for t in candidates]
    return float(candidates[int(np.argmin(losses))])


def paired_bootstrap(candidate, reference, samples=1000, seed=0, metric="nll"):
    def index(rows):
        return {(r["id"], r["question"]): r for r in rows if r["variant"] == "clean"}
    a, b = index(candidate), index(reference)
    if not a or a.keys() != b.keys():
        raise ValueError("paired comparison requires identical complete clean examples")
    groups = defaultdict(list)
    for key, row in a.items():
        other = b[key]
        if row["keys"] != other["keys"] or row["label"] != other["label"]:
            raise ValueError("paired comparison labels or option order differ")
        groups[(row["source"], row["group"])].append((row["task"], metrics([row])[metric] - metrics([other])[metric]))
    sources = defaultdict(list)
    for (source, group), pairs in groups.items():
        sources[source].append(pairs)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        tasks = defaultdict(list)
        for units in sources.values():
            for i in rng.integers(0, len(units), size=len(units)):
                for task, delta in units[i]:
                    tasks[task].append(delta)
        values.append(float(np.mean([np.mean(v) for v in tasks.values()])))
    observed = np.mean([m[metric] for m in grouped_metrics(list(a.values()), "task").values()]) - np.mean([m[metric] for m in grouped_metrics(list(b.values()), "task").values()])
    return {f"macro_{metric}_delta": float(observed), "ci95": np.quantile(values, [0.025, 0.975]).tolist(),
            "samples": samples, "unit": "source-stratified original record; sibling questions stay together"}


def summarize(rows, temperature=1.0, heldout_sources=("mnli", "sst5")):
    clean = [r for r in rows if r["variant"] == "clean"]
    tasks = grouped_metrics(clean, "task")
    variants = grouped_metrics(rows, "variant")
    lookup = {(r["id"], r["question"]): r for r in clean}
    diffs, flips = [], []
    for row in rows:
        if row["variant"] == "permuted" and row["type"] == "choice":
            original = lookup[(row["parent"], row["question"])]
            aligned = [row["p"][row["keys"].index(k)] for k in original["keys"]]
            diffs.append(float(np.max(np.abs(np.array(aligned) - original["p"]))))
            flips.append(int(np.argmax(aligned) != np.argmax(original["p"])))
    from kev.contrastive import paired_flip
    knowable = [r for r in clean if r["source"] != "unknowable"]     # unknowable records are scored on confidence, never on accuracy
    return {"objective": -float(np.mean([v["nll"] for k, v in tasks.items() if not k.startswith("unknowable_") or k.startswith("unknowable_control")])),
            "paired_flip": paired_flip(clean), "unknowable": unknowable_report(clean),
            "clean": metrics(knowable), "tasks": tasks, "variants": variants,
            "heldout_tasks": grouped_metrics([r for r in clean if r["source"] in heldout_sources], "task") if any(r["source"] in heldout_sources for r in clean) else {},
            "permutation": {"n": len(diffs), "mean_max_delta": float(np.mean(diffs)) if diffs else None,
                            "flip_rate": float(np.mean(flips)) if flips else None},
            "temperature": temperature, "calibrated_clean": metrics(clean, temperature),
            "metric_policy": {"nll_floor": EPSILON, "renormalize_returned_probabilities": True,
                              "raw_sums_outside_1e_5": sum(abs(r["raw_probability_sum"] - 1) > 1e-5 for r in rows),
                              "returned_zeros": sum(r["zero_count"] for r in rows)}}


def sync(device):
    if device == "mps": torch.mps.synchronize()
    elif device == "cuda": torch.cuda.synchronize()


class LocalPredictor:
    def __init__(self, run, device):
        self.run = resolve_run(run)
        if device == "cuda":
            # evaluation is fp32-exact: TF32 (10-bit mantissa) moves probabilities by ~1e-3, the isolation gate's tolerance
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
        self.tok, self.model = load(self.run, device)
        self.device = device

    def __call__(self, record):
        enc = self.model.encode(self.tok, materialize(record), strict=True)
        if len(enc["ids"]) > 2048:
            raise ValueError("packed request exceeds frozen 2048-token limit")
        sync(self.device)
        start = time.perf_counter()
        ps = self.model.probs(enc)
        sync(self.device)
        return {"probabilities": {qid: dict(zip(labels(q)[0], p.tolist())) for (qid, q), p in zip(record["questions"].items(), ps)},
                "latency_ms": 1000 * (time.perf_counter() - start), "input_tokens": len(enc["ids"])}


class RemotePredictor:
    """Score any TypeSafe System One-compatible endpoint (POST <base_url>/v1/systemone) on frozen records. Probabilities are
    taken from the response as returned (renormalised by validate_distribution like every other predictor). Records the
    server-reported model id so the manifest can pin what was scored."""

    def __init__(self, base_url, model="kev-latest", api_key="local", timeout=120, retries=3):
        import urllib.request
        self.base_url, self.model, self.api_key, self.timeout, self.retries = base_url.rstrip("/"), model, api_key, timeout, retries
        self.served_model = None; self._request = urllib.request

    def __call__(self, record):
        payload = json.dumps({**api_request(record), "model": self.model}).encode()
        req = self._request.Request(f"{self.base_url}/v1/systemone", data=payload, method="POST",
                                    headers={"content-type": "application/json", "authorization": f"Bearer {self.api_key}"})
        last = None
        for attempt in range(self.retries):
            try:
                start = time.perf_counter()
                with self._request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read())
                latency = 1000 * (time.perf_counter() - start)
                break
            except Exception as error:   # 5xx / timeouts: retry with backoff; anything persistent surfaces as a rejected record
                last = error; time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"remote endpoint failed after {self.retries} attempts: {last}")
        self.served_model = body.get("model", self.served_model)
        probs = {}
        for qid, q in record["questions"].items():
            a = body["answers"][qid]
            if q["type"] == "noul": probs[qid] = {"true": float(a["noul"]), "false": 1 - float(a["noul"])}
            else: probs[qid] = {str(k): float(v) for k, v in a["probabilities"].items()}
        return {"probabilities": probs, "latency_ms": latency, "input_tokens": (body.get("usage") or {}).get("input_tokens")}


def evaluate_records(records, predictor, directory, temperature=1.0, heldout_sources=("mnli", "sst5"), skip_overlong=False):
    """skip_overlong: for external data that was not frozen to Kev's context, records the model cannot encode (state > 384
    tokens or > 2048 packed) are counted in coverage["rejected_records"] and listed in rejected.json instead of aborting.
    Frozen suites never need this; reports must state that rejected records count as wrong in any headline number."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    coverage = {"requested_records": len(records), "requested_questions": sum(len(r["questions"]) for r in records),
                "evaluated_records": 0, "evaluated_questions": 0, "rejected_records": 0, "truncated_records": 0}
    rows, latencies, rejected = [], [], []
    with (directory / "predictions.jsonl").open("w") as output:
        for record in records:
            try:
                pred = predictor(record)
                new_rows = prediction_rows(record, pred)
            except ValueError as error:
                if skip_overlong and ("exceeds" in str(error) or "tokens" in str(error)):
                    coverage["rejected_records"] += 1; rejected.append({"id": record["_meta"]["id"], "error": str(error)}); continue
                coverage["rejected_records"] += 1
                write_json(directory / "failure.json", {"coverage": coverage, "record_id": record["_meta"]["id"], "error_type": type(error).__name__})
                raise
            except Exception as error:
                coverage["rejected_records"] += 1
                write_json(directory / "failure.json", {"coverage": coverage, "record_id": record["_meta"]["id"], "error_type": type(error).__name__})
                raise
            output.write(json.dumps({"request_sha256": record_digest(api_request(record)), "id": record["_meta"]["id"],
                                     "prediction": pred, "rows": new_rows}, allow_nan=False) + "\n")
            output.flush()
            rows.extend(new_rows)
            latencies.append(pred["latency_ms"])
            coverage["evaluated_records"] += 1
            coverage["evaluated_questions"] += len(new_rows)
            if coverage["evaluated_records"] % 50 == 0:
                print(f"evaluated {coverage['evaluated_records']}/{len(records)}", flush=True)
    write_json(directory / "rows.json", rows)
    if rejected: write_json(directory / "rejected.json", rejected)
    report = summarize(rows, temperature, heldout_sources)
    report.update(coverage=coverage, latency_ms={"median": float(np.median(latencies)), "p95": float(np.quantile(latencies, .95))})
    write_json(directory / "report.json", report)
    return report, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="checkpoint dir or Hub id (local scoring)")
    ap.add_argument("--remote", help="base URL of a System One-compatible endpoint to score instead of a local checkpoint")
    ap.add_argument("--remote-model", default="kev-latest")
    ap.add_argument("--suite", help="frozen suite directory (scores its development partition)")
    ap.add_argument("--data", help="your own labelled requests, one JSON object per line (kev.data.load_records); an alternative to --suite")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default=default_device())
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--date_facts", action="store_true", help="apply kev.api.with_date_facts to every state before scoring (the opt-in serving preprocessor); reported in report.json")
    a = ap.parse_args()
    if bool(a.run) == bool(a.remote): ap.error("give exactly one of --run or --remote")
    if bool(a.suite) == bool(a.data): ap.error("give exactly one of --suite or --data")
    if a.data:
        from kev.data import load_records
        records, heldout, split, source_hash = load_records(a.data), [], "custom", digest(Path(a.data))
    else:
        split = "test" if a.allow_test else "development"
        records = load_split(a.suite, split, allow_test=a.allow_test)
        heldout = json.loads((Path(a.suite) / "manifest.json").read_text())["holdout_sources"]; source_hash = digest(Path(a.suite) / "manifest.json")
    if a.date_facts:
        from kev.api import with_date_facts
        records = [{**r, "state": with_date_facts(r["state"])} for r in records]
    import os
    predictor = RemotePredictor(a.remote, a.remote_model, os.environ.get("KEV_REMOTE_API_KEY", "local")) if a.remote else LocalPredictor(a.run, a.device)
    report, _ = evaluate_records(records, predictor, a.out, heldout_sources=tuple(heldout), skip_overlong=bool(a.data))
    report.update(suite_sha256=source_hash, data=a.data, date_facts=a.date_facts, run=a.run or a.remote, split=split, calibration_applied=False,
                  remote={"base_url": a.remote, "requested_model": a.remote_model, "served_model": predictor.served_model} if a.remote else None)
    write_json(Path(a.out) / "report.json", report)
    print(json.dumps({"objective": report["objective"], "clean": report["clean"], "coverage": report["coverage"]}, indent=2))


if __name__ == "__main__":
    main()
