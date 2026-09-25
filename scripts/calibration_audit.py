import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kev.metrics import metrics, paired_bootstrap, probabilities_at_temperature, risk_coverage_curve
from kev.checkpoint import read_meta
from kev.suite import digest, load_split, read_json, write_json

RUNS = {
    "Kev-0.8B": "night2-08b-du2/00-trial-0",
    "Kev-4B": "night2-4b-du/00-trial-0",
    "Kev-9B": "night2-9b-du/00-trial-0",
}


def read_rows(path):
    return [r for r in read_json(path) if r["variant"] == "clean" and r["source"] != "unknowable"]


def tempered(rows, temperature):
    return [{**r, "p": probabilities_at_temperature(r, temperature).tolist(),
             "inference_temperature": r.get("inference_temperature", 1.0) * temperature,
             **({"logits": (np.asarray(r["logits"]) / temperature).tolist()} if "logits" in r else {})} for r in rows]


def legacy_coverage(rows, budget=0.05):
    confidence = np.asarray([max(r["p"]) for r in rows])
    correct = np.asarray([np.argmax(r["p"]) == r["label"] for r in rows])
    order = np.argsort(-confidence, kind="stable")
    n = np.arange(1, len(rows) + 1)
    ok = np.flatnonzero(np.cumsum(~correct[order]) <= budget * n)
    return float(n[ok[-1]] / len(rows)) if len(ok) else 0.0


def describe(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["task"]].append(row)
    conf = [max(r["p"]) for r in rows]
    correct = [np.argmax(r["p"]) == r["label"] for r in rows]
    return {
        "micro": metrics(rows), "tasks": {k: metrics(v) for k, v in sorted(grouped.items())},
        "risk_coverage": risk_coverage_curve(conf, correct),
        "legacy_order_dependent_coverage_5pct": legacy_coverage(rows),
        "largest_confidence_tie": max(Counter(conf).values()),
        "probability_zero_count": sum(p == 0 for r in rows for p in r["p"]),
        "confident_errors_by_task": dict(Counter(r["task"] for r, p, ok in zip(rows, conf, correct) if p >= 0.9 and not ok)),
    }


def failure_packet(rows, records, directory, seed):
    original = {r["_meta"]["id"]: r for r in records}
    failures = [r for r in rows if max(r["p"]) >= 0.9 and np.argmax(r["p"]) != r["label"]]
    rng = random.Random(seed)
    controls = defaultdict(list)
    for row in rows:
        if np.argmax(row["p"]) == row["label"] and max(row["p"]) >= 0.9:
            controls[(row["task"], len(row["keys"]))].append(row)
    for group in controls.values():
        rng.shuffle(group)
    selected = []
    for row in failures:
        selected.append(("confident_error", row))
        matches = controls[(row["task"], len(row["keys"]))]
        if matches:
            selected.append(("matched_correct", matches.pop()))
    rng.shuffle(selected)
    blind, key = [], []
    for i, (kind, row) in enumerate(selected):
        record = original[row["id"]]
        question = record["questions"][row["question"]]
        case = f"audit-{i:03d}"
        blind.append({"case": case, "state": record["state"],
                      "question": {k: v for k, v in question.items() if k in ("type", "instructions", "criteria")}})
        key.append({"case": case, "kind": kind, "id": row["id"], "question": row["question"], "task": row["task"],
                    "label": row["keys"][row["label"]], "prediction": row["keys"][int(np.argmax(row["p"]))],
                    "probabilities": dict(zip(row["keys"], row["p"]))})
    write_json(directory / "failure_cases_blind.json", blind)
    write_json(directory / "failure_cases_key.json", key)
    return {"cases": len(blind), "failures": len(failures), "selection_seed": seed,
            "review_status": "unreviewed; model/teacher disagreement is not a label correction"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=2000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    suite = ROOT / "evals/v4/transfer-v4"
    records = load_split(suite, "development")
    report = {"metric_version": 2, "suite_sha256": digest(suite / "manifest.json"), "split": "development",
              "input_records_including_variants": len(records), "models": {}, "paired": {},
              "caveats": ["Historical files are read-only. New metric values do not overwrite old reports.",
                          "Historical Kev rows lack logits. Temperature replay from probabilities uses the declared 1e-9 floor and is approximate.",
                          "Coverage is an in-sample descriptive threshold envelope, not a deployment guarantee.",
                          "AURC uses a right-step integral over complete confidence tie groups.",
                          "Bootstrap intervals condition on the observed source mixture and do not include training-seed variability."]}
    compared = {}
    for name, run in RUNS.items():
        path = ROOT / "runs" / run
        rows_path = path / "transfer/rows.json"
        rows = read_rows(rows_path)
        temperature = read_meta(path / "checkpoint").temperature
        cal = tempered(rows, temperature)
        report["models"][name] = {"rows": str(rows_path.relative_to(ROOT)), "rows_sha256": digest(rows_path),
                                  "head_sha256": digest(path / "checkpoint/head.pt"), "temperature": temperature,
                                  "raw": describe(rows), "temperature_replay": describe(cal),
                                  "logits_available": all("logits" in r for r in rows)}
        compared[name] = cal
        if name == "Kev-9B":
            report["failure_audit"] = failure_packet(cal, records, out, 20260921)
    jev_path = ROOT / "runs/jev-transfer-v4/rows.json"
    jev = read_rows(jev_path)
    compared["Jev"] = jev
    report["models"]["Jev"] = {"rows": str(jev_path.relative_to(ROOT)), "rows_sha256": digest(jev_path), "returned": describe(jev)}
    for name, rows in compared.items():
        if name == "Jev":
            continue
        report["paired"][name] = {metric: paired_bootstrap(rows, jev, samples=a.samples, seed=20260921,
                                                          metric=metric, aggregation="micro")
                                  for metric in ("acc", "coverage_at_5pct_error", "aurc", "brier")}
    write_json(out / "report.json", report)
    for name, entry in report["models"].items():
        for kind in ("raw", "temperature_replay", "returned"):
            if kind not in entry:
                continue
            m = entry[kind]["micro"]
            print(f"{name:9} {kind:18} n={m['n']} acc={m['acc']:.3f} cov5={m['coverage_at_5pct_error']:.3f} "
                  f"old_cov5={entry[kind]['legacy_order_dependent_coverage_5pct']:.3f} aurc={m['aurc']:.4f} "
                  f"brier={m['brier']:.3f} high_error_share={m['confident_error_rate']:.3f} "
                  f"error_among_accepted={m['error_rate_at_0_9']}")
    for name, comparison in report["paired"].items():
        result = comparison["coverage_at_5pct_error"]
        print(name, "minus Jev coverage:", result["micro_coverage_at_5pct_error_delta"], result["ci95"])


if __name__ == "__main__":
    main()
