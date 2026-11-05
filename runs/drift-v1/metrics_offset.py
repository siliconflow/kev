"""What the causal-conv1d kernel change does to a read's reported metrics: a committed read's rows with their logits
replaced by a probe read's (raw logits, divided by the committed read's temperature), summarised by kev.benchmark.summarize.

    uv run python runs/drift-v1/metrics_offset.py --reference runs/breadth-v1-kev-27b --probe runs/drift-v1/h200-breadth \
        --out runs/drift-v1/h200-breadth/metrics.json
"""
import argparse
import json
import math
from pathlib import Path

from kev.benchmark import summarize

METRICS = ("n", "acc", "nll", "brier", "ece", "coverage_at_5pct_error", "aurc")


def softmax(z):
    m = max(z); e = [math.exp(v - m) for v in z]; s = sum(e); return [v / s for v in e]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = json.loads((Path(a.reference) / "rows.json").read_text())
    out = {"reference": a.reference, "probe": a.probe}
    for f in sorted(Path(a.probe).glob("new_conv_*.json")):
        raw = {(r["id"], r["question"]): r["logits"] for r in json.loads(f.read_text())}
        swapped = []
        for r in rows:
            t = r.get("inference_temperature", 1.0)
            z = [v / t for v in raw[(r["id"], r["question"])]]
            swapped.append({**r, "logits": z, "p": softmax(z)})
        s = summarize(swapped)
        out[f.stem] = {k: s["clean"].get(k) for k in METRICS}
    s = summarize(rows)
    out["reference_as_committed"] = {k: s["clean"].get(k) for k in METRICS}
    print(json.dumps(out, indent=1))
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
