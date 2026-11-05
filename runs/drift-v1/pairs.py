"""Pairwise raw-logit differences between the reads of one probe directory (any checkpoint).

    python3 runs/drift-v1/pairs.py runs/drift-v1/h100-kev4b-fp32-ref --temperature 2.96
"""
import argparse
import json
import math
from pathlib import Path


def softmax(z, t):
    m = max(z); e = [math.exp((v - m) / t) for v in z]; s = sum(e); return [v / s for v in e]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--temperature", type=float, default=1.0)
    a = ap.parse_args()
    reads = {f.stem: {(r["id"], r["question"]): r["logits"] for r in json.loads(f.read_text())}
             for f in sorted(Path(a.dir).glob("*.json")) if f.stem not in ("env", "comparison", "pairs")}
    names, out = list(reads), {}
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            A, B = reads[x], reads[y]
            dl = sorted(max(abs(p - q) for p, q in zip(A[k], B[k])) for k in A)
            dp = max(max(abs(p - q) for p, q in zip(softmax(A[k], a.temperature), softmax(B[k], a.temperature))) for k in A)
            flips = sum(max(range(len(A[k])), key=A[k].__getitem__) != max(range(len(B[k])), key=B[k].__getitem__) for k in A)
            out[f"{x} vs {y}"] = r = {"rows": len(A), "identical_rows": sum(A[k] == B[k] for k in A), "max_dlogit": dl[-1],
                                      "median_dlogit": dl[len(dl) // 2], "max_dp": dp, "flips": flips, "temperature": a.temperature}
            print(f"{x:24s} vs {y:24s} identical {r['identical_rows']:3d}/{r['rows']}  max|dz| {r['max_dlogit']:.3g}  med|dz| {r['median_dlogit']:.3g}  max|dp| {dp:.3g}  flips {flips}")
    (Path(a.dir) / "pairs.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
