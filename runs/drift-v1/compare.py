"""Compare the probe's raw-logit reads with each other and with the two published reads.

    python3 runs/drift-v1/compare.py runs/drift-v1/h200-a [more dirs]

References: runs/r6-27bv2-s2-semif (2026-09-23, raw, T 1.0) and runs/rel27-public/3-jaredpalmer_kev-27b_v1-lora
(2026-09-30, logits divided by T 1.3819; multiplied back here). Reports max |dlogit| (raw), max |dp| at T 1.3819 and
argmax flips, and whether the logits are bit-identical (as written by json, float64 of the fp32 values).
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T = 1.381912879967776


def key(r): return (r["id"], r["question"])


def load_ref(path, scale=1.0):
    return {key(r): [v * scale for v in r["logits"]] for r in json.loads(Path(path).read_text())}


def softmax(z, t):
    m = max(z); e = [math.exp((v - m) / t) for v in z]; s = sum(e); return [v / s for v in e]


def diff(a, b):
    ks = sorted(a)
    assert set(ks) == set(b), "row sets differ"
    dl = [max(abs(x - y) for x, y in zip(a[k], b[k])) for k in ks]
    dp = [max(abs(x - y) for x, y in zip(softmax(a[k], T), softmax(b[k], T))) for k in ks]
    flips = sum(max(range(len(a[k])), key=a[k].__getitem__) != max(range(len(b[k])), key=b[k].__getitem__) for k in ks)
    exact = sum(a[k] == b[k] for k in ks)
    s = sorted(dl)
    return {"rows": len(ks), "identical_rows": exact, "max_dlogit": max(dl), "median_dlogit": s[len(s) // 2], "max_dp": max(dp), "flips": flips}


def main():
    refs = {"old_0923": load_ref(ROOT / "runs/r6-27bv2-s2-semif/rows.json"),
            "public_0930": load_ref(ROOT / "runs/rel27-public/3-jaredpalmer_kev-27b_v1-lora/rows.json", T)}
    reads = dict(refs)
    for d in sys.argv[1:]:
        for f in sorted(Path(d).glob("*.json")):
            if f.name in ("env.json", "comparison.json"): continue
            reads[f"{Path(d).name}/{f.stem}"] = load_ref(f)
    names = list(reads)
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out[f"{a} vs {b}"] = r = diff(reads[a], reads[b])
            print(f"{a:34s} vs {b:34s} identical {r['identical_rows']:3d}/{r['rows']}  max|dz| {r['max_dlogit']:.4g}  med|dz| {r['median_dlogit']:.3g}  max|dp| {r['max_dp']:.4g}  flips {r['flips']}")
    return out


if __name__ == "__main__":
    result = main()
    if len(sys.argv) > 1:
        (Path(sys.argv[1]) / "comparison.json").write_text(json.dumps(result, indent=1) + "\n")
