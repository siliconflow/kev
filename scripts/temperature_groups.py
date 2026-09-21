"""Trial 2a of the night-2 plan: does one temperature per (question type, option count) transfer out of domain better than
a single temperature? Fitted on the in-distribution development rows (never on transfer), applied to the out-of-domain
development rows. Reports raw / single-T / grouped-T for accuracy (unchanged by construction), Brier, ECE, confident-error
rate and coverage at <= 5 % error. No GPU: rows.json only.

    uv run python scripts/temperature_groups.py
"""
import json, math, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kev.benchmark import metrics  # noqa: E402

RUNS = {"Kev-9B": "q35-9b/01-trial-1", "Kev-4B": "q35-4b-s23/00-trial-0", "Kev-0.8B": "q35-08b/02-trial-2"}
GRID = np.exp(np.linspace(np.log(0.25), np.log(4), 81))


def group_key(row): return (row["type"], min(len(row["keys"]), 6))          # K capped: 6+ options share one temperature (sparse groups)


def fit(rows, key=None):
    """Temperature minimising mean NLL over the rows (optionally per group). Returns {group: T} or a float."""
    if key is None:
        return float(GRID[int(np.argmin([metrics(rows, float(t))["nll"] for t in GRID]))])
    groups = defaultdict(list)
    for r in rows: groups[key(r)].append(r)
    return {g: (float(GRID[int(np.argmin([metrics(rs, float(t))["nll"] for t in GRID]))]) if len(rs) >= 30 else 1.0) for g, rs in groups.items()}


def apply(rows, temps, key):
    out = []
    for r in rows:
        t = temps.get(key(r), 1.0) if isinstance(temps, dict) else temps
        p = np.array(r["p"]); z = np.log(np.maximum(p, 1e-9)) / t; p = np.exp(z - z.max()); p /= p.sum()
        out.append({**r, "p": p.tolist()})
    return out


def summary(rows):
    m = metrics(rows)
    return f"acc {m['acc']:.3f} brier {m['brier']:.3f} ece {m['ece']:.3f} conf-err {m['confident_error_rate']:.3f} cov@5% {m['coverage_at_5pct_error']:.2f} cov@1% {m['coverage_at_1pct_error']:.2f}"


def main():
    for name, run in RUNS.items():
        dev = [r for r in json.loads((ROOT / "runs" / run / "development/rows.json").read_text()) if r["variant"] == "clean"]
        ood = [r for r in json.loads((ROOT / "runs" / run / "transfer/rows.json").read_text()) if r["variant"] == "clean"]
        single = fit(dev); grouped = fit(dev, group_key)
        print(f"\n== {name}  (fitted on {len(dev)} in-distribution dev rows; applied to {len(ood)} out-of-domain dev rows)")
        print(f"   single T = {single:.2f}; grouped T = " + ", ".join(f"{t}/K{k}: {v:.2f}" for (t, k), v in sorted(grouped.items())))
        for label, rows in (("raw", ood), ("single T", apply(ood, single, group_key)), ("grouped T", apply(ood, grouped, group_key))):
            print(f"   OOD {label:10} {summary(rows)}")
        # by type, since the sign of miscalibration is what the two external studies said differs by type
        for t in ("noul", "choice", "score"):
            sub = [r for r in ood if r["type"] == t]
            if sub: print(f"     {t:6} raw {summary(sub)} | grouped {summary(apply(sub, grouped, group_key))}")


if __name__ == "__main__":
    main()
