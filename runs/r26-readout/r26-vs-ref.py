"""Report-only: a round-26 arm against a reference, each served at its own pooled temperature (the registered pool:
transfer-r3 calibration's eight held-out sources + transfer-v9 MMLU-Pro, minus transfer rows), on every panel of the rule
(development reads) or of the tests stage (test reads), every panel reported and none gating, plus short states split into
transfer-v4 development and transfer-r3 test. Paired, record-clustered bootstrap (kev.rounds.compare). Not a round-26 criterion.

References:
  w85      round 23's confirmed 27b-k-w85 (rows in /tmp/kev-r23run/runs: r23-27b-k-w85-*, tests r23c-27b-cand-*)
  r25      round 25's lr 1e-6 arm at the same fraction of its run (27b-lr1e6-s25/s50/s75/final; rows in /tmp/kev-r25run/runs)

    uv run python runs/r26-readout/r26-vs-ref.py <arm> --ref w85|r25 [--stage tests]
"""
import argparse, json, os
from pathlib import Path

from kev.rounds import ROOT, Pool, Side, arm_side, compare, load, panel_lengths, table
from kev.suite import write_json

R23 = Path(os.environ.get("R23_ROOT", "/tmp/kev-r23run"))
R25 = Path(os.environ.get("R25_ROOT", "/tmp/kev-r25run"))
W85 = {tag: f"runs/r23-27b-k-w85-{tag}" for tag in ("breadth", "tsheld", "hard", "devtools", "docs", "semif", "wanli2", "typesafe", "v9",
                                                   "r3test", "r3cal", "transfer4", "longdoc", "ood", "agentsood", "guardood")}
W85_TESTS = {tag: f"runs/r23c-27b-cand-{tag}" for tag in ("breadthtest", "tshtest", "hardtest", "devtest", "docs1test", "docs2", "longdoctest")}


def w85_side(spec, stage=None):
    t = spec["temperature"]
    pool = Pool([W85[r] for r in t["reads"]], {W85[r]: s for r, s in t["sources"].items()}, [W85["transfer4"]], t.get("ci"))
    dirs = {**W85, "transfer": W85["transfer4"]}
    if stage == "tests": dirs.update(W85_TESTS)
    return Side(None, dirs, R23, spec.get("drop_ids", ()), pool, "/runs/r23-wise/27b-k-w85/checkpoint")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm"); ap.add_argument("--ref", choices=("w85", "r25"), required=True)
    ap.add_argument("--stage", default=None); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    spec = load(ROOT / "experiments/rounds/r26.json")
    rule = dict(spec["confirm"][a.stage] if a.stage else spec["rule"])
    rule["criteria"], rule["rank"] = {}, []
    rule["panels"] = {k: {kk: vv for kk, vv in v.items() if kk != "optional"} for k, v in rule["panels"].items()}
    if not a.stage:
        rule["panels"]["short_transfer4"] = {"reads": ["transfer"], "exclude_sources": ["emotion"], "metrics": ["acc", "brier", "confident_error_rate", "ece"]}
        rule["panels"]["short_r3test"] = {"reads": ["r3test"], "exclude_sources": ["emotion"], "metrics": ["acc", "brier", "confident_error_rate", "ece"]}
    cand = arm_side(spec, a.arm, ROOT, a.stage)
    if a.ref == "w85":
        ref, name = w85_side(spec, a.stage), "27b-k-w85 (round 23, confirmed)"
    else:
        if a.stage: raise SystemExit("round 25's arms have no test reads")
        r25 = load(R25 / "experiments/rounds/r25.json")
        ref, name = arm_side(r25, a.arm, R25), f"round 25 {a.arm}"
    out = compare(cand, ref, rule, lambda panel: panel_lengths(spec, panel))
    out = {"report_only": f"round-26 arm vs {name}; each at its own pooled temperature; not a round-26 criterion",
           "arm": a.arm, "reference": name, "stage": a.stage or "rule (development)", "reference_temperature": ref.t, **out}
    path = Path(a.out or ROOT / f"runs/r26-readout/vs-{a.ref}-{a.arm}{'-' + a.stage if a.stage else ''}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, out)
    print(table(out) if "panels" in out else json.dumps(out, indent=1))
    print("->", path)


if __name__ == "__main__":
    main()
