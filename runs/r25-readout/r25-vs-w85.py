"""Report-only: a round-25 arm against round 23's confirmed 27b-k-w85, each served at its own pooled temperature
(the registered pool: transfer-r3 calibration's eight held-out sources + transfer-v9 MMLU-Pro, minus transfer rows),
on every panel of the rule (development reads) or of the tests stage (test reads) plus locked. Paired, record-clustered
bootstrap (kev.rounds.compare). Not a criterion of round 25.

    uv run python runs/r25-vs-w85.py <arm> [--stage tests] [--out runs/r25-readout/vs-w85-<arm>.json]
"""
import argparse, json, sys
from pathlib import Path

import os
from kev.rounds import ROOT as _ROOT, Pool, Side, arm_side, compare, panel_lengths, table
from kev.suite import write_json
ROOT = Path(os.environ.get("R25_ROOT", _ROOT))

W85 = {tag: f"runs/r23-27b-k-w85-{tag}" for tag in ("breadth", "tsheld", "hard", "devtools", "docs", "semif", "wanli2", "typesafe", "v9",
                                                   "r3test", "r3cal", "transfer4", "longdoc", "ood", "agentsood", "guardood")}
W85_TESTS = {tag: f"runs/r23c-27b-cand-{tag}" for tag in ("breadthtest", "tshtest", "hardtest", "devtest", "docs1test", "docs2", "longdoctest")}
W85_LOCKED = "runs/locked/kev-27b-r23-ungated/transfer"


def w85_side(spec, stage=None):
    t = spec["temperature"]
    pool = Pool([W85[r] for r in t["reads"]], {W85[r]: s for r, s in t["sources"].items()}, [W85["transfer4"]], t.get("ci"))
    dirs = {**W85, "transfer": W85["transfer4"]}
    if stage == "tests": dirs.update(W85_TESTS)
    if stage == "locked": dirs["locked"] = W85_LOCKED
    return Side(None, dirs, ROOT, spec.get("drop_ids", ()), pool, "/runs/r23-wise/27b-k-w85/checkpoint")


def strip(rule):
    return {**rule, "criteria": {}, "rank": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm"); ap.add_argument("--stage", default=None); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from kev.rounds import load
    spec = load(ROOT / "experiments/rounds/r25.json")
    rule = spec["confirm"][a.stage] if a.stage else spec["rule"]
    rule = strip(rule)
    rule["panels"] = {k: dict(v) for k, v in rule["panels"].items()}
    for p in rule["panels"].values(): p.pop("optional", None)   # every panel reported, none gates
    if not a.stage:   # locked-style short states, split: transfer-v4 development (the locked read's suite) and transfer-r3 test
        rule["panels"]["short_transfer4"] = {"reads": ["transfer"], "exclude_sources": ["emotion"], "metrics": ["acc", "brier", "confident_error_rate", "ece"]}
        rule["panels"]["short_r3test"] = {"reads": ["r3test"], "exclude_sources": ["emotion"], "metrics": ["acc", "brier", "confident_error_rate", "ece"]}
    cand = arm_side(spec, a.arm, ROOT, a.stage)
    ref = w85_side(spec, a.stage)
    out = compare(cand, ref, rule, lambda panel: panel_lengths(spec, panel))
    out = {"report_only": "round-25 arm vs round 23's confirmed 27b-k-w85; each at its own pooled temperature; not a round-25 criterion",
           "arm": a.arm, "reference": "27b-k-w85", "stage": a.stage or "rule (development)", "reference_temperature": ref.t, **out}
    path = Path(a.out or ROOT / f"runs/r25-readout/vs-w85-{a.arm}{'-' + a.stage if a.stage else ''}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, out)
    print(table(out) if "panels" in out else json.dumps(out, indent=1))
    print("->", path)


if __name__ == "__main__":
    main()
