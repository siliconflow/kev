"""Every number a model card prints for a release candidate and its parent, from committed rows, each served at the
temperature written into its head.pt: fitted on its own decision-v7 development rows (an arm with a `trial`), or, for a
checkpoint without a trial of its own (a blend, round 23), on the round's registered temperature pool (an arm with a
`pool`: {reads: {read dir: [sources] | null}, exclude: [read dirs]}, fitted by kev.rounds.pooled_temperature exactly as
scripts/calibrate_checkpoint.py wrote it). One JSON per release, the `source` that docs/claims.json points at. The release
spec (experiments/releases/<release>.json) names each arm's trial or pool and where each of its reads lives; comparisons
use kev.rounds' registered paired read. Numbers are whole-suite (minus drop_ids); a round's gated panels, with their
exclusions, are in its read-out and verdicts.

    uv run python scripts/release_numbers.py --release kev-4b-r8 --out runs/release/kev-4b-r8.json
"""
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kev.metrics import metrics, served_at, unknowable_report  # noqa: E402
from kev.rounds import Pool, paired, pooled_temperature, served_clean, temperature  # noqa: E402
from kev.suite import read_json, write_json  # noqa: E402

KEYS = ("n", "acc", "brier", "ece", "confident_error_rate", "coverage_at_5pct_error")
READS = ("docs1_dev", "docs1_test", "docs2", "long2", "r6test", "long3", "hard_dev", "devtools_dev", "hard_test", "devtools_test",
         "transfer_dev", "breadth_dev", "breadth_test", "r3test",
         "semif", "scienthoon", "wanli2", "typesafe")   # optional per release; the report keeps this order
# scienthoon (removed 2026-09-27), wanli2 and typesafe (removed 2026-09-30): no longer evals (kev.suite.REMOVED_SUITES); kept so the
# recorded releases reproduce from their committed rows
JEV_DOCS = "runs/jev-documents-v1/rows.json"   # Jev's documents-v1 development rows (research checkout); reported when present


def summary(rows):
    m = metrics(rows)
    return {k: m[k] for k in KEYS if k in m}


def arm_temperature(spec):
    """(temperature, the pool's fit report or None): the trial's development-rows fit, or the registered pool's."""
    if "pool" not in spec: return temperature(spec["trial"], "."), None
    pool = spec["pool"]
    t, fit = pooled_temperature(Pool(list(pool["reads"]), {d: s for d, s in pool["reads"].items() if s}, pool.get("exclude", [])), ".")
    return t, {"kind": "pool", **fit}


def arm(spec, drop):
    trial = spec.get("trial")
    t, pooled = arm_temperature(spec)
    rows = lambda path: [r for r in served_at(read_json(path), t) if r["id"] not in drop]
    read = {k: rows(f"{spec[k]}/rows.json") for k in READS if k in spec}
    locked = Path(spec["locked"])
    out = {**({"trial": trial} if trial else {"checkpoint": spec["checkpoint"]}), "temperature": t, **({"temperature_source": pooled} if pooled else {}),
           **({"decision_dev": summary(rows(Path(trial) / "development/rows.json")),
               "transfer_dev": summary(rows(Path(trial) / "transfer/rows.json")),
               "heldout_pairs_both_correct": read_json(Path(trial) / "result.json")["transfer"]["paired_flip"]["both_correct_rate"]} if trial else {}),
           "unknowable_share_at_0_9": unknowable_report(served_clean(read_json(f"{spec['v9']}/rows.json"), t))["share_at_0_9"],
           "mmlu_pro": read_json(f"{spec['v9']}/report.json")["tasks"]["mmlu_pro"]["acc"],
           "locked_transfer": summary(rows(locked / "transfer/rows.json")),
           **({"locked_decision": summary(rows(locked / "decision/rows.json"))} if (locked / "decision/rows.json").exists() else {}),   # only committed for some releases
           **{k: summary(read[k]) for k in read}}
    long = {k: [r for r in read[k] if r["source"] == "longstate"] for k in ("long2", "long3") if k in read}   # the buried questions the rule scores
    out.update({f"{k}_buried": summary(v) for k, v in long.items()})
    locked_rows = {"locked_transfer": rows(locked / "transfer/rows.json")}
    return out, {**read, **{f"{k}_buried": v for k, v in long.items()}, **locked_rows}


def main():
    releases = sorted(p.stem for p in (ROOT / "experiments/releases").glob("*.json"))
    ap = argparse.ArgumentParser(); ap.add_argument("--release", required=True, choices=releases); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    release = read_json(ROOT / f"experiments/releases/{a.release}.json")
    (cand, crows), (parent, prows) = (arm(release[k], set(release["drop_ids"])) for k in ("candidate", "parent"))
    report = {"release": a.release, "candidate": cand, "parent": parent,
              "paired_acc_delta": {k: paired(crows[k], prows[k], "acc") for k in crows if k in prows and (k != "locked_transfer" or release.get("paired_locked"))},
              **({"jev": {"docs1_dev": metrics([r for r in read_json(JEV_DOCS) if r["variant"] == "clean"])["acc"]}} if Path(JEV_DOCS).exists() else {})}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); write_json(Path(a.out), report)
    for k in [k for k in cand if isinstance(cand[k], dict) and "acc" in cand[k] and k in parent]:
        print(f"{k:16} parent {parent[k]['acc']:.3f} / {parent[k]['brier']:.3f}  ->  release {cand[k]['acc']:.3f} / {cand[k]['brier']:.3f}")
    pairs = lambda arm: round(arm["heldout_pairs_both_correct"], 3) if "heldout_pairs_both_correct" in arm else "-"
    print("T", round(parent["temperature"], 2), "->", round(cand["temperature"], 2), "| pairs", pairs(parent), "->", pairs(cand),
          "| MMLU-Pro", parent["mmlu_pro"], "->", cand["mmlu_pro"], "| unknowable", parent["unknowable_share_at_0_9"], "->", cand["unknowable_share_at_0_9"])


if __name__ == "__main__":
    main()
