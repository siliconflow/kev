"""Phase 2 read-out: Qwen3.5-based trials against the released Kev-4B / Kev-8B on the same frozen items.

    uv run python scripts/compare_q35.py

Prints, per trial: transfer accuracy with the record-clustered paired bootstrap against the matched released checkpoint,
the deadline family, held-out pairs, MMLU/PAWS retention, Brier, confident errors, dev accuracy; then the decision
criteria from PLAN.md ("Qwen3.5 port", section 7) evaluated mechanically.
"""
import json
from pathlib import Path

from kev.metrics import paired_bootstrap
from kev.suite import read_json

ROOT = Path(__file__).resolve().parents[1]
RELEASED = {"4b": ("Kev-4B", "v7-rc3/01-trial-1"), "9b": ("Kev-8B", "v7-final/00-trial-0")}
STUDIES = {"4b": ["q35-4b/00-trial-0", "q35-4b/01-trial-1", "q35-4b-s23/00-trial-0", "q35-4b-s23/01-trial-1"], "9b": ["q35-9b/00-trial-0", "q35-9b/01-trial-1"]}
JEV = read_json(ROOT / "runs/jev-transfer-v4/report.json")


def read(path):
    p = ROOT / "runs" / path
    if not (p / "result.json").exists(): return None
    if (p / "transfer" / "rows.json").exists(): rows = p / "transfer" / "rows.json"
    else: rows = next(p.glob("transfer*/rows.json"), None)
    r = read_json(p / "result.json"); t = r["transfer"]
    return {"dev": r["clean"]["acc"], "acc": t["clean"]["acc"], "brier": t["clean"]["brier"], "cerr": t["clean"]["confident_error_rate"], "pairs": t["paired_flip"]["both_correct_rate"],
            "deadline": t["tasks"]["contrastive_deadline"]["acc"], "mmlu": t["tasks"]["mmlu"]["acc"], "paws": t["tasks"]["paws"]["acc"], "emotion": t["tasks"]["emotion"]["acc"],
            "cov5": t["clean"].get("coverage_at_5pct_error"), "rows": rows, "seed": r["provenance"]["config"].get("seed"), "wall": r.get("wall_seconds")}


def main():
    for size, trials in STUDIES.items():
        name, ref_path = RELEASED[size]; ref = read(ref_path)
        print(f"\n== Qwen3.5-{size.upper()} trials vs {name} (transfer-v4 dev, 764 records; Jev {JEV['clean']['acc']:.3f})")
        print(f"   {'trial':26} {'seed':>4} {'dev':>6} {'transfer':>8} {'paired delta [95% CI]':>24} {'deadline':>8} {'pairs':>6} {'mmlu':>5} {'paws':>5} {'emo':>5} {'brier':>6} {'cerr':>5}")
        print(f"   {name:26} {str(ref['seed']):>4} {ref['dev']:6.3f} {ref['acc']:8.3f} {'reference':>24} {ref['deadline']:8.2f} {ref['pairs']:6.2f} {ref['mmlu']:5.2f} {ref['paws']:5.2f} {ref['emotion']:5.2f} {ref['brier']:6.3f} {ref['cerr']:5.3f}")
        verdicts = []
        for t in trials:
            r = read(t)
            if r is None: print(f"   {t:26} (not finished)"); continue
            b = paired_bootstrap(read_json(r["rows"]), read_json(ref["rows"]), metric="acc")
            ci = f"{b['macro_acc_delta']:+.3f} [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}]"
            print(f"   {t:26} {str(r['seed']):>4} {r['dev']:6.3f} {r['acc']:8.3f} {ci:>24} {r['deadline']:8.2f} {r['pairs']:6.2f} {r['mmlu']:5.2f} {r['paws']:5.2f} {r['emotion']:5.2f} {r['brier']:6.3f} {r['cerr']:5.3f}")
            verdicts.append({"trial": t, "beats_ref_ci": b["ci95"][0] > 0, "acc_ge_ref": r["acc"] >= ref["acc"], "deadline_ge_0_75": r["deadline"] >= 0.75,
                             "mmlu_paws_retained": r["mmlu"] >= ref["mmlu"] - 0.01 and r["paws"] >= ref["paws"] - 0.01, "brier_le_0_34": r["brier"] <= 0.34, "pairs_ge_0_70": r["pairs"] >= 0.70})
        for v in verdicts:
            ship = v["beats_ref_ci"] and v["acc_ge_ref"] and v["deadline_ge_0_75"] and v["mmlu_paws_retained"] and v["brier_le_0_34"]
            print(f"   criteria {v['trial']}: " + ", ".join(f"{k}={'yes' if val else 'no'}" for k, val in v.items() if k != "trial") + f" -> {'SHIP' if ship else 'no'}")


if __name__ == "__main__":
    main()
